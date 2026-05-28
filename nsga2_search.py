from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.multiprocessing as mp
from evox.algorithms.mo import NSGA2
from evox.core import Mutable, Problem
from omegaconf import DictConfig, OmegaConf

from ea_codec import GeneCodec
from env_utils import EVAL_SEED_OFFSET
from sb3_nas_policy import finetune_and_evaluate_arch
from setup_utils import ppo_config_to_dict
from supernet_backbone import ArchConfig


def random_integer_population(
    *, pop_size: int, lb: torch.Tensor, ub: torch.Tensor
) -> torch.Tensor:
    span = (ub - lb + 1).clamp_min(1)
    random_values = torch.rand(pop_size, lb.numel(), device=lb.device)
    return torch.floor(random_values * span + lb)


def uniform_integer_crossover(x: torch.Tensor, pro_c: float = 1.0) -> torch.Tensor:
    offspring = x.round().clone()
    pair_count = offspring.shape[0] // 2
    if pair_count == 0:
        return offspring

    parent1 = offspring[:pair_count]
    parent2 = offspring[pair_count : pair_count * 2]
    pair_mask = torch.rand(pair_count, 1, device=x.device) < pro_c
    gene_mask = torch.rand(pair_count, x.shape[1], device=x.device) < 0.5
    swap_mask = pair_mask & gene_mask

    parent1_values = parent1.clone()
    parent2_values = parent2.clone()
    parent1[swap_mask] = parent2_values[swap_mask]
    parent2[swap_mask] = parent1_values[swap_mask]
    return offspring


def random_reset_integer_mutation(
    x: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    mutation_prob: float | None = None,
) -> torch.Tensor:
    offspring = x.round().clone()
    if mutation_prob is None:
        mutation_prob = 1.0 / max(1, offspring.shape[1])

    mutation_mask = torch.rand_like(offspring, dtype=torch.float32) < mutation_prob
    sampled = random_integer_population(
        pop_size=offspring.shape[0],
        lb=lb.to(offspring.device),
        ub=ub.to(offspring.device),
    )
    offspring[mutation_mask] = sampled[mutation_mask]
    return offspring


class DiscreteNSGA2(NSGA2):
    """NSGA-II configured for fixed-length integer NAS genes."""

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        device: torch.device | None = None,
        crossover_prob: float = 1.0,
        mutation_prob: float | None = None,
        initial_population: torch.Tensor | None = None,
    ) -> None:
        lb = lb.to(dtype=torch.float32)
        ub = ub.to(dtype=torch.float32)
        super().__init__(
            pop_size=pop_size,
            n_objs=n_objs,
            lb=lb,
            ub=ub,
            crossover_op=lambda x: uniform_integer_crossover(x, pro_c=crossover_prob),
            mutation_op=lambda x, lower, upper: random_reset_integer_mutation(
                x, lower, upper, mutation_prob=mutation_prob
            ),
            device=device,
        )
        population = random_integer_population(
            pop_size=pop_size, lb=self.lb, ub=self.ub
        )
        if initial_population is not None and initial_population.numel() > 0:
            initial_population = initial_population.to(
                device=self.lb.device, dtype=torch.float32
            ).round()
            count = min(pop_size, initial_population.shape[0])
            population[:count] = initial_population[:count]
        self.pop = Mutable(population)


@dataclass(frozen=True)
class SubnetEvalConfig:
    args: dict[str, Any]
    ppo_config: dict[str, Any]
    arch_config: dict[str, Any]
    gene: list[int]
    train_seed: int
    eval_seed: int


def _evaluate_subnet_worker(config: SubnetEvalConfig) -> dict[str, Any]:
    worker_threads = int(config.args.get("worker_torch_threads", 1))
    if worker_threads > 0:
        torch.set_num_threads(worker_threads)
    args = argparse.Namespace(**config.args)
    ppo_config = OmegaConf.create(config.ppo_config)
    arch_config = ArchConfig.from_dict(config.arch_config)
    model, mean_return, std_return = finetune_and_evaluate_arch(
        args=args,
        ppo_config=ppo_config,
        arch_config=arch_config,
        train_seed=config.train_seed,
        eval_seed=config.eval_seed,
    )
    backbone = model.policy.features_extractor.backbone
    return {
        "gene": config.gene,
        "arch_config": config.arch_config,
        "return": float(mean_return),
        "return_std": float(std_return),
        "policy_backbone_params": int(backbone.elastic_num_params),
        "policy_params": int(
            sum(parameter.numel() for parameter in model.policy.parameters())
        ),
        "train_seed": config.train_seed,
        "eval_seed": config.eval_seed,
        "pid": os.getpid(),
    }


class RLSubnetProblem(Problem):
    """EvoX problem that evaluates subnet genes by PPO fine-tuning.

    Objectives are minimized as `[-mean_return, num_params]`.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        ppo_config: DictConfig,
        codec: GeneCodec,
    ) -> None:
        super().__init__()
        self.args_dict = vars(args).copy()
        self.ppo_config_dict = ppo_config_to_dict(ppo_config)
        self.codec = codec
        self.eval_workers = max(1, int(args.eval_workers))
        self.mp_start_method = args.mp_start_method
        self.eval_call_index = 0
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.last_records: list[dict[str, Any]] = []
        self.last_cache_hits = 0
        self._pool = None

    def _make_eval_config(
        self, gene: list[int], candidate_index: int
    ) -> SubnetEvalConfig:
        train_seed = (
            int(self.ppo_config_dict["seed"])
            + self.eval_call_index * int(self.args_dict["eval_call_seed_stride"])
            + candidate_index * int(self.args_dict["candidate_seed_stride"])
        )
        eval_seed = int(self.ppo_config_dict["seed"]) + EVAL_SEED_OFFSET
        return SubnetEvalConfig(
            args=self.args_dict,
            ppo_config=self.ppo_config_dict,
            arch_config=self.codec.gene_to_arch(gene).to_dict(),
            gene=gene,
            train_seed=train_seed,
            eval_seed=eval_seed,
        )

    def _ensure_pool(self):
        if self.eval_workers <= 1:
            return None
        if self._pool is None:
            context = mp.get_context(self.mp_start_method)
            self._pool = context.Pool(processes=self.eval_workers)
        return self._pool

    @torch.compiler.disable
    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        genes = [[round(value) for value in row.tolist()] for row in pop.detach().cpu()]
        records: list[dict[str, Any] | None] = [None] * len(genes)
        pending: list[SubnetEvalConfig] = []
        pending_indices: list[int] = []
        self.last_cache_hits = 0

        for index, gene in enumerate(genes):
            self.codec.validate_gene(gene)
            key = tuple(gene)
            if key in self.cache:
                records[index] = dict(self.cache[key])
                self.last_cache_hits += 1
            else:
                pending.append(self._make_eval_config(gene, index))
                pending_indices.append(index)

        if pending:
            if self.eval_workers <= 1:
                evaluated = [_evaluate_subnet_worker(config) for config in pending]
            else:
                pool = self._ensure_pool()
                evaluated = pool.map(_evaluate_subnet_worker, pending)
            for index, record in zip(pending_indices, evaluated, strict=True):
                key = tuple(record["gene"])
                self.cache[key] = dict(record)
                records[index] = record

        self.last_records = [record for record in records if record is not None]
        self.eval_call_index += 1
        objectives = [
            [-float(record["return"]), float(record["policy_params"])]
            for record in self.last_records
        ]
        return torch.tensor(objectives, dtype=torch.float32, device=pop.device)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
