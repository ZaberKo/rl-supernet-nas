from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from evox.core import Problem
from evox.operators.selection import non_dominate_rank
from evox.workflows import EvalMonitor, StdWorkflow
from omegaconf import DictConfig, OmegaConf

from checkpoint_utils import (
    build_policy_from_checkpoint,
    load_checkpoint,
    load_critic_from_checkpoint,
)
from ea_codec import GeneCodec
from env_utils import EVAL_SEED_OFFSET
from nsga2_search import DiscreteNSGA2
from ppo_utils import (
    PolicySupernet,
    actor_head_parameters,
    build_sb3_critic_model,
    collect_candidate_rollout,
    configure_actor_optimizer,
    count_parameters,
    create_ema_policy,
    critic_update,
    critic_warmup,
    evaluate_actor_subnet,
    fixed_arch_actor_update,
    make_vec_env_from_ppo_config,
    parse_schedule_value,
    resolve_device,
    update_actor_optimizer_learning_rate,
    update_ema_model,
    update_optimizer_learning_rate,
)
from setup_utils import (
    add_ppo_config_args,
    build_run_config,
    load_ppo_config,
    ppo_config_to_dict,
    set_global_seeds,
)
from supernet_backbone import ArchConfig, SearchSpace
from trajectory_data import (
    DynamicsRolloutBuffer,
)
from wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb,
    update_wandb_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 NSGA-II subnet search initialized from a policy supernet.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument(
        "--output_dir",
        default="runs/stage2_ea_search",
        help="Directory for NSGA-II records, search space, and manifest.",
    )
    parser.add_argument(
        "--supernet_checkpoint",
        default="runs/stage1_policy_supernet/policy_supernet_best.pt",
        help="Stage 1 policy-supernet checkpoint used to initialize subnet candidates.",
    )
    parser.add_argument(
        "--population_size", type=int, default=6, help="NSGA-II population size."
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of NSGA-II generations to evaluate.",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=1,
        help="Torch multiprocessing workers for parallel subnet evaluation.",
    )
    parser.add_argument(
        "--critic_warmup_timesteps",
        type=int,
        default=0,
        help="Critic-only warmup timesteps on current subnet rollouts before actor PPO finetune; 0 disables warmup.",
    )
    parser.add_argument(
        "--save_full_history",
        action="store_true",
        help="Store full EvoX monitor history in memory for debugging.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to the stage name (for W&B and manifests).",
    )
    args = parser.parse_args()
    if args.critic_warmup_timesteps < 0:
        raise ValueError("critic_warmup_timesteps must be non-negative.")
    args.eval_call_seed_stride = 10_000
    args.candidate_seed_stride = 100

    args.mp_start_method = "spawn"
    args.worker_torch_threads = 1
    return args


def build_initial_population(
    args: argparse.Namespace, codec: GeneCodec
) -> list[list[int]]:
    if args.population_size <= 0:
        raise ValueError("population_size must be positive.")
    population: list[list[int]] = [codec.max_gene()]
    while len(population) < args.population_size:
        population.append(codec.sample_gene())
    return population[: args.population_size]


def tensor_to_genes(pop: torch.Tensor) -> list[list[int]]:
    return [[round(value) for value in row.tolist()] for row in pop.detach().cpu()]


def build_generation_records(
    generation: int,
    pop: torch.Tensor,
    fit: torch.Tensor,
    codec: GeneCodec,
    problem: NewPolicySubnetProblem,
) -> list[dict[str, Any]]:
    genes = tensor_to_genes(pop)
    fit_cpu = fit.detach().cpu()
    rank = non_dominate_rank(fit_cpu)
    records = []
    worker_records = {tuple(record["gene"]): record for record in problem.last_records}
    for index, gene in enumerate(genes):
        worker_record = worker_records.get(tuple(gene), {})
        objectives = [float(value) for value in fit_cpu[index].tolist()]
        pareto_rank = int(rank[index].item())
        records.append(
            {
                "gen": generation,
                "generation": generation,
                "individual_index": index,
                "candidate_index": index,
                "gene": gene,
                "arch": codec.gene_to_arch(gene).to_dict(),
                "objectives": {
                    "negative_return": objectives[0],
                    "policy_params": objectives[1],
                },
                "return": -objectives[0],
                "policy_params": objectives[1],
                "pareto_rank": pareto_rank,
                "is_pareto": bool(pareto_rank == 0),
                "is_pareto_front": bool(pareto_rank == 0),
                "worker_record": worker_record,
            }
        )
    return records


def write_generation(records_path: Path, records: list[dict[str, Any]]) -> None:
    with records_path.open("a") as records_file:
        for record in records:
            records_file.write(json.dumps(record) + "\n")


def generation_summary(
    generation: int, records: list[dict[str, Any]], cache_hits: int
) -> dict[str, float | int]:
    if not records:
        return {
            "gen": generation,
            "candidates": 0,
            "pareto": 0,
            "best_return": 0.0,
            "min_params": 0.0,
            "cache_hits": cache_hits,
        }
    return {
        "gen": generation,
        "candidates": len(records),
        "pareto": sum(1 for record in records if bool(record["is_pareto"])),
        "best_return": max(float(record["return"]) for record in records),
        "min_policy_params": min(float(record["policy_params"]) for record in records),
        "cache_hits": cache_hits,
    }


def format_generation_log(
    generation: int, records: list[dict[str, Any]], cache_hits: int
) -> str:
    summary = generation_summary(generation, records, cache_hits)
    return (
        f"gen={int(summary['gen'])} candidates={int(summary['candidates'])} "
        f"pareto={int(summary['pareto'])} "
        f"best_return={float(summary['best_return']):.6g} "
        f"min_policy_params={float(summary['min_policy_params']):.0f} "
        f"cache_hits={int(summary['cache_hits'])}"
    )


def log_generation(
    log_path: Path,
    generation: int,
    records: list[dict[str, Any]],
    cache_hits: int,
    wandb_run: Any = None,
) -> None:
    message = format_generation_log(generation, records, cache_hits)
    print(message, flush=True)
    with log_path.open("a") as log_file:
        log_file.write(message + "\n")
    summary = generation_summary(generation, records, cache_hits)
    log_wandb(wandb_run, summary, step=generation)


def finetune_and_evaluate_candidate(
    args: argparse.Namespace,
    ppo_config: Any,
    arch_config: ArchConfig,
    train_seed: int,
    eval_seed: int,
) -> dict[str, Any]:
    set_global_seeds(train_seed)
    device = resolve_device(str(ppo_config.device))
    search_space = SearchSpace()
    checkpoint = load_checkpoint(args.supernet_checkpoint, map_location=device)

    train_env = make_vec_env_from_ppo_config(
        ppo_config, seed=train_seed, n_envs=ppo_config.train_n_envs
    )
    eval_env = make_vec_env_from_ppo_config(
        ppo_config, seed=eval_seed, n_envs=ppo_config.eval_n_envs
    )
    try:
        policy = build_policy_from_checkpoint(
            ppo_config=ppo_config,
            env=train_env,
            search_space=search_space,
            checkpoint=checkpoint,
            device=device,
        )
        policy.set_active_arch(arch_config)

        z_dyn_coef = float(ppo_config.z_dyn_coef)
        ema_policy: PolicySupernet | None = None
        if z_dyn_coef > 0.0:
            ema_policy = create_ema_policy(
                policy,
                device,
                checkpoint_ema_state_dict=checkpoint.get("ema_policy_state_dict"),
            )

        actor_lr_schedule = parse_schedule_value(ppo_config.policy_head_lr)
        backbone_lr_schedule = parse_schedule_value(ppo_config.policy_backbone_lr)
        critic_lr_schedule = parse_schedule_value(ppo_config.critic_lr)
        clip_range_schedule = parse_schedule_value(ppo_config.clip_range)
        actor_optimizer = configure_actor_optimizer(
            policy=policy,
            head_lr=actor_lr_schedule,
            backbone_lr=backbone_lr_schedule,
        )

        critic_model = build_sb3_critic_model(
            ppo_config=ppo_config,
            env=train_env,
            learning_rate=critic_lr_schedule,
        )
        loaded_critic = load_critic_from_checkpoint(critic_model, checkpoint)

        rollout_buffer = DynamicsRolloutBuffer(
            buffer_size=int(ppo_config.n_steps),
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            device=device,
            gae_lambda=float(ppo_config.gae_lambda),
            gamma=float(ppo_config.gamma),
            n_envs=int(train_env.num_envs),
        )

        observation = train_env.reset()
        episode_starts = np.ones((train_env.num_envs,), dtype=np.bool_)
        critic_warmup_actual_timesteps = 0
        critic_warmup_metrics: dict[str, float] = {}
        if int(args.critic_warmup_timesteps) > 0:
            (
                observation,
                episode_starts,
                critic_warmup_actual_timesteps,
                critic_warmup_metrics,
            ) = critic_warmup(
                policy=policy,
                critic_model=critic_model,
                env=train_env,
                arch=arch_config,
                rollout_buffer=rollout_buffer,
                initial_observation=observation,
                initial_episode_starts=episode_starts,
                target_timesteps=int(args.critic_warmup_timesteps),
                ppo_config=ppo_config,
                critic_lr_schedule=critic_lr_schedule,
                device=device,
            )

        total_timesteps = 0
        last_metrics: dict[str, float] = {}
        target_timesteps = max(0, int(ppo_config.total_timesteps))
        target_kl = ppo_config.target_kl

        while total_timesteps < target_timesteps:
            progress_remaining = 1.0 - float(total_timesteps) / float(
                max(1, target_timesteps)
            )
            actor_lr = (
                float(actor_lr_schedule(progress_remaining))
                if callable(actor_lr_schedule)
                else float(actor_lr_schedule)
            )
            backbone_lr = (
                float(backbone_lr_schedule(progress_remaining))
                if callable(backbone_lr_schedule)
                else float(backbone_lr_schedule)
            )
            critic_lr = (
                float(critic_lr_schedule(progress_remaining))
                if callable(critic_lr_schedule)
                else float(critic_lr_schedule)
            )
            clip_range = (
                float(clip_range_schedule(progress_remaining))
                if callable(clip_range_schedule)
                else float(clip_range_schedule)
            )
            update_actor_optimizer_learning_rate(actor_optimizer, actor_lr, backbone_lr)
            update_optimizer_learning_rate(critic_model.policy.optimizer, critic_lr)

            observation, episode_starts, rollout_metrics = collect_candidate_rollout(
                policy=policy,
                critic_model=critic_model,
                env=train_env,
                arch=arch_config,
                rollout_buffer=rollout_buffer,
                initial_observation=observation,
                initial_episode_starts=episode_starts,
                n_steps=int(ppo_config.n_steps),
                gamma=float(ppo_config.gamma),
                device=device,
            )
            total_timesteps += int(ppo_config.n_steps) * int(train_env.num_envs)
            actor_metrics = fixed_arch_actor_update(
                policy=policy,
                actor_optimizer=actor_optimizer,
                rollout_buffer=rollout_buffer,
                arch=arch_config,
                action_space=train_env.action_space,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                clip_range=clip_range,
                normalize_advantage=bool(ppo_config.normalize_advantage),
                ent_coef=float(ppo_config.ent_coef),
                max_grad_norm=float(ppo_config.max_grad_norm),
                target_kl=target_kl,
                ema_policy=ema_policy,
                z_dyn_coef=z_dyn_coef,
            )
            if ema_policy is not None and z_dyn_coef > 0.0:
                update_ema_model(ema_policy, policy, tau=float(ppo_config.ema_tau))
            critic_metrics = critic_update(
                critic_model=critic_model,
                optimizer=critic_model.policy.optimizer,
                rollout_buffer=rollout_buffer,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                max_grad_norm=float(ppo_config.max_grad_norm),
            )
            last_metrics = {
                "progress_remaining": float(progress_remaining),
                "actor_lr": float(actor_lr),
                "critic_lr": float(critic_lr),
                "clip_range": float(clip_range),
                **rollout_metrics,
                **actor_metrics,
                **critic_metrics,
            }

        eval_episodes = ppo_config.eval_episodes
        if eval_episodes <= 0:
            raise ValueError(
                "ppo.eval_episodes must be positive for search fitness evaluation."
            )
        eval_metrics = evaluate_actor_subnet(
            policy=policy,
            eval_env=eval_env,
            arch=arch_config,
            n_eval_episodes=eval_episodes,
            deterministic=ppo_config.eval_deterministic,
            device=device,
            train_env=train_env,
        )
        policy_backbone_params = int(policy.backbone.elastic_num_params)
        policy_head_params = count_parameters(actor_head_parameters(policy))
        policy_params = int(policy.elastic_num_params)
        trainable_policy_params = policy_head_params
        if any(p.requires_grad for p in policy.backbone.parameters()):
            trainable_policy_params += policy_backbone_params

        return {
            "return": float(eval_metrics["ep_return"]),
            "return_std": float(eval_metrics["ep_return_std"]),
            "ep_return": float(eval_metrics["ep_return"]),
            "ep_return_std": float(eval_metrics["ep_return_std"]),
            "ep_length": float(eval_metrics["ep_length"]),
            "ep_length_std": float(eval_metrics["ep_length_std"]),
            "policy_backbone_params": policy_backbone_params,
            "policy_head_params": policy_head_params,
            "policy_params": policy_params,
            "trainable_policy_params": trainable_policy_params,
            "total_timesteps": int(total_timesteps),
            "candidate_timesteps": int(target_timesteps),
            "critic_warmup_configured_timesteps": int(args.critic_warmup_timesteps),
            "critic_warmup_actual_timesteps": int(critic_warmup_actual_timesteps),
            "total_env_timesteps": int(
                critic_warmup_actual_timesteps + total_timesteps
            ),
            "loaded_critic": bool(loaded_critic),
            "critic_warmup_metrics": critic_warmup_metrics,
            "finetune_metrics": last_metrics,
        }
    finally:
        train_env.close()
        eval_env.close()


@dataclass(frozen=True)
class NewPolicySubnetEvalConfig:
    args: dict[str, Any]
    ppo_config: dict[str, Any]
    arch_config: dict[str, Any]
    gene: list[int]
    train_seed: int
    eval_seed: int


def _evaluate_new_policy_subnet_worker(
    config: NewPolicySubnetEvalConfig,
) -> dict[str, Any]:
    worker_threads = int(config.args.get("worker_torch_threads", 1))
    if worker_threads > 0:
        torch.set_num_threads(worker_threads)
    args = argparse.Namespace(**config.args)
    ppo_config = OmegaConf.create(config.ppo_config)
    arch_config = ArchConfig.from_dict(config.arch_config)
    result = finetune_and_evaluate_candidate(
        args=args,
        ppo_config=ppo_config,
        arch_config=arch_config,
        train_seed=config.train_seed,
        eval_seed=config.eval_seed,
    )
    return {
        "gene": config.gene,
        "arch_config": config.arch_config,
        "train_seed": int(config.train_seed),
        "eval_seed": int(config.eval_seed),
        "pid": os.getpid(),
        **result,
    }


class NewPolicySubnetProblem(Problem):
    """EvoX problem that evaluates policy-supernet subnet genes by PPO fine-tuning."""

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
    ) -> NewPolicySubnetEvalConfig:
        train_seed = (
            int(self.ppo_config_dict["seed"])
            + self.eval_call_index * int(self.args_dict["eval_call_seed_stride"])
            + candidate_index * int(self.args_dict["candidate_seed_stride"])
        )
        eval_seed = int(self.ppo_config_dict["seed"]) + EVAL_SEED_OFFSET
        return NewPolicySubnetEvalConfig(
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
        pending: list[NewPolicySubnetEvalConfig] = []
        pending_indices: list[int] = []
        self.last_cache_hits = 0

        for index, gene in enumerate(genes):
            self.codec.validate_gene(gene)
            key = tuple(gene)
            if key in self.cache:
                records[index] = copy.deepcopy(self.cache[key])
                self.last_cache_hits += 1
            else:
                pending.append(self._make_eval_config(gene, index))
                pending_indices.append(index)

        if pending:
            if self.eval_workers <= 1:
                evaluated = [
                    _evaluate_new_policy_subnet_worker(config) for config in pending
                ]
            else:
                pool = self._ensure_pool()
                evaluated = pool.map(_evaluate_new_policy_subnet_worker, pending)
            for index, record in zip(pending_indices, evaluated, strict=True):
                key = tuple(record["gene"])
                self.cache[key] = copy.deepcopy(record)
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


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    stage_name = (
        f"stage2_ea_search_{args.suffix}"
        if getattr(args, "suffix", "")
        else "stage2_ea_search"
    )
    wandb_run = init_wandb_run(stage_name, run_config, output_dir)
    search_space = SearchSpace()
    codec = GeneCodec(search_space)
    (output_dir / "search_space.json").write_text(
        json.dumps(search_space.to_dict(), indent=2)
    )

    lower_bounds, upper_bounds = codec.gene_bounds()
    initial_population = build_initial_population(args, codec)
    algorithm = DiscreteNSGA2(
        pop_size=args.population_size,
        n_objs=2,
        lb=torch.tensor(lower_bounds, dtype=torch.float32),
        ub=torch.tensor(upper_bounds, dtype=torch.float32),
        device=torch.device("cpu"),
        initial_population=torch.tensor(initial_population, dtype=torch.float32),
    )
    problem = NewPolicySubnetProblem(args=args, ppo_config=ppo_config, codec=codec)
    monitor = EvalMonitor(
        multi_obj=True,
        full_fit_history=args.save_full_history,
        full_sol_history=args.save_full_history,
        full_pop_history=args.save_full_history,
        device=torch.device("cpu"),
        history_device=torch.device("cpu"),
    )
    workflow = StdWorkflow(
        algorithm, problem, monitor=monitor, device=torch.device("cpu")
    )

    records_path = output_dir / "nsga2_records.jsonl"
    log_path = output_dir / "search.log"
    for path in (records_path, log_path):
        if path.exists():
            path.unlink()

    all_records: list[dict[str, Any]] = []
    try:
        workflow.init_step()
        latest_pop = monitor.get_latest_solution()
        latest_fit = monitor.get_latest_fitness()
        records = build_generation_records(0, latest_pop, latest_fit, codec, problem)
        write_generation(records_path, records)
        log_generation(log_path, 0, records, problem.last_cache_hits, wandb_run)
        all_records.extend(records)

        for generation in range(1, args.generations):
            workflow.step()
            latest_pop = monitor.get_latest_solution()
            latest_fit = monitor.get_latest_fitness()
            records = build_generation_records(
                generation, latest_pop, latest_fit, codec, problem
            )
            write_generation(records_path, records)
            log_generation(
                log_path, generation, records, problem.last_cache_hits, wandb_run
            )
            all_records.extend(records)
    finally:
        problem.close()

    final_pop = monitor.get_latest_solution().detach().cpu()
    final_fit = monitor.get_latest_fitness().detach().cpu()
    final_records = build_generation_records(
        max(0, args.generations - 1), final_pop, final_fit, codec, problem
    )
    pareto_records = [record for record in final_records if record["is_pareto_front"]]
    manifest = {
        "stage": stage_name,
        "records": str(records_path),
        "log": str(log_path),
        "search_space": str(output_dir / "search_space.json"),
        "supernet_checkpoint": str(args.supernet_checkpoint),
        "objectives": ["negative_return", "policy_params"],
        "fitness_procedure": "load_policy_supernet_then_ppo_finetune_then_eval_return",
        "pareto_front": pareto_records,
        "final_population": final_records,
        "num_logged_records": len(all_records),
        "cache_size": len(problem.cache),
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    update_wandb_summary(
        wandb_run,
        {
            "num_logged_records": len(all_records),
            "cache_size": len(problem.cache),
            "final_pareto_count": len(pareto_records),
        },
    )

    finish_wandb_run(wandb_run)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
