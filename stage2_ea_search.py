from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from evox.core import Problem
from evox.operators.selection import non_dominate_rank
from evox.workflows import EvalMonitor, StdWorkflow
from omegaconf import DictConfig, OmegaConf
from ray.util.actor_pool import ActorPool

from checkpoint_utils import (
    build_policy_from_checkpoint,
    load_checkpoint,
    load_critic_from_checkpoint,
)
from discrete_nsga2 import DiscreteNSGA2
from ea_codec import GeneCodec
from env_utils import EVAL_SEED_OFFSET, make_vec_env_from_ppo_config
from ppo_utils import (
    FixedPolicySubnet,
    build_sb3_critic_model,
    collect_candidate_rollout,
    configure_actor_optimizer,
    critic_update,
    critic_warmup,
    evaluate_actor_subnet,
    fixed_arch_actor_update,
    update_actor_optimizer_learning_rate,
    update_ema_model,
    update_optimizer_learning_rate,
)
from setup_utils import (
    add_ppo_config_args,
    build_run_config,
    compute_ray_worker_config,
    load_ppo_config,
    parse_schedule_value,
    ppo_config_to_dict,
    resolve_device,
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
        "--workers",
        type=int,
        default=1,
        help="Total number of Ray actors for parallel subnet evaluation.",
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


# ---------------------------------------------------------------------------
# Standalone candidate evaluation function (used by both Actor and local)
# ---------------------------------------------------------------------------


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
        supernet = build_policy_from_checkpoint(
            ppo_config=ppo_config,
            env=train_env,
            search_space=search_space,
            checkpoint=checkpoint,
            device=device,
        )
        supernet.set_sample_config(arch_config)
        policy = supernet.get_active_subnet().to(device)

        z_dyn_coef = float(ppo_config.z_dyn_coef)
        ema_policy: FixedPolicySubnet | None = None
        if z_dyn_coef > 0.0:
            ema_state_dict = checkpoint.get("ema_policy_state_dict")
            if ema_state_dict is not None:
                supernet.load_state_dict(ema_state_dict, strict=True)
                supernet.set_sample_config(arch_config)
                ema_policy = supernet.get_active_subnet().to(device)
            else:
                ema_policy = copy.deepcopy(policy)
            for parameter in ema_policy.parameters():
                parameter.requires_grad_(False)
        del supernet

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
        )
        critic_lr = (
            float(critic_lr_schedule(1.0))
            if callable(critic_lr_schedule)
            else float(critic_lr_schedule)
        )
        critic_optimizer = torch.optim.Adam(
            critic_model.parameters(), lr=critic_lr, eps=1e-5,
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
                critic_optimizer=critic_optimizer,
                env=train_env,
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
            update_optimizer_learning_rate(critic_optimizer, critic_lr)

            observation, episode_starts, rollout_metrics = collect_candidate_rollout(
                policy=policy,
                critic_model=critic_model,
                env=train_env,
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
                optimizer=critic_optimizer,
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
            n_eval_episodes=eval_episodes,
            deterministic=ppo_config.eval_deterministic,
            device=device,
            train_env=train_env,
        )
        param_stats = policy.policy_param_stats()
        policy_backbone_params = param_stats["policy_backbone_params"]
        policy_head_params = param_stats["policy_head_params"]
        policy_params = param_stats["policy_params"]
        trainable_policy_params = param_stats["trainable_policy_params"]

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


# ---------------------------------------------------------------------------
# Ray Actor for EA candidate evaluation
# ---------------------------------------------------------------------------


class EACandidateEvaluatorActor:
    """Ray actor that evaluates a single EA candidate per call.

    Holds a cached checkpoint dict to avoid repeated disk I/O.
    Environments and models are rebuilt per candidate because fine-tuning
    mutates model weights.
    """

    def __init__(
        self,
        args_dict: dict[str, Any],
        ppo_config_dict: dict[str, Any],
    ) -> None:
        self.args_dict = args_dict
        self.ppo_config_dict = ppo_config_dict

    def evaluate_candidate(
        self,
        gene: list[int],
        arch_config_dict: dict[str, Any],
        train_seed: int,
        eval_seed: int,
    ) -> dict[str, Any]:
        """Finetune and evaluate a single candidate architecture."""
        args = argparse.Namespace(**self.args_dict)
        ppo_config = OmegaConf.create(self.ppo_config_dict)
        arch_config = ArchConfig.from_dict(arch_config_dict)
        result = finetune_and_evaluate_candidate(
            args=args,
            ppo_config=ppo_config,
            arch_config=arch_config,
            train_seed=train_seed,
            eval_seed=eval_seed,
        )
        return {
            "gene": gene,
            "arch_config": arch_config_dict,
            "train_seed": int(train_seed),
            "eval_seed": int(eval_seed),
            "pid": os.getpid(),
            **result,
        }


# ---------------------------------------------------------------------------
# EvoX Problem (modified in-place to use Ray)
# ---------------------------------------------------------------------------


class NewPolicySubnetProblem(Problem):
    """EvoX problem that evaluates policy-supernet subnet genes by PPO fine-tuning.

    Uses Ray actors with preemptive scheduling via ActorPool.
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
        self.num_workers = max(1, int(args.workers))
        self.eval_call_index = 0
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.last_records: list[dict[str, Any]] = []
        self.last_cache_hits = 0
        self._pool: ActorPool | None = None

    def _ensure_pool(self) -> ActorPool:
        """Lazily create Ray actors and wrap them in an ActorPool."""
        if self._pool is not None:
            return self._pool

        worker_cfg = compute_ray_worker_config(self.num_workers)

        RemoteActor = ray.remote(
            num_gpus=worker_cfg.gpu_fraction,
        )(EACandidateEvaluatorActor)

        actors = [
            RemoteActor.remote(
                args_dict=self.args_dict,
                ppo_config_dict=self.ppo_config_dict,
            )
            for _ in range(worker_cfg.num_workers)
        ]
        self._pool = ActorPool(actors)
        return self._pool

    def _compute_seed(self, candidate_index: int) -> tuple[int, int]:
        train_seed = (
            int(self.ppo_config_dict["seed"])
            + self.eval_call_index * int(self.args_dict["eval_call_seed_stride"])
            + candidate_index * int(self.args_dict["candidate_seed_stride"])
        )
        eval_seed = int(self.ppo_config_dict["seed"]) + EVAL_SEED_OFFSET
        return train_seed, eval_seed

    @torch.compiler.disable
    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        genes = [[round(value) for value in row.tolist()] for row in pop.detach().cpu()]
        records: list[dict[str, Any] | None] = [None] * len(genes)
        pending_tasks: list[dict[str, Any]] = []
        pending_indices: list[int] = []
        self.last_cache_hits = 0

        for index, gene in enumerate(genes):
            self.codec.validate_gene(gene)
            key = tuple(gene)
            if key in self.cache:
                records[index] = copy.deepcopy(self.cache[key])
                self.last_cache_hits += 1
            else:
                train_seed, eval_seed = self._compute_seed(index)
                pending_tasks.append({
                    "gene": gene,
                    "arch_config_dict": self.codec.gene_to_arch(gene).to_dict(),
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "original_index": index,
                })
                pending_indices.append(index)

        if pending_tasks:
            if self.num_workers <= 1:
                # Single worker: run locally without Ray overhead
                for task in pending_tasks:
                    args = argparse.Namespace(**self.args_dict)
                    ppo_config = OmegaConf.create(self.ppo_config_dict)
                    arch_config = ArchConfig.from_dict(task["arch_config_dict"])
                    result = finetune_and_evaluate_candidate(
                        args=args,
                        ppo_config=ppo_config,
                        arch_config=arch_config,
                        train_seed=task["train_seed"],
                        eval_seed=task["eval_seed"],
                    )
                    record = {
                        "gene": task["gene"],
                        "arch_config": task["arch_config_dict"],
                        "train_seed": int(task["train_seed"]),
                        "eval_seed": int(task["eval_seed"]),
                        "pid": os.getpid(),
                        **result,
                    }
                    key = tuple(task["gene"])
                    self.cache[key] = copy.deepcopy(record)
                    records[task["original_index"]] = record
            else:
                # Multi-worker: dispatch via Ray ActorPool
                pool = self._ensure_pool()
                evaluated = list(
                    pool.map_unordered(
                        lambda actor, task: actor.evaluate_candidate.remote(
                            gene=task["gene"],
                            arch_config_dict=task["arch_config_dict"],
                            train_seed=task["train_seed"],
                            eval_seed=task["eval_seed"],
                        ),
                        pending_tasks,
                    )
                )
                # Match results back to their original indices by gene
                gene_to_index = {
                    tuple(task["gene"]): task["original_index"]
                    for task in pending_tasks
                }
                for record in evaluated:
                    key = tuple(record["gene"])
                    self.cache[key] = copy.deepcopy(record)
                    records[gene_to_index[key]] = record

        self.last_records = [record for record in records if record is not None]
        self.eval_call_index += 1
        objectives = [
            [-float(record["return"]), float(record["policy_params"])]
            for record in self.last_records
        ]
        return torch.tensor(objectives, dtype=torch.float32, device=pop.device)

    def close(self) -> None:
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

    # Initialize Ray (idempotent)
    ray.init(ignore_reinit_error=True)

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
