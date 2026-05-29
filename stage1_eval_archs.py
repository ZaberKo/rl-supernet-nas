from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import ray
import torch
from ray.util.actor_pool import ActorPool

from checkpoint_utils import (
    build_policy_from_checkpoint,
    load_checkpoint,
)
from env_utils import EVAL_SEED_OFFSET, make_vec_env_from_ppo_config
from ppo_utils import (
    PolicySupernet,
    evaluate_actor_subnet,
)
from setup_utils import (
    add_ppo_config_args,
    build_run_config,
    compute_ray_worker_config,
    load_ppo_config,
    ppo_config_to_dict,
    resolve_device,
    set_global_seeds,
)
from supernet_backbone import ArchConfig, SearchSpace
from wandb_utils import finish_wandb_run, init_wandb_run, update_wandb_summary



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: Evaluate pre-trained supernet subnets without any training/search. "
        "Loads a policy-supernet checkpoint, sets each architecture, and "
        "evaluates the inherited parameters directly.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument(
        "--arch_configs",
        required=True,
        help="JSON file containing a list of ArchConfig dicts to evaluate.",
    )
    parser.add_argument(
        "--supernet_checkpoint",
        default="runs/stage1_policy_supernet/policy_supernet_best.pt",
        help="Policy-supernet checkpoint whose parameters subnets inherit.",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/stage1_eval_archs",
        help="Directory for evaluation results and manifest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Total number of Ray actors for parallel subnet evaluation.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to the stage name.",
    )
    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_arch_configs_list(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON file containing a list of ArchConfig dicts."""
    arch_path = Path(path)
    if not arch_path.exists():
        raise FileNotFoundError(f"Arch configs file does not exist: {arch_path}")
    data = json.loads(arch_path.read_text())
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("arch_configs JSON must be a non-empty list of ArchConfig dicts.")
    return data


def validate_arch_config(search_space: SearchSpace, arch_config: ArchConfig) -> None:
    if len(arch_config.stage_depths) != search_space.num_stages:
        raise ValueError("Architecture stage count does not match the search space.")
    if len(arch_config.layer_configs) != search_space.num_stages:
        raise ValueError(
            "Architecture layer config stage count does not match the search space."
        )
    for stage_index, (depth, candidates, stage_layers, max_depth) in enumerate(
        zip(
            arch_config.stage_depths,
            search_space.stage_depth_candidates,
            arch_config.layer_configs,
            search_space.max_stage_depths,
            strict=True,
        )
    ):
        if int(depth) not in candidates:
            raise ValueError(f"Stage {stage_index} depth is not in the search space.")
        if len(stage_layers) != max_depth:
            raise ValueError(
                f"Stage {stage_index} must contain {max_depth} layer configs."
            )
        for layer_index, layer_config in enumerate(stage_layers):
            if int(layer_config.kernel_size) not in search_space.kernel_size_candidates:
                raise ValueError(
                    f"Stage {stage_index} layer {layer_index} kernel size is not in the search space."
                )
            if (
                int(layer_config.expand_ratio)
                not in search_space.expand_ratio_candidates
            ):
                raise ValueError(
                    f"Stage {stage_index} layer {layer_index} expand ratio is not in the search space."
                )


def evaluate_single_arch(
    policy: PolicySupernet,
    eval_env: Any,
    arch_config: ArchConfig,
    ppo_config: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Set arch on the shared policy, evaluate on the shared eval_env, return metrics."""
    eval_episodes = ppo_config.eval_episodes
    if eval_episodes <= 0:
        raise ValueError(
            "ppo.eval_episodes must be positive for evaluation."
        )
    policy.set_sample_config(arch_config)
    eval_metrics = evaluate_actor_subnet(
        policy=policy,
        eval_env=eval_env,
        n_eval_episodes=eval_episodes,
        deterministic=ppo_config.eval_deterministic,
        device=device,
    )
    param_stats = policy.policy_param_stats()
    return {
        "return": float(eval_metrics["ep_return"]),
        "return_std": float(eval_metrics["ep_return_std"]),
        "ep_return": float(eval_metrics["ep_return"]),
        "ep_return_std": float(eval_metrics["ep_return_std"]),
        "ep_length": float(eval_metrics["ep_length"]),
        "ep_length_std": float(eval_metrics["ep_length_std"]),
        "policy_backbone_params": param_stats["policy_backbone_params"],
        "policy_head_params": param_stats["policy_head_params"],
        "policy_params": param_stats["policy_params"],
    }


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------


class SubnetEvaluatorActor:
    """Ray actor that holds a persistent policy supernet and eval env.

    Each call to :meth:`evaluate` sets the arch config on the shared policy
    and evaluates a single architecture (preemptive 1-at-a-time scheduling).
    """

    def __init__(
        self,
        supernet_checkpoint: str,
        ppo_config_dict: dict[str, Any],
    ) -> None:
        from omegaconf import OmegaConf

        self.ppo_config = OmegaConf.create(ppo_config_dict)
        eval_seed = int(self.ppo_config.seed) + EVAL_SEED_OFFSET
        set_global_seeds(eval_seed)
        self.device = resolve_device(str(self.ppo_config.device))
        search_space = SearchSpace()
        checkpoint = load_checkpoint(supernet_checkpoint, map_location=self.device)

        self.eval_env = make_vec_env_from_ppo_config(
            self.ppo_config, seed=eval_seed, n_envs=self.ppo_config.eval_n_envs
        )
        self.policy = build_policy_from_checkpoint(
            ppo_config=self.ppo_config,
            env=self.eval_env,
            search_space=search_space,
            checkpoint=checkpoint,
            device=self.device,
        )

    def evaluate(
        self, arch_index: int, arch_config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate a single architecture and return metrics."""
        arch_config = ArchConfig.from_dict(arch_config_dict)
        t0 = time.monotonic()
        result = evaluate_single_arch(
            policy=self.policy,
            eval_env=self.eval_env,
            arch_config=arch_config,
            ppo_config=self.ppo_config,
            device=self.device,
        )
        elapsed = time.monotonic() - t0
        return {
            "arch_index": arch_index,
            "arch_config": arch_config_dict,
            "eval_seed": int(self.ppo_config.seed) + EVAL_SEED_OFFSET,
            "pid": os.getpid(),
            "eval_time_s": round(elapsed, 2),
            **result,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_name = (
        f"stage1_eval_archs_{args.suffix}" if getattr(args, "suffix", "") else "stage1_eval_archs"
    )
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run(stage_name, run_config, output_dir)

    search_space = SearchSpace()
    (output_dir / "search_space.json").write_text(
        json.dumps(search_space.to_dict(), indent=2)
    )

    # Load and validate arch configs
    raw_arch_configs = load_arch_configs_list(args.arch_configs)
    arch_configs: list[ArchConfig] = []
    for i, raw in enumerate(raw_arch_configs):
        try:
            arch = ArchConfig.from_dict(raw)
            validate_arch_config(search_space, arch)
            arch_configs.append(arch)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid arch config at index {i}: {e}") from e

    eval_seed = int(ppo_config.seed) + EVAL_SEED_OFFSET
    ppo_config_dict = ppo_config_to_dict(ppo_config)

    worker_cfg = compute_ray_worker_config(args.workers)

    print(
        f"Evaluating {len(arch_configs)} architectures with {worker_cfg.summary()}...",
        flush=True,
    )

    # Initialize Ray (idempotent)
    ray.init(ignore_reinit_error=True)

    # Create Ray Actor class with resource spec
    RemoteActor = ray.remote(
        num_gpus=worker_cfg.gpu_fraction,
    )(SubnetEvaluatorActor)

    actors = [
        RemoteActor.remote(
            supernet_checkpoint=args.supernet_checkpoint,
            ppo_config_dict=ppo_config_dict,
        )
        for _ in range(worker_cfg.num_workers)
    ]
    pool = ActorPool(actors)

    # Submit all archs for preemptive scheduling (1 arch per call)
    task_args = [
        (arch_index, arch.to_dict())
        for arch_index, arch in enumerate(arch_configs)
    ]
    results: list[dict[str, Any]] = list(
        pool.map_unordered(
            lambda actor, item: actor.evaluate.remote(item[0], item[1]),
            task_args,
        )
    )

    # Sort by arch_index to preserve input order
    results.sort(key=lambda r: r["arch_index"])

    # Write per-architecture results as JSONL
    records_path = output_dir / "eval_records.jsonl"
    with records_path.open("w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    # Print summary for each arch
    for record in results:
        print(
            f"  arch[{record['arch_index']}] "
            f"return={record['ep_return']:.6g} "
            f"return_std={record['ep_return_std']:.6g} "
            f"policy_backbone_params={record['policy_backbone_params']} "
            f"time={record['eval_time_s']:.1f}s",
            flush=True,
        )

    # Find best
    best = max(results, key=lambda r: r["ep_return"])
    print(
        f"\nBest: arch[{best['arch_index']}] "
        f"return={best['ep_return']:.6g} policy_backbone_params={best['policy_backbone_params']}",
        flush=True,
    )

    # Build manifest
    manifest = {
        "stage": stage_name,
        "supernet_checkpoint": str(args.supernet_checkpoint),
        "arch_configs_path": str(args.arch_configs),
        "num_archs": len(arch_configs),
        "eval_seed": eval_seed,
        "num_workers": worker_cfg.num_workers,
        "records": str(records_path),
        "search_space": str(output_dir / "search_space.json"),
        "results": results,
        "best_arch_index": best["arch_index"],
        "best_return": best["ep_return"],
        "best_policy_backbone_params": best["policy_backbone_params"],
        "args": vars(args),
        "ppo_config": ppo_config_dict,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    update_wandb_summary(
        wandb_run,
        {
            "num_archs": len(arch_configs),
            "best_return": best["ep_return"],
            "best_policy_backbone_params": best["policy_backbone_params"],
            "best_arch_index": best["arch_index"],
        },
    )
    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
