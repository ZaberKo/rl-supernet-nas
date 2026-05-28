from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from checkpoint_utils import (
    build_policy_from_checkpoint,
    load_checkpoint,
)
from env_utils import EVAL_SEED_OFFSET, make_vec_env_from_ppo_config
from ppo_utils import (
    actor_head_parameters,
    count_parameters,
    evaluate_actor_subnet,
)
from setup_utils import (
    add_ppo_config_args,
    build_run_config,
    load_ppo_config,
    ppo_config_to_dict,
    resolve_device,
    set_global_seeds,
)
from supernet_backbone import ArchConfig, SearchSpace
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb


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
        "--eval_workers",
        type=int,
        default=1,
        help="Torch multiprocessing workers for parallel subnet evaluation.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to the stage name.",
    )
    args = parser.parse_args()
    args.mp_start_method = "spawn"
    args.worker_torch_threads = 1
    return args


def load_arch_configs_list(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON file that is a list of ArchConfig dicts."""
    arch_path = Path(path)
    if not arch_path.exists():
        raise FileNotFoundError(f"Arch configs file does not exist: {arch_path}")
    data = json.loads(arch_path.read_text())
    if isinstance(data, Mapping):
        # Single arch config wrapped in a dict with an "archs" key
        if "archs" in data:
            data = data["archs"]
        else:
            # Single arch config, wrap in a list
            data = [data]
    if not isinstance(data, list):
        raise ValueError("arch_configs JSON must be a list of ArchConfig dicts.")
    if len(data) == 0:
        raise ValueError("arch_configs JSON list must not be empty.")
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
    args: argparse.Namespace,
    ppo_config: Any,
    arch_config: ArchConfig,
    eval_seed: int,
) -> dict[str, Any]:
    """Load supernet checkpoint, set arch, evaluate and return metrics."""
    set_global_seeds(eval_seed)
    device = resolve_device(str(ppo_config.device))
    search_space = SearchSpace()
    checkpoint = load_checkpoint(args.supernet_checkpoint, map_location=device)

    eval_env = make_vec_env_from_ppo_config(
        ppo_config, seed=eval_seed, n_envs=ppo_config.eval_n_envs
    )
    # We still need a train_env for evaluate_actor_subnet (obs normalization stats)
    train_env = make_vec_env_from_ppo_config(
        ppo_config, seed=eval_seed, n_envs=ppo_config.train_n_envs
    )
    try:
        policy = build_policy_from_checkpoint(
            ppo_config=ppo_config,
            train_env=train_env,
            search_space=search_space,
            checkpoint=checkpoint,
            device=device,
        )
        policy.set_active_arch(arch_config)

        eval_episodes = ppo_config.eval_episodes
        if eval_episodes <= 0:
            raise ValueError(
                "ppo.eval_episodes must be positive for evaluation."
            )
        eval_metrics = evaluate_actor_subnet(
            policy=policy,
            train_env=train_env,
            eval_env=eval_env,
            arch=arch_config,
            n_eval_episodes=eval_episodes,
            deterministic=ppo_config.eval_deterministic,
            device=device,
        )
        active_backbone_params = int(policy.backbone.elastic_num_params)
        actor_head_params = count_parameters(actor_head_parameters(policy))
        policy_params = int(
            sum(parameter.numel() for parameter in policy.parameters())
        )
        trainable_policy_params = int(
            sum(
                parameter.numel()
                for parameter in policy.parameters()
                if parameter.requires_grad
            )
        )
        return {
            "return": float(eval_metrics["ep_return"]),
            "return_std": float(eval_metrics["ep_return_std"]),
            "ep_return": float(eval_metrics["ep_return"]),
            "ep_return_std": float(eval_metrics["ep_return_std"]),
            "ep_length": float(eval_metrics["ep_length"]),
            "ep_length_std": float(eval_metrics["ep_length_std"]),
            "params": active_backbone_params,
            "actor_head_params": actor_head_params,
            "policy_params": policy_params,
            "trainable_policy_params": trainable_policy_params,
        }
    finally:
        train_env.close()
        eval_env.close()


@dataclass(frozen=True)
class EvalArchWorkerConfig:
    args: dict[str, Any]
    ppo_config: dict[str, Any]
    arch_config: dict[str, Any]
    arch_index: int
    eval_seed: int


def _eval_arch_worker(config: EvalArchWorkerConfig) -> dict[str, Any]:
    worker_threads = int(config.args.get("worker_torch_threads", 1))
    if worker_threads > 0:
        torch.set_num_threads(worker_threads)
    args = argparse.Namespace(**config.args)
    ppo_config = OmegaConf.create(config.ppo_config)
    arch_config = ArchConfig.from_dict(config.arch_config)

    t0 = time.monotonic()
    result = evaluate_single_arch(
        args=args,
        ppo_config=ppo_config,
        arch_config=arch_config,
        eval_seed=config.eval_seed,
    )
    elapsed = time.monotonic() - t0
    return {
        "arch_index": config.arch_index,
        "arch_config": config.arch_config,
        "eval_seed": config.eval_seed,
        "pid": os.getpid(),
        "eval_time_s": round(elapsed, 2),
        **result,
    }


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
    args_dict = vars(args).copy()
    ppo_config_dict = ppo_config_to_dict(ppo_config)

    # Build worker configs
    worker_configs = [
        EvalArchWorkerConfig(
            args=args_dict,
            ppo_config=ppo_config_dict,
            arch_config=arch.to_dict(),
            arch_index=i,
            eval_seed=eval_seed,
        )
        for i, arch in enumerate(arch_configs)
    ]

    eval_workers = max(1, int(args.eval_workers))
    print(
        f"Evaluating {len(arch_configs)} architectures with {eval_workers} worker(s)...",
        flush=True,
    )

    # Run evaluations
    if eval_workers <= 1:
        results = [_eval_arch_worker(config) for config in worker_configs]
    else:
        context = mp.get_context(args.mp_start_method)
        with context.Pool(processes=eval_workers) as pool:
            results = pool.map(_eval_arch_worker, worker_configs)

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
            f"params={record['params']} "
            f"time={record['eval_time_s']:.1f}s",
            flush=True,
        )

    # Find best
    best = max(results, key=lambda r: r["ep_return"])
    print(
        f"\nBest: arch[{best['arch_index']}] "
        f"return={best['ep_return']:.6g} params={best['params']}",
        flush=True,
    )

    # Build manifest
    manifest = {
        "stage": stage_name,
        "supernet_checkpoint": str(args.supernet_checkpoint),
        "arch_configs_path": str(args.arch_configs),
        "num_archs": len(arch_configs),
        "eval_seed": eval_seed,
        "eval_workers": eval_workers,
        "records": str(records_path),
        "search_space": str(output_dir / "search_space.json"),
        "results": results,
        "best_arch_index": best["arch_index"],
        "best_return": best["ep_return"],
        "best_params": best["params"],
        "args": vars(args),
        "ppo_config": ppo_config_dict,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log_wandb(
        wandb_run,
        {
            "num_archs": len(arch_configs),
            "best_return": best["ep_return"],
            "best_params": best["params"],
            "best_arch_index": best["arch_index"],
        },
    )
    finish_wandb_run(wandb_run)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
