"""Stage 1: Multi-worker PPO finetune for a list of subnet architectures.

Accepts ``--arch_configs`` (a JSON list of ArchConfig dicts, same format as
``stage1_eval_archs.py``).  Uses Ray actors with preemptive scheduling via
``ActorPool``.  Each actor trains one architecture at a time; idle actors
automatically pick up the next available arch.

Each subnet gets its own independent wandb run under a shared wandb **group**,
with run name suffixed ``_<arch_index>`` (0-based, preserving the JSON order).

The per-architecture training loop is fully reused from
``stage1_train_arch_ppo.run()``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import ray
import torch
from ray.util.actor_pool import ActorPool

from setup_utils import (
    add_ppo_config_args,
    compute_ray_worker_config,
    load_ppo_config,
    ppo_config_to_dict,
)
from stage1_eval_archs import load_arch_configs_list, validate_arch_config
from supernet_backbone import ArchConfig, SearchSpace
from wandb_utils import (
    WANDB_PROJECT,
    sanitize_wandb_value,
)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: Multi-worker PPO finetune for a list of subnet architectures.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument(
        "--arch_configs",
        required=True,
        help="JSON file containing a list of ArchConfig dicts.",
    )
    parser.add_argument(
        "--supernet_checkpoint",
        default="runs/stage1_policy_supernet/policy_supernet_best.pt",
        help="Stage 1 policy-supernet checkpoint used to initialize the actor supernet and critic.",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/stage1_train_archs_ppo",
        help="Root output directory. Per-arch results are stored under <output_dir>/arch_<i>/.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Total number of Ray actors for parallel training.",
    )
    parser.add_argument(
        "--critic_warmup_timesteps",
        type=int,
        default=0,
        help="Critic-only warmup timesteps before actor PPO finetune; 0 disables warmup.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to the stage name.",
    )
    args = parser.parse_args()
    if args.critic_warmup_timesteps < 0:
        raise ValueError("critic_warmup_timesteps must be non-negative.")
    return args


# ---------------------------------------------------------------------------
# wandb init with explicit group / run-name control
# ---------------------------------------------------------------------------


def _make_wandb_init_fn(group: str, run_name: str):
    """Return a wandb init callable with bound group and run_name.

    The returned function has the same signature as ``init_wandb_run``:
    ``(stage, run_config, output_dir) -> wandb_run | None``.
    """

    def _init(stage: str, run_config: dict[str, Any], output_dir: str | Path):
        try:
            import wandb
        except Exception as exc:
            print(f"wandb_init_failed stage={stage} error={exc}", flush=True)
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = os.environ.get("WANDB_MODE", "online")
        config = sanitize_wandb_value(run_config)

        env_id = config.get("ppo_config", {}).get("env_id", "unknown_env")
        tags = [env_id] if env_id != "unknown_env" else []

        try:
            return wandb.init(
                project=WANDB_PROJECT,
                group=group,
                name=run_name,
                tags=tags,
                config=config,
                dir=str(output_dir),
                mode=mode,
                settings=wandb.Settings(silent=True),
            )
        except Exception as exc:
            print(f"wandb_init_failed stage={stage} error={exc}", flush=True)
            return None

    return _init


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------


class PPOTrainWorkerActor:
    """Ray actor that trains a single architecture per call via stage1_train_arch_ppo.run()."""

    def __init__(
        self,
        args_dict: dict[str, Any],
        ppo_config_dict: dict[str, Any],
        stage_name: str,
        wandb_group: str,
    ) -> None:
        self.args_dict = args_dict
        self.ppo_config_dict = ppo_config_dict
        self.stage_name = stage_name
        self.wandb_group = wandb_group

    def train_arch(
        self,
        arch_index: int,
        arch_config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Train a single architecture and return the manifest dict."""
        from omegaconf import OmegaConf

        from stage1_train_arch_ppo import run as run_single_arch

        args = argparse.Namespace(**self.args_dict)
        ppo_config = OmegaConf.create(self.ppo_config_dict)
        arch_config = ArchConfig.from_dict(arch_config_dict)

        env_id = self.ppo_config_dict.get("env_id", "unknown_env")
        clean_env_id = env_id.replace("/", "-")
        run_name = f"{self.stage_name}-{clean_env_id}_{arch_index}"
        wandb_init_fn = _make_wandb_init_fn(
            group=self.wandb_group, run_name=run_name
        )
        per_arch_output_dir = Path(args.output_dir) / f"arch_{arch_index}"

        t0 = time.monotonic()
        try:
            manifest = run_single_arch(
                args,
                ppo_config,
                arch_config=arch_config,
                output_dir=per_arch_output_dir,
                wandb_init_fn=wandb_init_fn,
                extra_config={
                    "arch_index": arch_index,
                },
                progress_bar_desc=f"arch_{arch_index}",
            )
            elapsed = time.monotonic() - t0
            manifest["arch_index"] = arch_index
            manifest["train_time_s"] = round(elapsed, 2)
            manifest["status"] = "success"
            return manifest
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(
                f"[worker pid={os.getpid()}] "
                f"arch_{arch_index} FAILED after {elapsed:.1f}s: {exc}",
                flush=True,
            )
            traceback.print_exc()
            return {
                "arch_index": arch_index,
                "status": "error",
                "error": str(exc),
                "train_time_s": round(elapsed, 2),
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
        f"stage1_train_archs_ppo_{args.suffix}"
        if getattr(args, "suffix", "")
        else "stage1_train_archs_ppo"
    )

    # wandb group: all arch runs share this group
    ppo_config_dict = ppo_config_to_dict(ppo_config)
    env_id = ppo_config_dict.get("env_id", "unknown_env")
    clean_env_id = env_id.replace("/", "-")
    wandb_group = f"{stage_name}-{clean_env_id}"

    # Load and validate arch configs
    search_space = SearchSpace()
    (output_dir / "search_space.json").write_text(
        json.dumps(search_space.to_dict(), indent=2)
    )

    raw_arch_configs = load_arch_configs_list(args.arch_configs)
    arch_configs: list[ArchConfig] = []
    for i, raw in enumerate(raw_arch_configs):
        try:
            arch = ArchConfig.from_dict(raw)
            validate_arch_config(search_space, arch)
            arch_configs.append(arch)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid arch config at index {i}: {e}") from e

    num_archs = len(arch_configs)
    worker_cfg = compute_ray_worker_config(args.workers)

    print(
        f"Training {num_archs} architectures with {worker_cfg.summary()}...",
        flush=True,
    )

    # Initialize Ray (idempotent)
    ray.init(ignore_reinit_error=True)

    # Create Ray Actor class with resource spec
    RemoteActor = ray.remote(
        num_gpus=worker_cfg.gpu_fraction,
    )(PPOTrainWorkerActor)

    actors = [
        RemoteActor.remote(
            args_dict=vars(args),
            ppo_config_dict=ppo_config_dict,
            stage_name=stage_name,
            wandb_group=wandb_group,
        )
        for _ in range(worker_cfg.num_workers)
    ]
    pool = ActorPool(actors)

    # Submit all archs for preemptive scheduling (1 arch per call)
    task_args = [
        (arch_index, arch.to_dict())
        for arch_index, arch in enumerate(arch_configs)
    ]
    all_results: list[dict[str, Any]] = list(
        pool.map_unordered(
            lambda actor, item: actor.train_arch.remote(item[0], item[1]),
            task_args,
        )
    )

    # Sort by arch_index
    all_results.sort(key=lambda r: r.get("arch_index", -1))

    # Write combined results as JSONL
    records_path = output_dir / "train_records.jsonl"
    with records_path.open("w") as f:
        for record in all_results:
            f.write(json.dumps(record) + "\n")

    # Print summary
    for record in all_results:
        status = record.get("status", "unknown")
        if status == "success":
            best_ret = record.get("best_eval_ep_return", "N/A")
            params = record.get("policy_params", "N/A")
            print(
                f"  arch[{record['arch_index']}] status={status} "
                f"best_eval_return={best_ret} "
                f"policy_params={params} "
                f"time={record.get('train_time_s', 0):.1f}s",
                flush=True,
            )
        else:
            print(
                f"  arch[{record['arch_index']}] status={status} "
                f"error={record.get('error', '')} "
                f"time={record.get('train_time_s', 0):.1f}s",
                flush=True,
            )

    # Find best among successful runs
    successful = [r for r in all_results if r.get("status") == "success"]
    if successful:
        best = max(
            successful,
            key=lambda r: r.get("best_eval_ep_return") or float("-inf"),
        )
        print(
            f"\nBest: arch[{best['arch_index']}] "
            f"best_eval_return={best.get('best_eval_ep_return')} "
            f"policy_params={best.get('policy_params')}",
            flush=True,
        )

    # Build top-level manifest
    manifest = {
        "stage": stage_name,
        "supernet_checkpoint": str(args.supernet_checkpoint),
        "arch_configs_path": str(args.arch_configs),
        "num_archs": num_archs,
        "num_workers": worker_cfg.num_workers,
        "workers_per_gpu": worker_cfg.workers_per_gpu,
        "num_gpus": worker_cfg.num_gpus,
        "wandb_group": wandb_group,
        "records": str(records_path),
        "search_space": str(output_dir / "search_space.json"),
        "results_summary": [
            {
                "arch_index": r.get("arch_index"),
                "status": r.get("status"),
                "best_eval_ep_return": r.get("best_eval_ep_return"),
                "policy_params": r.get("policy_params"),
                "train_time_s": r.get("train_time_s"),
            }
            for r in all_results
        ],
        "args": vars(args),
        "ppo_config": ppo_config_dict,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
