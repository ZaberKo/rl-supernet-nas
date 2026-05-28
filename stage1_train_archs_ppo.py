"""Stage 1: Multi-worker PPO finetune for a list of subnet architectures.

Accepts ``--arch_configs`` (a JSON list of ArchConfig dicts, same format as
``stage1_eval_archs.py``).  Spawns ``--train_workers`` processes via
``torch.multiprocessing`` – workers can exceed the number of GPUs since each is
assigned a GPU in round-robin fashion.

Each subnet gets its own independent wandb run under a shared wandb **group**,
with run name suffixed ``_<arch_index>`` (0-based, preserving the JSON order).

The per-architecture training loop is fully reused from
``stage1_train_arch_ppo.run()``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from itertools import batched
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

from setup_utils import (
    add_ppo_config_args,
    load_ppo_config,
    ppo_config_to_dict,
)
from stage1_eval_archs import load_arch_configs_list, validate_arch_config
from stage1_train_arch_ppo import run as run_single_arch
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
        "--train_workers",
        type=int,
        default=1,
        help="Number of torch.mp workers.  Can exceed the number of GPUs (round-robin assignment).",
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
    args.mp_start_method = "spawn"
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
        wandb_dir = output_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
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
                dir=str(wandb_dir),
                mode=mode,
                settings=wandb.Settings(silent=True),
            )
        except Exception as exc:
            print(f"wandb_init_failed stage={stage} error={exc}", flush=True)
            return None

    return _init


# ---------------------------------------------------------------------------
# Worker entry-point
# ---------------------------------------------------------------------------


def _train_worker(
    args: argparse.Namespace,
    ppo_config: Any,
    arch_entries: list[tuple[int, ArchConfig]],
    gpu_id: int,
    stage_name: str,
    wandb_group: str,
) -> list[dict[str, Any]]:
    """Worker process: trains a batch of architectures sequentially on one GPU."""
    env_id = ppo_config_to_dict(ppo_config).get("env_id", "unknown_env")
    clean_env_id = env_id.replace("/", "-")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    results: list[dict[str, Any]] = []
    for arch_index, arch_config in arch_entries:
        per_arch_output_dir = Path(args.output_dir) / f"arch_{arch_index}"
        run_name = f"{stage_name}-{clean_env_id}_{arch_index}"
        wandb_init_fn = _make_wandb_init_fn(group=wandb_group, run_name=run_name)

        t0 = time.monotonic()
        try:
            manifest = run_single_arch(
                args,
                ppo_config,
                arch_config=arch_config,
                device=device,
                output_dir=per_arch_output_dir,
                wandb_init_fn=wandb_init_fn,
                extra_config={
                    "arch_index": arch_index,
                    "gpu_id": gpu_id,
                },
                progress_bar_desc=f"arch_{arch_index}",
            )
            elapsed = time.monotonic() - t0
            manifest["arch_index"] = arch_index
            manifest["gpu_id"] = gpu_id
            manifest["train_time_s"] = round(elapsed, 2)
            manifest["status"] = "success"
            results.append(manifest)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(
                f"[worker pid={os.getpid()}] arch_{arch_index} FAILED after {elapsed:.1f}s: {exc}",
                flush=True,
            )
            traceback.print_exc()
            results.append({
                "arch_index": arch_index,
                "status": "error",
                "error": str(exc),
                "train_time_s": round(elapsed, 2),
            })
    return results


def _train_worker_entry(
    result_queue: mp.Queue,
    args: argparse.Namespace,
    ppo_config: Any,
    arch_entries: list[tuple[int, ArchConfig]],
    gpu_id: int,
    stage_name: str,
    wandb_group: str,
) -> None:
    """Entry-point for ``mp.Process``; puts results into *result_queue*."""
    results = _train_worker(args, ppo_config, arch_entries, gpu_id, stage_name, wandb_group)
    result_queue.put(results)


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
    env_id = ppo_config_to_dict(ppo_config).get("env_id", "unknown_env")
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

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    train_workers = max(1, int(args.train_workers))
    print(
        f"Training {len(arch_configs)} architectures with {train_workers} worker(s) "
        f"across {num_gpus} GPU(s)...",
        flush=True,
    )

    arch_entries = list(enumerate(arch_configs))

    if train_workers <= 1:
        all_results = _train_worker(
            args, ppo_config, arch_entries,
            gpu_id=0,
            stage_name=stage_name, wandb_group=wandb_group,
        )
    else:
        batch_size = math.ceil(len(arch_entries) / train_workers)
        partitions = list(batched(arch_entries, batch_size))
        context = mp.get_context(args.mp_start_method)
        result_queue: mp.Queue = context.Queue()
        processes: list[mp.Process] = []
        for worker_idx, part in enumerate(partitions):
            gpu_id = worker_idx % max(1, num_gpus)
            p = context.Process(
                target=_train_worker_entry,
                args=(result_queue, args, ppo_config, list(part),
                      gpu_id, stage_name, wandb_group),
            )
            p.start()
            processes.append(p)
        all_results: list[dict[str, Any]] = []
        for _ in processes:
            all_results.extend(result_queue.get())
        for p in processes:
            p.join()

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
        "num_archs": len(arch_configs),
        "train_workers": train_workers,
        "num_gpus": num_gpus,
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
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
