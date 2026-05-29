"""Stage 1: Multi-worker PPO finetune for a list of subnet architectures.

Accepts ``--arch_configs`` (a JSON list of ArchConfig dicts, same format as
``stage1_eval_archs.py``).  Spawns ``num_gpus * workers_per_gpu`` processes
via ``torch.multiprocessing``.  Each worker is pinned to a GPU (round-robin)
and pulls architectures from a shared task queue, so fast-finishing workers
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
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of concurrent training processes per GPU (default: 1). "
             "Total workers = num_gpus * workers_per_gpu.",
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
    if args.workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be >= 1.")
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
# Worker loop (pulls tasks from a shared queue)
# ---------------------------------------------------------------------------


def _worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    args: argparse.Namespace,
    ppo_config: Any,
    gpu_id: int,
    stage_name: str,
    wandb_group: str,
) -> None:
    """Worker process: pulls (arch_index, arch_config) from *task_queue*,
    trains each one, and pushes the result dict into *result_queue*.

    Stops when it receives a ``None`` sentinel from the queue.
    """
    env_id = ppo_config_to_dict(ppo_config).get("env_id", "unknown_env")
    clean_env_id = env_id.replace("/", "-")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    while True:
        item = task_queue.get()
        if item is None:  # sentinel → shut down
            break

        arch_index, arch_config = item
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
            result_queue.put(manifest)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(
                f"[worker pid={os.getpid()} gpu={gpu_id}] "
                f"arch_{arch_index} FAILED after {elapsed:.1f}s: {exc}",
                flush=True,
            )
            traceback.print_exc()
            result_queue.put({
                "arch_index": arch_index,
                "status": "error",
                "error": str(exc),
                "train_time_s": round(elapsed, 2),
            })


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

    num_archs = len(arch_configs)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    effective_gpu_count = max(1, num_gpus)  # treat CPU-only as 1 device
    workers_per_gpu = max(1, int(args.workers_per_gpu))
    total_workers = effective_gpu_count * workers_per_gpu

    print(
        f"Training {num_archs} architectures with {total_workers} worker(s) "
        f"({workers_per_gpu}/gpu x {effective_gpu_count} gpu(s))...",
        flush=True,
    )

    if total_workers <= 1 and num_archs <= 1:
        # Fast path: single arch, single worker - no mp overhead
        all_results: list[dict[str, Any]] = []
        for arch_index, arch_config in enumerate(arch_configs):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            per_arch_output_dir = output_dir / f"arch_{arch_index}"
            run_name = f"{stage_name}-{clean_env_id}_{arch_index}"
            wandb_init_fn = _make_wandb_init_fn(group=wandb_group, run_name=run_name)
            t0 = time.monotonic()
            manifest = run_single_arch(
                args,
                ppo_config,
                arch_config=arch_config,
                device=device,
                output_dir=per_arch_output_dir,
                wandb_init_fn=wandb_init_fn,
                extra_config={"arch_index": arch_index, "gpu_id": 0},
                progress_bar_desc=f"arch_{arch_index}",
            )
            elapsed = time.monotonic() - t0
            manifest["arch_index"] = arch_index
            manifest["gpu_id"] = 0
            manifest["train_time_s"] = round(elapsed, 2)
            manifest["status"] = "success"
            all_results.append(manifest)
    else:
        # Multi-worker path with shared task queue
        context = mp.get_context(args.mp_start_method)
        task_queue: mp.Queue = context.Queue()
        result_queue: mp.Queue = context.Queue()

        # Fill task queue
        for arch_index, arch_config in enumerate(arch_configs):
            task_queue.put((arch_index, arch_config))

        # Add sentinels (one per worker)
        for _ in range(total_workers):
            task_queue.put(None)

        # Start workers
        processes: list[mp.Process] = []
        for worker_idx in range(total_workers):
            gpu_id = worker_idx % effective_gpu_count
            p = context.Process(
                target=_worker_loop,
                args=(task_queue, result_queue, args, ppo_config,
                      gpu_id, stage_name, wandb_group),
            )
            p.start()
            processes.append(p)

        # Collect results (one per arch)
        all_results = []
        for _ in range(num_archs):
            all_results.append(result_queue.get())

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
        "num_archs": num_archs,
        "total_workers": total_workers,
        "workers_per_gpu": workers_per_gpu,
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
