from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from config_utils import add_ppo_config_args, load_ppo_config
from ppo_utils import make_vec_env_from_args, steps_for_transition_budget
from trajectory_data import (
    collect_random_trajectories,
    count_trajectory_file,
    read_metadata,
    write_mixed_trajectory_dataset,
    write_supervised_transition_dataset,
    write_trajectory_prefix,
)
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1B: collect or subset random trajectories and write one mixed dataset.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--ppo_trajectory_file", default="runs/stage1_ppo_max_arch/ppo_train_trajectories.arrow", help="PPO trajectory dataset produced by stage1_train_max_ppo.py.")
    parser.add_argument("--output_dir", default="runs/stage1_mix", help="Directory for random trajectories, mixed trajectories, and manifest.")
    parser.add_argument("--existing_random_trajectory_file", default="", help="Reuse an existing random trajectory file instead of collecting new random data.")
    parser.add_argument("--random_steps", type=int, default=0, help="Random-policy environment steps to collect before multiplying by train_n_envs.")
    parser.add_argument("--random_transitions", type=int, default=0, help="Exact random transition budget; takes priority over ratio and fraction.")
    parser.add_argument("--random_to_ppo_ratio", type=float, default=1.0, help="Random/PPO transition ratio when no explicit random budget is set.")
    parser.add_argument("--random_fraction", type=float, default=-1.0, help="Target random fraction in the mixed dataset; use a negative value to disable.")
    parser.add_argument("--random_seed_offset", type=int, default=10_000, help="Offset added to the base seed for random-policy collection.")
    parser.add_argument("--random_output_name", default="random_trajectories.arrow", help="Directory name for the random trajectory Arrow dataset inside output_dir.")
    parser.add_argument("--mixed_output_name", default="mixed_trajectories.arrow", help="Directory name for the combined PPO+random raw trajectory Arrow dataset inside output_dir.")
    parser.add_argument("--representation_horizon", type=int, default=1, help="Number of future steps packed into each stage2 supervised sample.")
    parser.add_argument("--representation_output_name", default="representation_data.arrow", help="Directory name for the stage2 supervised transition dataset inside output_dir.")
    parser.add_argument("--manifest_name", default="manifest.json", help="File name for the mixed-data summary manifest inside output_dir.")
    args = parser.parse_args()
    load_ppo_config(args)
    return args

def resolve_random_transitions(args: argparse.Namespace, ppo_transitions: int) -> int:
    if args.random_transitions > 0:
        return args.random_transitions
    if args.random_steps > 0:
        return args.random_steps * args.train_n_envs
    if args.random_fraction >= 0.0:
        if args.random_fraction >= 1.0:
            raise ValueError("random_fraction must be in [0, 1).")
        return int(math.ceil(ppo_transitions * args.random_fraction / (1.0 - args.random_fraction)))
    if args.random_to_ppo_ratio < 0.0:
        raise ValueError("random_to_ppo_ratio must be non-negative.")
    return int(math.ceil(ppo_transitions * args.random_to_ppo_ratio))


def run(args: argparse.Namespace) -> dict:
    ppo_path = Path(args.ppo_trajectory_file)
    if not ppo_path.exists():
        raise FileNotFoundError(f"PPO trajectory file does not exist: {ppo_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb_run("stage1_mix_random_data", args, output_dir)
    ppo_count = count_trajectory_file(ppo_path)
    ppo_metadata = read_metadata(ppo_path)
    random_transitions = resolve_random_transitions(args, ppo_count["num_transitions"])
    random_steps = steps_for_transition_budget(random_transitions, args.train_n_envs)
    random_path = output_dir / args.random_output_name

    random_metadata = {
        "source": "random_policy",
        "env_id": args.env_id,
        "seed": args.seed + args.random_seed_offset,
        "train_n_envs": args.train_n_envs,
        "image_size": args.image_size,
        "target_random_transitions": random_transitions,
        "target_random_steps": random_steps,
        "ppo_trajectory_file": str(ppo_path),
        "ppo_num_transitions": ppo_count["num_transitions"],
        "random_to_ppo_ratio": random_transitions / max(1, ppo_count["num_transitions"]),
        "random_fraction": random_transitions / max(1, ppo_count["num_transitions"] + random_transitions),
        "args": vars(args),
    }

    if args.existing_random_trajectory_file:
        write_trajectory_prefix(
            input_path=args.existing_random_trajectory_file,
            output_path=random_path,
            num_steps=random_steps,
            metadata=random_metadata,
        )
    else:
        random_env = make_vec_env_from_args(
            args,
            seed=args.seed + args.random_seed_offset,
            n_envs=args.train_n_envs,
        )
        try:
            collect_random_trajectories(
                env=random_env,
                num_steps=random_steps,
                output_path=random_path,
                metadata=random_metadata,
            )
        finally:
            random_env.close()

    random_count = count_trajectory_file(random_path)
    mixed_path = output_dir / args.mixed_output_name
    mixed_metadata = {
        "source": "ppo_random_mixed",
        "env_id": args.env_id,
        "seed": args.seed,
        "train_n_envs": args.train_n_envs,
        "image_size": args.image_size,
        "ppo_trajectory_file": str(ppo_path),
        "random_trajectory_file": str(random_path),
        "ppo_count": ppo_count,
        "random_count": random_count,
        "args": vars(args),
    }
    write_mixed_trajectory_dataset(
        input_paths=[ppo_path, random_path],
        output_path=mixed_path,
        metadata=mixed_metadata,
    )
    mixed_count = count_trajectory_file(mixed_path)
    representation_path = output_dir / args.representation_output_name
    representation_metadata = {
        "source": "stage1_supervised_transition_samples",
        "env_id": args.env_id,
        "seed": args.seed,
        "image_size": args.image_size,
        "horizon": int(args.representation_horizon),
        "raw_mixed_trajectory_file": str(mixed_path),
        "args": vars(args),
    }
    write_supervised_transition_dataset(
        input_paths=[mixed_path],
        output_path=representation_path,
        horizon=args.representation_horizon,
        metadata=representation_metadata,
    )
    representation_count = count_trajectory_file(representation_path)
    mixed_manifest = {
        "stage": "stage1_mix_random",
        "mixed_trajectory_file": str(mixed_path),
        "representation_data": str(representation_path),
        "representation_horizon": int(args.representation_horizon),
        "ppo_trajectory_file": str(ppo_path),
        "random_trajectory_file": str(random_path),
        "ppo_count": ppo_count,
        "random_count": random_count,
        "mixed_count": mixed_count,
        "representation_count": representation_count,
        "actual_random_to_ppo_ratio": random_count["num_transitions"] / max(1, ppo_count["num_transitions"]),
        "actual_random_fraction": random_count["num_transitions"] / max(1, ppo_count["num_transitions"] + random_count["num_transitions"]),
        "ppo_metadata": ppo_metadata,
        "args": vars(args),
    }
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(mixed_manifest, indent=2))
    mixed_manifest["manifest"] = str(manifest_path)
    log_wandb(
        wandb_run,
        {
            "stage1_mix/ppo_transitions": ppo_count["num_transitions"],
            "stage1_mix/random_transitions": random_count["num_transitions"],
            "stage1_mix/mixed_transitions": mixed_count["num_transitions"],
            "stage1_mix/representation_samples": representation_count["num_transitions"],
            "stage1_mix/representation_horizon": int(args.representation_horizon),
            "stage1_mix/random_to_ppo_ratio": mixed_manifest["actual_random_to_ppo_ratio"],
            "stage1_mix/random_fraction": mixed_manifest["actual_random_fraction"],
            "stage1_mix/mixed_trajectories": mixed_count.get("num_trajectories", mixed_count["num_envs"]),
        },
        step=0,
    )
    log_wandb_artifact(
        wandb_run,
        name=f"stage1-mix-{output_dir.name}",
        artifact_type="stage1-mix-output",
        paths=[random_path, mixed_path, representation_path, manifest_path],
    )
    finish_wandb_run(wandb_run)
    return mixed_manifest


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
