from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import DictConfig

from config_utils import add_ppo_config_args, build_run_config, load_ppo_config, ppo_config_to_dict
from ppo_utils import make_vec_env_from_ppo_config, steps_for_transition_budget
from trajectory_data import (
    collect_random_trajectories,
    count_trajectory_file,
    read_metadata,
    write_mixed_trajectory_dataset,
    write_supervised_transition_dataset,
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
    parser.add_argument("--random_transitions", type=int, default=0, help="Exact random transition count. Use 0 to skip random-policy collection.")
    parser.add_argument("--random_seed_offset", type=int, default=10_000, help="Offset added to the base seed for random-policy collection.")
    parser.add_argument("--random_output_name", default="random_trajectories.arrow", help="Directory name for the random trajectory Arrow dataset inside output_dir.")
    parser.add_argument("--mixed_output_name", default="mixed_trajectories.arrow", help="Directory name for the combined PPO+random raw trajectory Arrow dataset inside output_dir.")
    parser.add_argument("--representation_horizon", type=int, default=1, help="Number of future steps packed into each stage2 supervised sample.")
    parser.add_argument("--representation_output_name", default="representation_data.arrow", help="Directory name for the stage2 supervised transition dataset inside output_dir.")
    parser.add_argument("--manifest_name", default="manifest.json", help="File name for the mixed-data summary manifest inside output_dir.")
    return parser.parse_args()


def empty_trajectory_count() -> dict[str, int]:
    return {
        "num_steps": 0,
        "num_envs": 0,
        "num_transitions": 0,
        "num_terminated": 0,
        "num_truncated": 0,
        "num_done": 0,
        "num_trajectories": 0,
    }


def run(args: argparse.Namespace, ppo_config: DictConfig) -> dict:
    ppo_path = Path(args.ppo_trajectory_file)
    if not ppo_path.exists():
        raise FileNotFoundError(f"PPO trajectory file does not exist: {ppo_path}")
    if args.random_transitions < 0:
        raise ValueError("random_transitions must be non-negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run("stage1_mix_random_data", run_config, output_dir)
    ppo_count = count_trajectory_file(ppo_path)
    ppo_metadata = read_metadata(ppo_path)
    random_transitions = int(args.random_transitions)
    random_steps = steps_for_transition_budget(random_transitions, ppo_config.train_n_envs)
    random_path = output_dir / args.random_output_name if random_transitions > 0 else None

    if random_transitions > 0:
        random_metadata = {
            "source": "random_policy",
            "env_id": ppo_config.env_id,
            "seed": ppo_config.seed + args.random_seed_offset,
            "train_n_envs": ppo_config.train_n_envs,
            "image_size": ppo_config.image_size,
            "target_random_transitions": random_transitions,
            "target_random_steps": random_steps,
            "ppo_trajectory_file": str(ppo_path),
            "ppo_num_transitions": ppo_count["num_transitions"],
            "target_random_to_ppo_ratio": random_transitions / max(1, ppo_count["num_transitions"]),
            "target_random_fraction": random_transitions / max(1, ppo_count["num_transitions"] + random_transitions),
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        }
        random_env = make_vec_env_from_ppo_config(
            ppo_config,
            seed=ppo_config.seed + args.random_seed_offset,
            n_envs=ppo_config.train_n_envs,
        )
        try:
            collect_random_trajectories(
                env=random_env,
                num_steps=random_steps,
                output_path=random_path,
                metadata=random_metadata,
                max_transitions=random_transitions,
            )
        finally:
            random_env.close()
        random_count = count_trajectory_file(random_path)
    else:
        random_count = empty_trajectory_count()

    source_paths = [ppo_path] if random_path is None else [ppo_path, random_path]
    mixed_path = output_dir / args.mixed_output_name
    mixed_metadata = {
        "source": "ppo_random_mixed",
        "env_id": ppo_config.env_id,
        "seed": ppo_config.seed,
        "train_n_envs": ppo_config.train_n_envs,
        "image_size": ppo_config.image_size,
        "ppo_trajectory_file": str(ppo_path),
        "random_trajectory_file": str(random_path) if random_path is not None else None,
        "ppo_count": ppo_count,
        "random_count": random_count,
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    write_mixed_trajectory_dataset(
        input_paths=source_paths,
        output_path=mixed_path,
        metadata=mixed_metadata,
    )
    mixed_count = count_trajectory_file(mixed_path)
    representation_path = output_dir / args.representation_output_name
    representation_metadata = {
        "source": "stage1_supervised_transition_samples",
        "env_id": ppo_config.env_id,
        "seed": ppo_config.seed,
        "image_size": ppo_config.image_size,
        "horizon": int(args.representation_horizon),
        "raw_mixed_trajectory_file": str(mixed_path),
        "supervised_source_files": [str(path) for path in source_paths],
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    write_supervised_transition_dataset(
        input_paths=source_paths,
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
        "random_trajectory_file": str(random_path) if random_path is not None else None,
        "ppo_count": ppo_count,
        "random_count": random_count,
        "mixed_count": mixed_count,
        "representation_count": representation_count,
        "actual_random_to_ppo_ratio": random_count["num_transitions"] / max(1, ppo_count["num_transitions"]),
        "actual_random_fraction": random_count["num_transitions"] / max(1, ppo_count["num_transitions"] + random_count["num_transitions"]),
        "ppo_metadata": ppo_metadata,
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
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
    artifact_paths = [mixed_path, representation_path, manifest_path]
    if random_path is not None:
        artifact_paths.insert(0, random_path)
    log_wandb_artifact(
        wandb_run,
        name=f"stage1-mix-{output_dir.name}",
        artifact_type="stage1-mix-output",
        paths=artifact_paths,
    )
    finish_wandb_run(wandb_run)
    return mixed_manifest


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    print(json.dumps(run(args, ppo_config), indent=2))


if __name__ == "__main__":
    main()
