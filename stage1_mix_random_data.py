from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import DictConfig

from config_utils import add_ppo_config_args, build_run_config, load_ppo_config, ppo_config_to_dict
from ppo_utils import make_vec_env_from_ppo_config
from trajectory_data import (
    collect_random_supervised_transition_samples_to_hdf5,
    count_trajectory_file,
    read_hdf5_metadata,
    write_supervised_transition_samples_from_hdf5_files,
)
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


REPRESENTATION_DATASET_NAME = "representation_data.arrow"
RANDOM_REPRESENTATION_FILE = "random_representation_samples.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1B: mix PPO HDF5 samples with optional random samples and write one Arrow dataset.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--ppo_data_file", default="runs/stage1_ppo_max_arch/ppo_representation_samples.h5", help="PPO HDF5 representation samples produced by stage1_train_max_ppo.py.")
    parser.add_argument("--output_dir", default="runs/stage1_mix", help="Directory for the mixed representation dataset and manifest.")
    parser.add_argument("--random_samples", type=int, default=0, help="Exact number of random-policy horizon samples to save. Use 0 to skip random-policy collection.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for random-policy collection. Defaults to the PPO config seed.")
    parser.add_argument("--horizon", type=int, default=1, help="Number of future steps packed into each stage2 supervised sample.")
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
    ppo_path = Path(args.ppo_data_file)
    if not ppo_path.exists():
        raise FileNotFoundError(f"PPO HDF5 file does not exist: {ppo_path}")
    if args.random_samples < 0:
        raise ValueError("random_samples must be non-negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run("stage1_mix_random_data", run_config, output_dir)
    ppo_count = count_trajectory_file(ppo_path)
    ppo_metadata = read_hdf5_metadata(ppo_path)
    ppo_num_raw_transitions = int(ppo_metadata.get("num_raw_transitions", ppo_count["num_samples"]))
    random_samples = int(args.random_samples)
    random_seed = int(args.seed) if args.seed is not None else int(ppo_config.seed)
    random_rollout_steps = int(ppo_config.n_steps)
    hdf5_horizon = int(ppo_metadata.get("horizon", args.horizon))
    if int(args.horizon) != hdf5_horizon:
        raise ValueError("horizon must match the PPO HDF5 representation horizon.")

    random_path = None
    random_count = empty_trajectory_count() | {"num_samples": 0}
    random_num_raw_transitions = 0
    if random_samples > 0:
        random_path = output_dir / RANDOM_REPRESENTATION_FILE
        random_metadata = {
            "source": "random_policy_representation_samples",
            "env_id": ppo_config.env_id,
            "seed": random_seed,
            "train_n_envs": ppo_config.train_n_envs,
            "image_size": int(ppo_config.image_size),
            "horizon": hdf5_horizon,
            "target_random_samples": random_samples,
            "random_rollout_steps": random_rollout_steps,
            "ppo_data_file": str(ppo_path),
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        }
        random_env = make_vec_env_from_ppo_config(
            ppo_config,
            seed=random_seed,
            n_envs=ppo_config.train_n_envs,
        )
        try:
            _, random_num_raw_transitions = collect_random_supervised_transition_samples_to_hdf5(
                env=random_env,
                output_path=random_path,
                horizon=hdf5_horizon,
                max_samples=random_samples,
                rollout_steps=random_rollout_steps,
                metadata=random_metadata,
            )
        finally:
            random_env.close()
        random_count = count_trajectory_file(random_path)

    representation_path = output_dir / REPRESENTATION_DATASET_NAME
    source_paths = [ppo_path] if random_path is None else [ppo_path, random_path]
    representation_metadata = {
        "source": "stage1_supervised_transition_samples",
        "env_id": ppo_config.env_id,
        "seed": random_seed,
        "image_size": int(ppo_config.image_size),
        "horizon": hdf5_horizon,
        "ppo_data_file": str(ppo_path),
        "random_data_file": str(random_path) if random_path is not None else None,
        "ppo_count": ppo_count,
        "random_count": random_count,
        "ppo_num_raw_transitions": ppo_num_raw_transitions,
        "random_seed": random_seed,
        "random_num_raw_transitions": random_num_raw_transitions,
        "random_rollout_steps": random_rollout_steps,
        "ppo_metadata": ppo_metadata,
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    write_supervised_transition_samples_from_hdf5_files(
        input_paths=source_paths,
        output_path=representation_path,
        horizon=hdf5_horizon,
        metadata=representation_metadata,
        num_raw_transitions=ppo_num_raw_transitions + random_num_raw_transitions,
    )
    representation_count = count_trajectory_file(representation_path)
    mixed_manifest = {
        "stage": "stage1_mix_random",
        "representation_data": str(representation_path),
        "horizon": hdf5_horizon,
        "ppo_data_file": str(ppo_path),
        "random_data_file": str(random_path) if random_path is not None else None,
        "ppo_count": ppo_count,
        "random_count": random_count,
        "ppo_num_raw_transitions": ppo_num_raw_transitions,
        "random_seed": random_seed,
        "random_num_raw_transitions": random_num_raw_transitions,
        "random_rollout_steps": random_rollout_steps,
        "representation_count": representation_count,
        "actual_random_to_ppo_ratio": random_count["num_samples"] / max(1, ppo_count["num_samples"]),
        "actual_random_fraction": random_count["num_samples"] / max(1, ppo_count["num_samples"] + random_count["num_samples"]),
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
            "ppo_samples": ppo_count["num_samples"],
            "ppo_raw_transitions": ppo_num_raw_transitions,
            "random_samples": random_count["num_samples"],
            "random_raw_transitions": random_num_raw_transitions,
            "representation_samples": representation_count["num_samples"],
            "horizon": hdf5_horizon,
            "random_to_ppo_ratio": mixed_manifest["actual_random_to_ppo_ratio"],
            "random_fraction": mixed_manifest["actual_random_fraction"],
        },
        step=0,
    )
    log_wandb_artifact(
        wandb_run,
        name=f"stage1-mix-{output_dir.name}",
        artifact_type="stage1-mix-output",
        paths=[manifest_path],
    )
    finish_wandb_run(wandb_run)
    return mixed_manifest


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    print(json.dumps(run(args, ppo_config), indent=2))


if __name__ == "__main__":
    main()
