from __future__ import annotations

import argparse
import json
from pathlib import Path

from config_utils import add_ppo_config_args, load_ppo_config
from ppo_utils import build_ppo_model_from_args, learn_ppo, make_vec_env_from_args
from sb3_nas_policy import save_policy_backbone
from supernet_backbone import SearchSpace
from trajectory_data import TrajectoryRecorderCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1A: train PPO with the maximum supernet subnet.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--output_dir", default="runs/stage1_ppo_max_arch", help="Directory for PPO trajectories, backbone checkpoint, and manifest.")
    parser.add_argument("--save_ppo_model", action="store_true", help="Also save the complete SB3 PPO model zip.")
    args = parser.parse_args()
    load_ppo_config(args)
    return args


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_space = SearchSpace()
    max_arch = search_space.max_arch()
    search_space_path = output_dir / "search_space.json"
    search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))

    train_env = make_vec_env_from_args(args, seed=args.seed)
    try:
        model = build_ppo_model_from_args(
            args=args,
            env=train_env,
            arch_config=max_arch,
        )
        ppo_trajectory_path = output_dir / "ppo_train_trajectories.arrow"
        callback = TrajectoryRecorderCallback(save_path=ppo_trajectory_path)
        learn_ppo(
            model,
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=args.progress_bar,
        )
        ppo_metadata = {
            "source": "ppo_training",
            "env_id": args.env_id,
            "seed": args.seed,
            "train_n_envs": args.train_n_envs,
            "image_size": args.image_size,
            "arch_config": max_arch.to_dict(),
            "search_space": search_space.to_dict(),
            "args": vars(args),
        }
        callback.save(metadata=ppo_metadata)

        backbone_path = output_dir / "supernet_backbone_stage1.pt"
        save_policy_backbone(
            model.policy,
            backbone_path,
            search_space=search_space,
            extra={"stage": "stage1_ppo_max_arch", "arch_config": max_arch.to_dict(), "args": vars(args)},
        )
        ppo_model_path = output_dir / "ppo_max_supernet_model"
        if args.save_ppo_model:
            model.save(ppo_model_path)
    finally:
        train_env.close()

    manifest = {
        "stage": "stage1_ppo_max_arch",
        "ppo_trajectories": str(ppo_trajectory_path),
        "backbone_checkpoint": str(backbone_path),
        "ppo_model": str(ppo_model_path.with_suffix(".zip")) if args.save_ppo_model else None,
        "search_space": str(search_space_path),
        "max_arch": max_arch.to_dict(),
        "args": vars(args),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
