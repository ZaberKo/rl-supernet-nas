from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import DictConfig

from config_utils import add_ppo_config_args, build_run_config, load_ppo_config, ppo_config_to_dict
from ppo_utils import build_ppo_model_from_config, learn_ppo, make_vec_env_from_ppo_config
from sb3_nas_policy import save_policy_backbone
from supernet_backbone import SearchSpace
from trajectory_data import TrajectoryRecorderCallback, count_trajectory_file
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1A: train PPO with the maximum supernet subnet.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--output_dir", default="runs/stage1_ppo_max_arch", help="Directory for PPO trajectories, backbone checkpoint, and manifest.")
    parser.add_argument("--save_ppo_model", default=None, help="Path for saving the complete SB3 PPO model zip. Defaults to output_dir/ppo_max_supernet_model.zip.")
    return parser.parse_args()


def resolve_ppo_model_path(path: str | None, output_dir: Path) -> Path:
    ppo_model_path = output_dir / "ppo_max_supernet_model.zip" if path is None else Path(path)
    if ppo_model_path.suffix != ".zip":
        ppo_model_path = ppo_model_path.with_name(f"{ppo_model_path.name}.zip")
    return ppo_model_path


def run(args: argparse.Namespace, ppo_config: DictConfig) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_model_path = resolve_ppo_model_path(args.save_ppo_model, output_dir)
    args.save_ppo_model = str(ppo_model_path)
    ppo_model_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run("stage1_train_max_ppo", run_config, output_dir)
    search_space = SearchSpace()
    max_arch = search_space.max_arch()
    search_space_path = output_dir / "search_space.json"
    search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))

    train_env = make_vec_env_from_ppo_config(ppo_config, seed=ppo_config.seed)
    try:
        model = build_ppo_model_from_config(
            ppo_config=ppo_config,
            env=train_env,
            arch_config=max_arch,
        )
        ppo_trajectory_path = output_dir / "ppo_train_trajectories.arrow"

        def log_training_progress(values: dict[str, float | int], step: int) -> None:
            log_wandb(wandb_run, {f"stage1/{key}": value for key, value in values.items()}, step=step)

        callback = TrajectoryRecorderCallback(
            save_path=ppo_trajectory_path,
            log_fn=log_training_progress,
            log_interval=max(1, int(ppo_config.n_steps)),
        )
        learn_ppo(
            model,
            total_timesteps=ppo_config.total_timesteps,
            callback=callback,
            progress_bar=ppo_config.progress_bar,
        )
        ppo_metadata = {
            "source": "ppo_training",
            "env_id": ppo_config.env_id,
            "seed": ppo_config.seed,
            "train_n_envs": ppo_config.train_n_envs,
            "image_size": ppo_config.image_size,
            "arch_config": max_arch.to_dict(),
            "search_space": search_space.to_dict(),
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        }
        callback.save(metadata=ppo_metadata)

        backbone_path = output_dir / "supernet_backbone_stage1.pt"
        save_policy_backbone(
            model.policy,
            backbone_path,
            search_space=search_space,
            extra={
                "stage": "stage1_ppo_max_arch",
                "arch_config": max_arch.to_dict(),
                "args": vars(args),
                "ppo_config": ppo_config_to_dict(ppo_config),
            },
        )
        model.save(ppo_model_path)
    finally:
        train_env.close()

    ppo_count = count_trajectory_file(ppo_trajectory_path)
    manifest = {
        "stage": "stage1_ppo_max_arch",
        "ppo_trajectories": str(ppo_trajectory_path),
        "backbone_checkpoint": str(backbone_path),
        "ppo_model": str(ppo_model_path),
        "search_space": str(search_space_path),
        "trajectory_count": ppo_count,
        "max_arch": max_arch.to_dict(),
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log_wandb(
        wandb_run,
        {
            "stage1/num_transitions": ppo_count["num_transitions"],
            "stage1/num_steps": ppo_count["num_steps"],
            "stage1/num_trajectories": ppo_count.get("num_trajectories", ppo_count["num_envs"]),
        },
        step=int(ppo_config.total_timesteps),
    )
    artifact_paths = [ppo_trajectory_path, backbone_path, search_space_path, manifest_path]
    artifact_paths.append(ppo_model_path)
    log_wandb_artifact(
        wandb_run,
        name=f"stage1-ppo-{output_dir.name}",
        artifact_type="stage1-output",
        paths=artifact_paths,
    )
    finish_wandb_run(wandb_run)
    return manifest


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    print(json.dumps(run(args, ppo_config), indent=2))


if __name__ == "__main__":
    main()
