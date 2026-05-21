from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv

from config_utils import add_ppo_config_args, build_run_config, load_ppo_config, ppo_config_to_dict
from ppo_utils import (
    TrainingMetricsCallback,
    append_jsonl_record,
    build_ppo_model_from_config,
    evaluate_ppo_model,
    learn_ppo,
    make_vec_env_from_ppo_config,
)
from sb3_nas_policy import save_ppo_supernet_checkpoint
from supernet_backbone import SearchSpace
from trajectory_data import TrajectoryRecorderCallback, count_trajectory_file
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


PPO_REPRESENTATION_FILE = "ppo_representation_samples.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1A: train PPO with the maximum supernet subnet.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--output_dir", default="runs/stage1_ppo_max_arch", help="Directory for PPO trajectories, PPO checkpoint, and manifest.")
    parser.add_argument("--horizon", type=int, default=1, help="Number of future steps packed into each sampled representation record.")
    parser.add_argument("--sample_ratio", type=float, default=0.01, help="Sampling probability for eligible representation windows.")
    parser.add_argument("--sample_seed", type=int, default=0, help="Seed used for PPO representation sample selection.")
    parser.add_argument("--max_samples", type=int, default=0, help="Hard cap for saved representation samples; 0 disables the cap.")
    return parser.parse_args()


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        metrics_path: Path,
        last_checkpoint_path: Path,
        best_checkpoint_path: Path,
        search_space: SearchSpace,
        checkpoint_extra: dict[str, Any],
        log_fn,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self.metrics_path = metrics_path
        self.last_checkpoint_path = last_checkpoint_path
        self.best_checkpoint_path = best_checkpoint_path
        self.search_space = search_space
        self.checkpoint_extra = dict(checkpoint_extra)
        self.log_fn = log_fn
        self.next_eval_timestep = self.eval_freq
        self.best_mean_reward: float | None = None
        self.best_timestep: int | None = None

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_eval_episodes <= 0:
            return True
        if self.num_timesteps < self.next_eval_timestep:
            return True

        while self.next_eval_timestep <= self.num_timesteps:
            self.next_eval_timestep += self.eval_freq

        mean_return, std_return = evaluate_ppo_model(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
        )
        is_best = self.best_mean_reward is None or mean_return > self.best_mean_reward
        if is_best:
            self.best_mean_reward = float(mean_return)
            self.best_timestep = int(self.num_timesteps)
        record = {
            "type": "eval",
            "phase": "periodic",
            "total_timesteps": int(self.num_timesteps),
            "eval/mean_reward": float(mean_return),
            "eval/std_reward": float(std_return),
            "eval/is_best": bool(is_best),
        }
        append_jsonl_record(self.metrics_path, record)
        checkpoint_extra = dict(self.checkpoint_extra)
        checkpoint_extra.update(
            {
                "checkpoint_role": "last",
                "eval_mean_reward": float(mean_return),
                "eval_std_reward": float(std_return),
                "eval_timestep": int(self.num_timesteps),
                "total_timesteps": int(self.num_timesteps),
                "best_mean_reward": self.best_mean_reward,
                "best_timestep": self.best_timestep,
            }
        )
        save_ppo_supernet_checkpoint(
            self.model,
            self.last_checkpoint_path,
            search_space=self.search_space,
            extra=checkpoint_extra,
        )
        if is_best:
            best_extra = dict(checkpoint_extra)
            best_extra["checkpoint_role"] = "best"
            save_ppo_supernet_checkpoint(
                self.model,
                self.best_checkpoint_path,
                search_space=self.search_space,
                extra=best_extra,
            )
        print(
            (
                f"stage1_eval step={record['total_timesteps']} "
                f"mean_reward={record['eval/mean_reward']:.6g} "
                f"std_reward={record['eval/std_reward']:.6g}"
            ),
            flush=True,
        )
        if self.log_fn is not None:
            self.log_fn(
                {
                    "total_timesteps": record["total_timesteps"],
                    "eval/mean_reward": record["eval/mean_reward"],
                    "eval/std_reward": record["eval/std_reward"],
                    "eval/is_best": record["eval/is_best"],
                },
                int(self.num_timesteps),
            )
        return True


def run(args: argparse.Namespace, ppo_config: DictConfig) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run("stage1_train_max_ppo", run_config, output_dir)
    search_space = SearchSpace()
    max_arch = search_space.max_arch()
    search_space_path = output_dir / "search_space.json"
    search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))

    train_env = make_vec_env_from_ppo_config(ppo_config, seed=ppo_config.seed)
    eval_env = None
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "ppo_supernet_stage1.pt"
    last_checkpoint_path = output_dir / "ppo_supernet_stage1_last.pt"
    best_checkpoint_path = output_dir / "ppo_supernet_stage1_best.pt"
    eval_callback = None
    actual_total_timesteps = 0
    try:
        model = build_ppo_model_from_config(
            ppo_config=ppo_config,
            env=train_env,
            arch_config=max_arch,
        )
        representation_path = output_dir / PPO_REPRESENTATION_FILE

        def log_progress(values: dict[str, Any], step: int) -> None:
            log_wandb(wandb_run, values, step=step)

        trajectory_callback = TrajectoryRecorderCallback(
            save_path=representation_path,
            horizon=int(args.horizon),
            sample_ratio=float(args.sample_ratio),
            sample_seed=int(args.sample_seed),
            max_samples=int(args.max_samples) if int(args.max_samples) > 0 else None,
        )
        callbacks: list[BaseCallback] = [
            trajectory_callback,
            TrainingMetricsCallback(metrics_path, log_progress),
        ]
        eval_freq = int(getattr(ppo_config, "eval_freq", 0) or 0)
        eval_episodes = int(getattr(ppo_config, "eval_episodes", 0) or 0)
        checkpoint_extra = {
            "stage": "stage1_ppo_max_arch",
            "arch_config": max_arch.to_dict(),
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        }
        if eval_freq > 0 and eval_episodes > 0:
            eval_env = make_vec_env_from_ppo_config(
                ppo_config,
                seed=ppo_config.seed + 50,
                n_envs=int(ppo_config.eval_n_envs),
            )
            eval_callback = PeriodicEvalCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=eval_episodes,
                deterministic=bool(ppo_config.eval_deterministic),
                metrics_path=metrics_path,
                last_checkpoint_path=last_checkpoint_path,
                best_checkpoint_path=best_checkpoint_path,
                search_space=search_space,
                checkpoint_extra=checkpoint_extra,
                log_fn=log_progress,
            )
            callbacks.append(eval_callback)
        callback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
        learn_ppo(
            model,
            total_timesteps=ppo_config.total_timesteps,
            callback=callback,
            progress_bar=ppo_config.progress_bar,
        )
        actual_total_timesteps = int(model.num_timesteps)
        ppo_metadata = {
            "source": "ppo_training",
            "env_id": ppo_config.env_id,
            "seed": ppo_config.seed,
            "train_n_envs": ppo_config.train_n_envs,
            "total_timesteps": actual_total_timesteps,
            "image_size": int(ppo_config.image_size),
            "arch_config": max_arch.to_dict(),
            "search_space": search_space.to_dict(),
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        }
        representation_metadata = dict(ppo_metadata)
        representation_metadata.update(
            {
                "source": "ppo_rollout_representation_samples",
                "horizon": int(args.horizon),
                "sample_ratio": float(args.sample_ratio),
                "sample_seed": int(args.sample_seed),
                "max_samples": int(args.max_samples),
            }
        )
        trajectory_callback.save(metadata=representation_metadata)

        final_checkpoint_extra = dict(checkpoint_extra)
        final_checkpoint_extra["total_timesteps"] = actual_total_timesteps
        save_ppo_supernet_checkpoint(
            model,
            checkpoint_path,
            search_space=search_space,
            extra=final_checkpoint_extra,
        )
        last_extra = dict(final_checkpoint_extra)
        last_extra.update(
            {
                "checkpoint_role": "last",
                "best_mean_reward": None if eval_callback is None else eval_callback.best_mean_reward,
                "best_timestep": None if eval_callback is None else eval_callback.best_timestep,
            }
        )
        save_ppo_supernet_checkpoint(
            model,
            last_checkpoint_path,
            search_space=search_space,
            extra=last_extra,
        )
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    representation_count = count_trajectory_file(representation_path)
    manifest = {
        "stage": "stage1_ppo_max_arch",
        "representation_data": str(representation_path),
        "horizon": int(args.horizon),
        "sample_ratio": float(args.sample_ratio),
        "sample_seed": int(args.sample_seed),
        "max_samples": int(args.max_samples),
        "representation_count": representation_count,
        "total_timesteps": actual_total_timesteps,
        "configured_total_timesteps": int(ppo_config.total_timesteps),
        "ppo_supernet_checkpoint": str(checkpoint_path),
        "last_ppo_supernet_checkpoint": str(last_checkpoint_path),
        "best_ppo_supernet_checkpoint": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
        "best_eval_mean_reward": None if eval_callback is None else eval_callback.best_mean_reward,
        "best_eval_timestep": None if eval_callback is None else eval_callback.best_timestep,
        "checkpoint_fields": [
            "policy_state_dict",
            "search_space",
            "active_arch",
        ],
        "search_space": str(search_space_path),
        "metrics": str(metrics_path),
        "trajectory_count": representation_count,
        "max_arch": max_arch.to_dict(),
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log_wandb(
        wandb_run,
        {
            "representation_samples": representation_count["num_samples"],
            "representation_transitions": representation_count["num_transitions"],
            "representation_steps": representation_count["num_steps"],
            "representation_trajectories": representation_count.get("num_trajectories", representation_count["num_envs"]),
        },
        step=actual_total_timesteps,
    )
    artifact_paths = [checkpoint_path, search_space_path, manifest_path]
    if last_checkpoint_path.exists():
        artifact_paths.append(last_checkpoint_path)
    if best_checkpoint_path.exists():
        artifact_paths.append(best_checkpoint_path)
    if metrics_path.exists():
        artifact_paths.append(metrics_path)
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
