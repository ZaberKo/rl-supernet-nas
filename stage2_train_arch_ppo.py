from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch
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
    parse_schedule_value,
)
from sb3_nas_policy import configure_policy_optimizer, save_ppo_supernet_checkpoint
from supernet_backbone import ArchConfig, SearchSpace
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


DEFAULT_ARCH_CONFIG_PATH = Path(__file__).resolve().parent / "arch_configs" / "max_arch.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 diagnostic PPO finetune for a single architecture initialized from the learned supernet.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--arch_config", default=str(DEFAULT_ARCH_CONFIG_PATH), help="JSON file containing one ArchConfig payload.")
    parser.add_argument("--supernet_checkpoint", default="runs/stage2/supernet_backbone_stage2.pt", help="Stage2 supernet checkpoint used to initialize the PPO backbone.")
    parser.add_argument("--output_dir", default="runs/stage2_arch_ppo", help="Directory for PPO metrics, checkpoint, and manifest.")
    parser.add_argument("--candidate_timesteps", type=int, default=1024, help="PPO finetune timesteps for the single architecture; <=0 skips PPO updates.")
    parser.add_argument("--supernet_backbone_lr", type=float, default=0.0, help="Backbone learning rate during PPO finetune; <=0 freezes the inherited stage2 backbone.")
    parser.add_argument("--train_seed", type=int, default=None, help="Optional PPO train seed; defaults to ppo.seed.")
    parser.add_argument("--eval_seed", type=int, default=None, help="Optional evaluation seed; defaults to train_seed plus eval_seed_offset.")
    parser.add_argument("--eval_seed_offset", type=int, default=50, help="Offset used when eval_seed is not set.")
    parser.add_argument("--initial_eval", action=argparse.BooleanOptionalAction, default=True, help="Evaluate the initialized policy before PPO finetuning.")
    parser.add_argument("--final_eval", action=argparse.BooleanOptionalAction, default=True, help="Evaluate the policy after PPO finetuning.")
    parser.add_argument("--save_checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Save the final PPO policy checkpoint.")
    return parser.parse_args()


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        metrics_path: Path,
        log_fn,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self.metrics_path = metrics_path
        self.log_fn = log_fn
        self.next_eval_timestep = self.eval_freq

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_eval_episodes <= 0:
            return True
        if self.num_timesteps < self.next_eval_timestep:
            return True

        while self.next_eval_timestep <= self.num_timesteps:
            self.next_eval_timestep += self.eval_freq

        record = evaluate_and_record(
            model=self.model,
            eval_env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            metrics_path=self.metrics_path,
            log_fn=self.log_fn,
            num_timesteps=int(self.num_timesteps),
            phase="periodic",
        )
        print(
            (
                f"stage2_arch_eval step={record['total_timesteps']} "
                f"mean_reward={record['eval/mean_reward']:.6g} "
                f"std_reward={record['eval/std_reward']:.6g}"
            ),
            flush=True,
        )
        return True


def extract_arch_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("arch_config", "arch"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def load_arch_config(path: str | Path) -> ArchConfig:
    arch_path = Path(path)
    if not arch_path.exists():
        raise FileNotFoundError(f"Architecture config does not exist: {arch_path}")
    payload = json.loads(arch_path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("Architecture config JSON must contain a mapping.")
    return ArchConfig.from_dict(dict(extract_arch_payload(payload)))


def validate_arch_config(search_space: SearchSpace, arch_config: ArchConfig) -> None:
    if len(arch_config.stage_depths) != search_space.num_stages:
        raise ValueError("Architecture stage count does not match the search space.")
    if len(arch_config.layer_configs) != search_space.num_stages:
        raise ValueError("Architecture layer config stage count does not match the search space.")

    for stage_index, (depth, candidates, stage_layers, max_depth) in enumerate(
        zip(
            arch_config.stage_depths,
            search_space.stage_depth_candidates,
            arch_config.layer_configs,
            search_space.max_stage_depths,
        )
    ):
        if int(depth) not in candidates:
            raise ValueError(f"Stage {stage_index} depth is not in the search space.")
        if len(stage_layers) != max_depth:
            raise ValueError(f"Stage {stage_index} must contain {max_depth} layer configs.")
        for layer_index, layer_config in enumerate(stage_layers):
            if int(layer_config.kernel_size) not in search_space.kernel_size_candidates:
                raise ValueError(f"Stage {stage_index} layer {layer_index} kernel size is not in the search space.")
            if int(layer_config.expand_ratio) not in search_space.expand_ratio_candidates:
                raise ValueError(f"Stage {stage_index} layer {layer_index} expand ratio is not in the search space.")


def evaluate_and_record(
    model,
    eval_env: VecEnv,
    n_eval_episodes: int,
    deterministic: bool,
    metrics_path: Path,
    log_fn,
    num_timesteps: int,
    phase: str,
) -> dict[str, Any]:
    mean_return, std_return = evaluate_ppo_model(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )
    record = {
        "type": "eval",
        "phase": phase,
        "total_timesteps": int(num_timesteps),
        "eval/mean_reward": float(mean_return),
        "eval/std_reward": float(std_return),
    }
    append_jsonl_record(metrics_path, record)
    if log_fn is not None:
        log_fn(
            {
                "total_timesteps": record["total_timesteps"],
                "eval/mean_reward": record["eval/mean_reward"],
                "eval/std_reward": record["eval/std_reward"],
            },
            int(num_timesteps),
        )
    return record


def run(args: argparse.Namespace, ppo_config: DictConfig) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run("stage2_train_arch_ppo", run_config, output_dir)
    search_space = SearchSpace()
    search_space_path = output_dir / "search_space.json"
    search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))

    arch_config = load_arch_config(args.arch_config)
    validate_arch_config(search_space, arch_config)
    supernet_checkpoint_path = Path(args.supernet_checkpoint)
    if not supernet_checkpoint_path.exists():
        raise FileNotFoundError(f"Stage2 supernet checkpoint does not exist: {supernet_checkpoint_path}")

    train_seed = int(ppo_config.seed) if args.train_seed is None else int(args.train_seed)
    eval_seed = train_seed + int(args.eval_seed_offset) if args.eval_seed is None else int(args.eval_seed)
    torch.manual_seed(train_seed)

    metrics_path = output_dir / "metrics.jsonl"

    train_env = make_vec_env_from_ppo_config(ppo_config, seed=train_seed)
    eval_env = None
    checkpoint_path = output_dir / "ppo_supernet_stage2_arch.pt"
    initial_eval_record = None
    final_eval_record = None
    actual_timesteps = 0
    training_metrics_callback = None
    try:
        model = build_ppo_model_from_config(
            ppo_config=ppo_config,
            env=train_env,
            arch_config=arch_config,
            backbone_checkpoint_path=str(supernet_checkpoint_path),
            learning_rate_attr="head_lr",
            model_seed=train_seed,
        )
        configure_policy_optimizer(
            model,
            head_lr=parse_schedule_value(ppo_config.head_lr),
            backbone_lr=float(args.supernet_backbone_lr),
        )

        eval_episodes = int(getattr(ppo_config, "eval_episodes", 0) or 0)
        eval_deterministic = bool(getattr(ppo_config, "eval_deterministic", True))
        eval_freq = int(getattr(ppo_config, "eval_freq", 0) or 0)

        def log_progress(values: dict[str, Any], step: int) -> None:
            log_wandb(wandb_run, values, step=step)

        if eval_episodes > 0:
            eval_env = make_vec_env_from_ppo_config(
                ppo_config,
                seed=eval_seed,
                n_envs=int(ppo_config.eval_n_envs),
            )
            if bool(args.initial_eval):
                initial_eval_record = evaluate_and_record(
                    model=model,
                    eval_env=eval_env,
                    n_eval_episodes=eval_episodes,
                    deterministic=eval_deterministic,
                    metrics_path=metrics_path,
                    log_fn=log_progress,
                    num_timesteps=0,
                    phase="initial",
                )

        training_metrics_callback = TrainingMetricsCallback(metrics_path, log_progress)
        callbacks: list[BaseCallback] = [training_metrics_callback]
        if eval_env is not None and eval_freq > 0:
            callbacks.append(
                PeriodicEvalCallback(
                    eval_env=eval_env,
                    eval_freq=eval_freq,
                    n_eval_episodes=eval_episodes,
                    deterministic=eval_deterministic,
                    metrics_path=metrics_path,
                    log_fn=log_progress,
                )
            )
        callback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
        learn_ppo(
            model,
            total_timesteps=int(args.candidate_timesteps),
            callback=callback,
            progress_bar=bool(ppo_config.progress_bar),
        )
        actual_timesteps = int(model.num_timesteps)

        if eval_env is not None and bool(args.final_eval):
            final_eval_record = evaluate_and_record(
                model=model,
                eval_env=eval_env,
                n_eval_episodes=eval_episodes,
                deterministic=eval_deterministic,
                metrics_path=metrics_path,
                log_fn=log_progress,
                num_timesteps=int(model.num_timesteps),
                phase="final",
            )

        backbone = model.policy.features_extractor.backbone
        policy_params = int(sum(parameter.numel() for parameter in model.policy.parameters()))
        active_backbone_params = int(backbone.elastic_num_params)
        if bool(args.save_checkpoint):
            save_ppo_supernet_checkpoint(
                model,
                checkpoint_path,
                search_space=search_space,
                extra={
                    "stage": "stage2_arch_ppo",
                    "arch_config": arch_config.to_dict(),
                    "supernet_checkpoint": str(supernet_checkpoint_path),
                    "candidate_timesteps": int(args.candidate_timesteps),
                    "actual_timesteps": actual_timesteps,
                    "supernet_backbone_lr": float(args.supernet_backbone_lr),
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "args": vars(args),
                    "ppo_config": ppo_config_to_dict(ppo_config),
                },
            )
    finally:
        if training_metrics_callback is not None and training_metrics_callback.writer is not None:
            training_metrics_callback.writer.close()
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    manifest = {
        "stage": "stage2_arch_ppo",
        "arch_config": arch_config.to_dict(),
        "arch_config_path": str(args.arch_config),
        "supernet_checkpoint": str(supernet_checkpoint_path),
        "ppo_checkpoint": str(checkpoint_path) if bool(args.save_checkpoint) else None,
        "metrics": str(metrics_path),
        "search_space": str(search_space_path),
        "candidate_timesteps": int(args.candidate_timesteps),
        "actual_timesteps": actual_timesteps,
        "supernet_backbone_lr": float(args.supernet_backbone_lr),
        "active_backbone_params": active_backbone_params,
        "policy_params": policy_params,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "initial_eval": initial_eval_record,
        "final_eval": final_eval_record,
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log_wandb(
        wandb_run,
        {
            "candidate_timesteps": int(args.candidate_timesteps),
            "actual_timesteps": actual_timesteps,
            "active_backbone_params": active_backbone_params,
            "policy_params": policy_params,
        },
        step=actual_timesteps,
    )
    artifact_paths = [metrics_path, search_space_path, manifest_path]
    if bool(args.save_checkpoint) and checkpoint_path.exists():
        artifact_paths.append(checkpoint_path)
    log_wandb_artifact(
        wandb_run,
        name=f"stage2-arch-ppo-{output_dir.name}",
        artifact_type="stage2-arch-ppo-output",
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
