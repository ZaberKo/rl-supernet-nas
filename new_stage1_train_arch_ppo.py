from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from tqdm.auto import tqdm

from config_utils import add_ppo_config_args, build_run_config, load_ppo_config, ppo_config_to_dict
from new_stage1_train_policy_supernet import DynamicsRolloutBuffer, build_sb3_critic_model, evaluate_actor_subnet
from checkpoint_utils import (
    build_network_ppo_config,
    build_policy_from_checkpoint,
    load_checkpoint,
    load_critic_from_checkpoint,
    validate_checkpoint_search_space,
)
from new_stage2_ea_search import (
    actor_head_parameters,
    collect_candidate_rollout,
    configure_actor_optimizer,
    count_parameters,
    critic_update,
    fixed_arch_actor_update,
    prefixed_metrics,
    set_global_seeds,
    update_actor_optimizer_learning_rate,
    update_optimizer_learning_rate,
)
from ppo_utils import (
    append_jsonl_record,
    make_vec_env_from_ppo_config,
    parse_optional_float,
    parse_schedule_value,
    resolve_device,
)
from supernet_backbone import ArchConfig, SearchSpace
from env_utils import EVAL_SEED_OFFSET
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb


DEFAULT_ARCH_CONFIG_PATH = Path(__file__).resolve().parent / "arch_configs" / "max_arch.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="New stage 1 single-architecture PPO finetune diagnostic initialized from a policy supernet checkpoint.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--arch_config", default=str(DEFAULT_ARCH_CONFIG_PATH), help="JSON file containing one ArchConfig object.")
    parser.add_argument(
        "--supernet_checkpoint",
        default="runs/new_stage1_policy_supernet/policy_supernet_best.pt",
        help="New stage1 policy-supernet checkpoint used to initialize the actor supernet and critic.",
    )
    parser.add_argument("--output_dir", default="runs/new_stage1_arch_ppo", help="Directory for PPO metrics, checkpoint, and manifest.")

    parser.add_argument("--suffix", default="", help="Optional suffix to append to the stage name.")
    args = parser.parse_args()
    return args


def extract_arch_config(config_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("arch_config", "arch"):
        value = config_dict.get(key)
        if isinstance(value, Mapping):
            return value
    return config_dict


def load_arch_config(path: str | Path) -> ArchConfig:
    arch_path = Path(path)
    if not arch_path.exists():
        raise FileNotFoundError(f"Architecture config does not exist: {arch_path}")
    config_dict = json.loads(arch_path.read_text())
    if not isinstance(config_dict, Mapping):
        raise ValueError("Architecture config JSON must contain a mapping.")
    return ArchConfig.from_dict(dict(extract_arch_config(config_dict)))


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
            strict=True,
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


def numeric_values(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if isinstance(value, (int, float, bool))
    }


def log_record(metrics_path: Path, wandb_run: Any, record: Mapping[str, Any], step: int) -> None:
    append_jsonl_record(metrics_path, record)
    log_wandb(wandb_run, numeric_values(record), step=step)


def write_progress(progress_bar: tqdm | None, message: str) -> None:
    if progress_bar is None:
        print(message, flush=True)
        return
    progress_bar.write(message)


def update_progress_bar(progress_bar: tqdm | None, configured_total: int, current_total: int) -> None:
    if progress_bar is None:
        return
    bounded_total = min(max(0, int(configured_total)), max(0, int(current_total)))
    progress_bar.update(max(0, bounded_total - int(progress_bar.n)))


def evaluate_and_record(
    policy,
    train_env: VecEnv,
    eval_env: VecEnv,
    arch_config: ArchConfig,
    n_eval_episodes: int,
    deterministic: bool,
    device: torch.device,
    metrics_path: Path,
    wandb_run: Any,
    num_timesteps: int,
    total_env_timesteps: int,
    phase: str,
) -> dict[str, Any]:
    eval_metrics = evaluate_actor_subnet(
        policy=policy,
        train_env=train_env,
        eval_env=eval_env,
        arch=arch_config,
        n_eval_episodes=int(n_eval_episodes),
        deterministic=bool(deterministic),
        device=device,
    )
    record = {
        "type": "eval",
        "phase": phase,
        "total_timesteps": int(num_timesteps),
        "total_env_timesteps": int(total_env_timesteps),
        "eval/ep_return": float(eval_metrics["ep_return"]),
        "eval/ep_return_std": float(eval_metrics["ep_return_std"]),
        "eval/ep_length": float(eval_metrics["ep_length"]),
        "eval/ep_length_std": float(eval_metrics["ep_length_std"]),
    }
    log_record(metrics_path, wandb_run, record, step=int(total_env_timesteps))
    return record


def save_arch_checkpoint(
    path: Path,
    args: argparse.Namespace,
    ppo_config: DictConfig,
    policy,
    critic_model: PPO,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    search_space: SearchSpace,
    arch_config: ArchConfig,
    checkpoint: Mapping[str, Any],
    actual_timesteps: int,
    critic_warmup_timesteps: int,
    total_env_timesteps: int,
    stage_name: str,
) -> None:
    torch.save(
        {
            "stage": stage_name,
            "source_stage": checkpoint.get("stage"),
            "total_timesteps": int(actual_timesteps),
            "critic_warmup_timesteps": int(critic_warmup_timesteps),
            "total_env_timesteps": int(total_env_timesteps),
            "policy_state_dict": policy.state_dict(),
            "critic_policy_state_dict": critic_model.policy.state_dict(),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "search_space": search_space.to_dict(),
            "arch_config": arch_config.to_dict(),
            "supernet_checkpoint": str(args.supernet_checkpoint),
            "features_dim": checkpoint.get("features_dim", ppo_config.features_dim),
            "projection_dim": checkpoint.get("projection_dim", ppo_config.projection_dim),
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        },
        path,
    )


def run(args: argparse.Namespace, ppo_config: DictConfig) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    search_space_path = output_dir / "search_space.json"
    last_checkpoint_path = output_dir / "policy_supernet_arch_ppo_last.pt"
    best_checkpoint_path = output_dir / "policy_supernet_arch_ppo_best.pt"

    stage_name = f"new_stage1_train_arch_ppo_{args.suffix}" if getattr(args, "suffix", "") else "new_stage1_train_arch_ppo"
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run(stage_name, run_config, output_dir)
    train_env = None
    eval_env = None
    progress_bar = None

    try:
        search_space = SearchSpace()
        search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))
        arch_config = load_arch_config(args.arch_config)
        validate_arch_config(search_space, arch_config)

        train_seed = int(ppo_config.seed)
        eval_seed = train_seed + EVAL_SEED_OFFSET
        set_global_seeds(train_seed)
        device = resolve_device(str(ppo_config.device))

        checkpoint = load_checkpoint(args.supernet_checkpoint, map_location=device)
        validate_checkpoint_search_space(checkpoint, search_space)

        train_env = make_vec_env_from_ppo_config(ppo_config, seed=train_seed, n_envs=ppo_config.train_n_envs)
        eval_episodes = ppo_config.eval_episodes
        eval_deterministic = ppo_config.eval_deterministic
        eval_freq = ppo_config.eval_freq
        if eval_episodes > 0:
            eval_env = make_vec_env_from_ppo_config(ppo_config, seed=eval_seed, n_envs=ppo_config.eval_n_envs)

        policy = build_policy_from_checkpoint(
            args=args,
            ppo_config=ppo_config,
            train_env=train_env,
            search_space=search_space,
            checkpoint=checkpoint,
            device=device,
        )
        policy.set_active_arch(arch_config)

        critic_lr_schedule = parse_schedule_value(ppo_config.critic_lr)
        policy_head_lr_schedule = parse_schedule_value(ppo_config.policy_head_lr)
        policy_backbone_lr_schedule = parse_schedule_value(ppo_config.policy_backbone_lr)
        clip_range_schedule = parse_schedule_value(ppo_config.clip_range)
        actor_optimizer = configure_actor_optimizer(
            policy=policy,
            head_lr=policy_head_lr_schedule,
            backbone_lr=policy_backbone_lr_schedule,
        )

        critic_ppo_config = build_network_ppo_config(checkpoint.get("ppo_config", {}), ppo_config, seed=train_seed)
        critic_model = build_sb3_critic_model(
            ppo_config=critic_ppo_config,
            env=train_env,
            learning_rate=critic_lr_schedule,
        )
        loaded_critic = load_critic_from_checkpoint(critic_model, checkpoint)
        critic_optimizer = critic_model.policy.optimizer

        rollout_buffer = DynamicsRolloutBuffer(
            buffer_size=int(ppo_config.n_steps),
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            device=device,
            gae_lambda=float(ppo_config.gae_lambda),
            gamma=float(ppo_config.gamma),
            n_envs=int(train_env.num_envs),
        )

        append_jsonl_record(
            metrics_path,
            {
                "type": "config",
                "stage": stage_name,
                "supernet_checkpoint": str(args.supernet_checkpoint),
                "arch_config_path": str(args.arch_config),
                "candidate_timesteps": ppo_config.total_timesteps,
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "device": str(device),
                "loaded_critic": bool(loaded_critic),
                "observation_shape": list(train_env.observation_space.shape),
                "action_space": type(train_env.action_space).__name__,
            },
        )

        configured_progress_total = max(0, ppo_config.total_timesteps)
        if configured_progress_total > 0:
            progress_bar = tqdm(
                total=configured_progress_total,
                desc="new_stage1_arch_ppo",
                unit="step",
                dynamic_ncols=True,
                disable=ppo_config.quiet,
            )

        initial_eval_record = None
        final_eval_record = None
        actual_timesteps = 0
        critic_warmup_actual_timesteps = 0
        total_env_timesteps = 0
        best_eval_ep_return: float | None = None
        best_eval_record: dict[str, Any] | None = None

        def maybe_save_best_checkpoint(record: dict[str, Any], current_actual_timesteps: int, current_total_env_timesteps: int) -> None:
            nonlocal best_eval_ep_return, best_eval_record
            ep_return = float(record["eval/ep_return"])
            if best_eval_ep_return is None or ep_return > best_eval_ep_return:
                best_eval_ep_return = ep_return
                best_eval_record = dict(record)
                save_arch_checkpoint(
                    best_checkpoint_path,
                    args=args,
                    ppo_config=ppo_config,
                    policy=policy,
                    critic_model=critic_model,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    search_space=search_space,
                    arch_config=arch_config,
                    checkpoint=checkpoint,
                    actual_timesteps=current_actual_timesteps,
                    critic_warmup_timesteps=critic_warmup_actual_timesteps,
                    total_env_timesteps=current_total_env_timesteps,
                    stage_name=stage_name,
                )

        if eval_env is not None:
            initial_eval_record = evaluate_and_record(
                policy=policy,
                train_env=train_env,
                eval_env=eval_env,
                arch_config=arch_config,
                n_eval_episodes=eval_episodes,
                deterministic=eval_deterministic,
                device=device,
                metrics_path=metrics_path,
                wandb_run=wandb_run,
                num_timesteps=0,
                total_env_timesteps=0,
                phase="initial",
            )
            write_progress(
                progress_bar,
                (
                    "new_stage1_arch_eval phase=initial step=0 "
                    f"ep_return={initial_eval_record['eval/ep_return']:.6g} "
                    f"ep_return_std={initial_eval_record['eval/ep_return_std']:.6g} "
                    f"ep_length={initial_eval_record['eval/ep_length']:.6g}"
                ),
            )
            maybe_save_best_checkpoint(initial_eval_record, 0, 0)

        observation = train_env.reset()
        episode_starts = np.ones((train_env.num_envs,), dtype=np.bool_)
        warmup_target = 0
        warmup_iteration = 0
        while critic_warmup_actual_timesteps < warmup_target:
            warmup_iteration += 1
            progress_remaining = 1.0 - float(critic_warmup_actual_timesteps) / float(max(1, warmup_target))
            critic_lr = float(critic_lr_schedule(progress_remaining)) if callable(critic_lr_schedule) else float(critic_lr_schedule)
            update_optimizer_learning_rate(critic_optimizer, critic_lr)
            observation, episode_starts, rollout_metrics = collect_candidate_rollout(
                policy=policy,
                critic_model=critic_model,
                env=train_env,
                arch=arch_config,
                rollout_buffer=rollout_buffer,
                initial_observation=observation,
                initial_episode_starts=episode_starts,
                n_steps=int(ppo_config.n_steps),
                gamma=float(ppo_config.gamma),
                device=device,
            )
            critic_warmup_actual_timesteps += int(ppo_config.n_steps) * int(train_env.num_envs)
            total_env_timesteps = critic_warmup_actual_timesteps + actual_timesteps
            critic_metrics = critic_update(
                critic_model=critic_model,
                optimizer=critic_optimizer,
                rollout_buffer=rollout_buffer,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                max_grad_norm=float(ppo_config.max_grad_norm),
            )
            record = {
                "type": "critic_warmup",
                "iteration": int(warmup_iteration),
                "total_timesteps": int(actual_timesteps),
                "critic_warmup_timesteps": int(critic_warmup_actual_timesteps),
                "total_env_timesteps": int(total_env_timesteps),
                "progress_remaining": float(progress_remaining),
                "critic_lr": float(critic_lr),
                **prefixed_metrics("critic_warmup_rollout", rollout_metrics),
                **prefixed_metrics("critic_warmup", critic_metrics),
            }
            log_record(metrics_path, wandb_run, record, step=total_env_timesteps)
            update_progress_bar(progress_bar, configured_progress_total, total_env_timesteps)
            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "phase": "warmup",
                        "ret": f"{rollout_metrics['rollout/ep_return']:.3g}",
                        "critic": f"{critic_metrics['critic/loss']:.3g}",
                        "lr": f"{critic_lr:.2g}",
                    },
                    refresh=True,
                )

        target_timesteps = max(0, ppo_config.total_timesteps)
        target_kl = parse_optional_float(ppo_config.target_kl)
        next_eval_timestep = eval_freq if eval_freq > 0 else 0
        train_iteration = 0
        last_train_record: dict[str, Any] = {}
        while actual_timesteps < target_timesteps:
            train_iteration += 1
            progress_remaining = 1.0 - float(actual_timesteps) / float(max(1, target_timesteps))
            actor_lr = float(policy_head_lr_schedule(progress_remaining)) if callable(policy_head_lr_schedule) else float(policy_head_lr_schedule)
            backbone_lr = float(policy_backbone_lr_schedule(progress_remaining)) if callable(policy_backbone_lr_schedule) else float(policy_backbone_lr_schedule)
            critic_lr = float(critic_lr_schedule(progress_remaining)) if callable(critic_lr_schedule) else float(critic_lr_schedule)
            clip_range = float(clip_range_schedule(progress_remaining)) if callable(clip_range_schedule) else float(clip_range_schedule)
            update_actor_optimizer_learning_rate(actor_optimizer, actor_lr, backbone_lr)
            update_optimizer_learning_rate(critic_optimizer, critic_lr)

            observation, episode_starts, rollout_metrics = collect_candidate_rollout(
                policy=policy,
                critic_model=critic_model,
                env=train_env,
                arch=arch_config,
                rollout_buffer=rollout_buffer,
                initial_observation=observation,
                initial_episode_starts=episode_starts,
                n_steps=int(ppo_config.n_steps),
                gamma=float(ppo_config.gamma),
                device=device,
            )
            actual_timesteps += int(ppo_config.n_steps) * int(train_env.num_envs)
            total_env_timesteps = critic_warmup_actual_timesteps + actual_timesteps
            actor_metrics = fixed_arch_actor_update(
                policy=policy,
                actor_optimizer=actor_optimizer,
                rollout_buffer=rollout_buffer,
                arch=arch_config,
                action_space=train_env.action_space,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                clip_range=clip_range,
                normalize_advantage=bool(ppo_config.normalize_advantage),
                ent_coef=float(ppo_config.ent_coef),
                max_grad_norm=float(ppo_config.max_grad_norm),
                target_kl=target_kl,
            )
            critic_metrics = critic_update(
                critic_model=critic_model,
                optimizer=critic_optimizer,
                rollout_buffer=rollout_buffer,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                max_grad_norm=float(ppo_config.max_grad_norm),
            )
            train_record = {
                "type": "train",
                "iteration": int(train_iteration),
                "total_timesteps": int(actual_timesteps),
                "critic_warmup_timesteps": int(critic_warmup_actual_timesteps),
                "total_env_timesteps": int(total_env_timesteps),
                "progress_remaining": float(progress_remaining),
                "actor_lr": float(actor_lr),
                "critic_lr": float(critic_lr),
                "backbone_lr": float(backbone_lr),
                "clip_range": float(clip_range),
                **rollout_metrics,
                **actor_metrics,
                **critic_metrics,
            }
            last_train_record = dict(train_record)
            log_record(metrics_path, wandb_run, train_record, step=total_env_timesteps)
            update_progress_bar(progress_bar, configured_progress_total, total_env_timesteps)
            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "phase": "train",
                        "ret": f"{rollout_metrics['rollout/ep_return']:.3g}",
                        "actor": f"{actor_metrics['actor/loss']:.3g}",
                        "critic": f"{critic_metrics['critic/loss']:.3g}",
                        "lr": f"{actor_lr:.2g}",
                    },
                    refresh=True,
                )

            if eval_env is not None and next_eval_timestep > 0 and actual_timesteps >= next_eval_timestep:
                while next_eval_timestep <= actual_timesteps:
                    next_eval_timestep += eval_freq
                eval_record = evaluate_and_record(
                    policy=policy,
                    train_env=train_env,
                    eval_env=eval_env,
                    arch_config=arch_config,
                    n_eval_episodes=eval_episodes,
                    deterministic=eval_deterministic,
                    device=device,
                    metrics_path=metrics_path,
                    wandb_run=wandb_run,
                    num_timesteps=actual_timesteps,
                    total_env_timesteps=total_env_timesteps,
                    phase="periodic",
                )
                write_progress(
                    progress_bar,
                    (
                        f"new_stage1_arch_eval phase=periodic step={actual_timesteps} "
                        f"ep_return={eval_record['eval/ep_return']:.6g} "
                        f"ep_return_std={eval_record['eval/ep_return_std']:.6g} "
                        f"ep_length={eval_record['eval/ep_length']:.6g}"
                    ),
                )
                maybe_save_best_checkpoint(eval_record, actual_timesteps, total_env_timesteps)

            save_arch_checkpoint(
                last_checkpoint_path,
                args=args,
                ppo_config=ppo_config,
                policy=policy,
                critic_model=critic_model,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                search_space=search_space,
                arch_config=arch_config,
                checkpoint=checkpoint,
                actual_timesteps=actual_timesteps,
                critic_warmup_timesteps=critic_warmup_actual_timesteps,
                total_env_timesteps=total_env_timesteps,
                stage_name=stage_name,
            )

        if eval_env is not None:
            final_eval_record = evaluate_and_record(
                policy=policy,
                train_env=train_env,
                eval_env=eval_env,
                arch_config=arch_config,
                n_eval_episodes=eval_episodes,
                deterministic=eval_deterministic,
                device=device,
                metrics_path=metrics_path,
                wandb_run=wandb_run,
                num_timesteps=actual_timesteps,
                total_env_timesteps=total_env_timesteps,
                phase="final",
            )
            write_progress(
                progress_bar,
                (
                    f"new_stage1_arch_eval phase=final step={actual_timesteps} "
                    f"ep_return={final_eval_record['eval/ep_return']:.6g} "
                    f"ep_return_std={final_eval_record['eval/ep_return_std']:.6g} "
                    f"ep_length={final_eval_record['eval/ep_length']:.6g}"
                ),
            )
            maybe_save_best_checkpoint(final_eval_record, actual_timesteps, total_env_timesteps)

        policy.set_active_arch(arch_config)
        active_backbone_params = int(policy.backbone.elastic_num_params)
        actor_head_params = count_parameters(actor_head_parameters(policy))
        policy_params = int(sum(parameter.numel() for parameter in policy.parameters()))
        trainable_policy_params = int(sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad))

        save_arch_checkpoint(
            last_checkpoint_path,
            args=args,
            ppo_config=ppo_config,
            policy=policy,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            search_space=search_space,
            arch_config=arch_config,
            checkpoint=checkpoint,
            actual_timesteps=actual_timesteps,
            critic_warmup_timesteps=critic_warmup_actual_timesteps,
            total_env_timesteps=total_env_timesteps,
            stage_name=stage_name,
        )

        manifest = {
            "stage": stage_name,
            "source_stage": checkpoint.get("stage"),
            "arch_config": arch_config.to_dict(),
            "arch_config_path": str(args.arch_config),
            "supernet_checkpoint": str(args.supernet_checkpoint),
            "last_checkpoint": str(last_checkpoint_path),
            "best_checkpoint": str(best_checkpoint_path),
            "metrics": str(metrics_path),
            "search_space": str(search_space_path),
            "configured_candidate_timesteps": ppo_config.total_timesteps,
            "critic_warmup_actual_timesteps": int(critic_warmup_actual_timesteps),
            "total_env_timesteps": int(total_env_timesteps),
            "active_backbone_params": active_backbone_params,
            "actor_head_params": actor_head_params,
            "policy_params": policy_params,
            "trainable_policy_params": trainable_policy_params,
            "loaded_critic": bool(loaded_critic),
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "initial_eval": initial_eval_record,
            "final_eval": final_eval_record,
            "best_eval_ep_return": best_eval_ep_return,
            "best_eval": best_eval_record,
            "last_train": last_train_record,
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log_wandb(
            wandb_run,
            {
                "actual_timesteps": int(actual_timesteps),
                "critic_warmup_actual_timesteps": int(critic_warmup_actual_timesteps),
                "total_env_timesteps": int(total_env_timesteps),
                "active_backbone_params": active_backbone_params,
                "actor_head_params": actor_head_params,
                "policy_params": policy_params,
                "trainable_policy_params": trainable_policy_params,
            },
            step=total_env_timesteps,
        )

        return manifest
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        finish_wandb_run(wandb_run)


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    print(json.dumps(run(args, ppo_config), indent=2))


if __name__ == "__main__":
    main()
