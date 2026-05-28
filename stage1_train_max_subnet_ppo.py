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

from checkpoint_utils import save_checkpoint
from env_utils import EVAL_SEED_OFFSET, make_vec_env_from_ppo_config

# ---------------------------------------------------------------------------
# Re-use the same collect_rollout from stage1_train_policy_supernet
# but inline it here so the script is self-contained.
# ---------------------------------------------------------------------------
from ppo_utils import (
    PolicySupernet,
    actor_head_parameters,
    append_jsonl_record,
    bootstrap_time_limit_rewards,
    build_sb3_critic_model,
    configure_actor_optimizer,
    count_parameters,
    create_ema_policy,
    critic_update,
    evaluate_actor_subnet,
    fixed_arch_actor_update,
    predict_critic_values,
    prefixed_metrics,
    prepare_env_actions,
    require_monitor_episode_info,
    update_actor_optimizer_learning_rate,
    update_ema_model,
    update_optimizer_learning_rate,
)
from setup_utils import (
    add_ppo_config_args,
    build_run_config,
    load_ppo_config,
    parse_schedule_value,
    ppo_config_to_dict,
    resolve_activation_fn,
    resolve_device,
    set_global_seeds,
)
from supernet_backbone import ArchConfig, SearchSpace
from trajectory_data import (
    DynamicsRolloutBuffer,
    resolve_terminal_next_observations,
    split_done_flags,
)
from wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb,
    update_wandb_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: train the max subnet (supernet) from scratch with PPO (no distillation).",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument(
        "--output_dir",
        default="runs/stage1_max_subnet_ppo",
        help="Directory for checkpoints, metrics, and manifest.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to the stage name (for W&B and manifests).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Rollout collection — identical to stage1_train_policy_supernet.py
# ---------------------------------------------------------------------------
def collect_rollout(
    policy: PolicySupernet,
    critic_model: PPO,
    env: VecEnv,
    rollout_buffer: DynamicsRolloutBuffer,
    initial_observation: np.ndarray,
    initial_episode_starts: np.ndarray,
    n_steps: int,
    gamma: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    policy.eval()
    critic_model.policy.eval()
    policy.set_max_arch()
    rollout_buffer.reset()

    num_envs = int(env.num_envs)
    observation = np.asarray(initial_observation)
    episode_starts = np.asarray(initial_episode_starts, dtype=np.bool_)
    rollout_reward_sum = 0.0
    rollout_done_count = 0
    rollout_episode_returns: list[float] = []
    rollout_episode_lengths: list[float] = []
    last_dones = episode_starts

    with torch.no_grad():
        for _ in range(n_steps):
            observation_tensor = torch.as_tensor(observation, device=device)
            action_tensor, log_prob_tensor, _ = policy.act(
                observation_tensor, deterministic=False
            )
            value_tensor = predict_critic_values(critic_model, observation_tensor)

            stored_actions, env_actions = prepare_env_actions(
                env.action_space, action_tensor, num_envs
            )

            next_observation, raw_rewards, raw_dones, infos = env.step(env_actions)
            info_list = list(infos)
            resolved_next_observation = resolve_terminal_next_observations(
                next_observation, info_list
            )
            terminated, _truncated = split_done_flags(raw_dones, info_list)
            dynamics_mask = (~terminated).astype(np.float32)
            adjusted_rewards = bootstrap_time_limit_rewards(
                rewards=np.asarray(raw_rewards, dtype=np.float32),
                dones=np.asarray(raw_dones, dtype=np.bool_),
                infos=info_list,
                critic_model=critic_model,
                device=device,
                gamma=gamma,
            )

            rollout_buffer.add_transition(
                obs=observation,
                next_obs=resolved_next_observation,
                action=stored_actions,
                reward=adjusted_rewards,
                episode_start=episode_starts,
                value=value_tensor,
                log_prob=log_prob_tensor,
                dynamics_mask=dynamics_mask,
            )

            rollout_reward_sum += float(np.asarray(raw_rewards, dtype=np.float32).sum())
            done_array = np.asarray(raw_dones, dtype=np.bool_)
            rollout_done_count += int(done_array.sum())
            for env_index, done in enumerate(done_array):
                if not bool(done):
                    continue
                episode_info = require_monitor_episode_info(
                    info_list[env_index], "training rollout"
                )
                rollout_episode_returns.append(float(episode_info["r"]))
                rollout_episode_lengths.append(float(episode_info["l"]))
            observation = np.asarray(next_observation)
            episode_starts = done_array
            last_dones = episode_starts

        last_observation_tensor = torch.as_tensor(observation, device=device)
        last_values = predict_critic_values(critic_model, last_observation_tensor)

    rollout_buffer.compute_returns_and_advantage(
        last_values=last_values, dones=last_dones
    )
    ep_return = (
        float(np.mean(rollout_episode_returns)) if rollout_episode_returns else 0.0
    )
    ep_length = (
        float(np.mean(rollout_episode_lengths)) if rollout_episode_lengths else 0.0
    )
    metrics = {
        "rollout/reward_per_step": rollout_reward_sum
        / float(max(1, n_steps * num_envs)),
        "rollout/ep_return": ep_return,
        "rollout/ep_length": ep_length,
        "rollout/done_count": float(rollout_done_count),
        "rollout/advantage_mean": float(rollout_buffer.advantages.mean()),
        "rollout/advantage_std": float(rollout_buffer.advantages.std()),
        "rollout/dynamics_mask_mean": float(rollout_buffer.dynamics_masks.mean()),
    }
    return observation, episode_starts, metrics


# ---------------------------------------------------------------------------
# Utility helpers (same as arch_ppo)
# ---------------------------------------------------------------------------
def numeric_values(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if isinstance(value, (int, float, bool))
    }


def log_record(
    metrics_path: Path, wandb_run: Any, record: Mapping[str, Any], step: int
) -> None:
    append_jsonl_record(metrics_path, record)
    log_wandb(wandb_run, numeric_values(record), step=step)


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
    total_timesteps: int,
    phase: str,
) -> dict[str, Any]:
    eval_metrics = evaluate_actor_subnet(
        policy=policy,
        eval_env=eval_env,
        arch=arch_config,
        n_eval_episodes=int(n_eval_episodes),
        deterministic=bool(deterministic),
        device=device,
        train_env=train_env,
    )
    record = {
        "type": "eval",
        "phase": phase,
        "total_timesteps": int(total_timesteps),
        "eval/ep_return": float(eval_metrics["ep_return"]),
        "eval/ep_return_std": float(eval_metrics["ep_return_std"]),
        "eval/ep_length": float(eval_metrics["ep_length"]),
        "eval/ep_length_std": float(eval_metrics["ep_length_std"]),
    }
    log_record(metrics_path, wandb_run, record, step=int(total_timesteps))
    return record


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------
def _save_max_subnet_checkpoint(
    path: Path,
    args: argparse.Namespace,
    ppo_config: DictConfig,
    policy: PolicySupernet,
    ema_policy: PolicySupernet | None,
    critic_model: PPO,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    total_timesteps: int,
    iteration: int,
    stage_name: str,
) -> None:
    extra: dict[str, Any] = {}
    if ema_policy is not None:
        extra["ema_policy_state_dict"] = ema_policy.state_dict()
    save_checkpoint(
        path,
        stage=stage_name,
        args=args,
        ppo_config=ppo_config,
        policy_state_dict=policy.state_dict(),
        critic_policy_state_dict=critic_model.policy.state_dict(),
        actor_optimizer_state_dict=actor_optimizer.state_dict(),
        critic_optimizer_state_dict=critic_optimizer.state_dict(),
        total_timesteps=total_timesteps,
        iteration=iteration,
        **extra,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace, ppo_config: DictConfig) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    search_space_path = output_dir / "search_space.json"
    last_checkpoint_path = output_dir / "max_subnet_ppo_last.pt"
    best_checkpoint_path = output_dir / "max_subnet_ppo_best.pt"

    stage_name = (
        f"stage1_train_max_subnet_ppo_{args.suffix}"
        if getattr(args, "suffix", "")
        else "stage1_train_max_subnet_ppo"
    )
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run(stage_name, run_config, output_dir)
    train_env = None
    eval_env = None
    progress_bar = None

    try:
        search_space = SearchSpace()
        search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))
        max_arch = search_space.max_arch()

        train_seed = int(ppo_config.seed)
        eval_seed = train_seed + EVAL_SEED_OFFSET
        set_global_seeds(train_seed)
        device = resolve_device(str(ppo_config.device))

        train_env = make_vec_env_from_ppo_config(
            ppo_config, seed=train_seed, n_envs=ppo_config.train_n_envs
        )
        eval_episodes = int(ppo_config.eval_episodes)
        eval_deterministic = bool(ppo_config.eval_deterministic)
        eval_freq = int(ppo_config.eval_freq)
        if eval_freq <= 0 or eval_episodes <= 0:
            raise ValueError(
                "stage1_max_subnet_ppo requires positive ppo.eval_freq and ppo.eval_episodes."
            )
        eval_env = make_vec_env_from_ppo_config(
            ppo_config, seed=eval_seed, n_envs=ppo_config.eval_n_envs
        )

        # --- Build policy from scratch (no checkpoint) ---
        policy = PolicySupernet(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            search_space=search_space,
            features_dim=ppo_config.features_dim,
            policy_net_arch=list(ppo_config.policy_net_arch),
            activation_fn=resolve_activation_fn(ppo_config.activation_fn),
            log_std_init=ppo_config.log_std_init,
            ortho_init=ppo_config.ortho_init,
            projection_dim=ppo_config.projection_dim,
            predictor_hidden_dim=ppo_config.predictor_hidden_dim,
        ).to(device)
        policy.set_active_arch(max_arch)

        z_dyn_coef = float(ppo_config.z_dyn_coef)
        ema_policy: PolicySupernet | None = None
        if z_dyn_coef > 0.0:
            ema_policy = create_ema_policy(policy, device)

        critic_lr_schedule = parse_schedule_value(ppo_config.critic_lr)
        policy_head_lr_schedule = parse_schedule_value(ppo_config.policy_head_lr)
        policy_backbone_lr_schedule = parse_schedule_value(
            ppo_config.policy_backbone_lr
        )
        clip_range_schedule = parse_schedule_value(ppo_config.clip_range)
        target_timesteps = max(0, int(ppo_config.total_timesteps))

        actor_optimizer = configure_actor_optimizer(
            policy=policy,
            head_lr=policy_head_lr_schedule,
            backbone_lr=policy_backbone_lr_schedule,
        )
        critic_model = build_sb3_critic_model(
            ppo_config=ppo_config,
            env=train_env,
            learning_rate=critic_lr_schedule,
        )
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

        # --- Config log ---
        append_jsonl_record(
            metrics_path,
            {
                "type": "config",
                "stage": stage_name,
                "candidate_timesteps": int(target_timesteps),
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "device": str(device),
                "observation_shape": list(train_env.observation_space.shape),
                "action_space": type(train_env.action_space).__name__,
                "arch_config": max_arch.to_dict(),
                "train_from_scratch": True,
            },
        )

        progress_bar = tqdm(
            total=target_timesteps,
            desc="stage1_max_subnet_ppo",
            unit="step",
            dynamic_ncols=True,
            disable=ppo_config.quiet,
        )

        initial_eval_record = None
        final_eval_record = None
        total_timesteps = 0
        best_eval_ep_return: float | None = None
        best_eval_record: dict[str, Any] | None = None
        target_kl = ppo_config.target_kl
        next_eval_timestep = min(eval_freq, target_timesteps)
        train_iteration = 0
        last_train_record: dict[str, Any] = {}

        # --- Closure: save best checkpoint on eval improvement ---
        def maybe_save_best_checkpoint(record: dict[str, Any]) -> None:
            nonlocal best_eval_ep_return, best_eval_record
            ep_return = float(record["eval/ep_return"])
            if best_eval_ep_return is None or ep_return > best_eval_ep_return:
                best_eval_ep_return = ep_return
                best_eval_record = dict(record)
                _save_max_subnet_checkpoint(
                    best_checkpoint_path,
                    args=args,
                    ppo_config=ppo_config,
                    policy=policy,
                    ema_policy=ema_policy,
                    critic_model=critic_model,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    total_timesteps=total_timesteps,
                    iteration=train_iteration,
                    stage_name=stage_name,
                )

        # --- Initial evaluation ---
        initial_eval_record = evaluate_and_record(
            policy=policy,
            train_env=train_env,
            eval_env=eval_env,
            arch_config=max_arch,
            n_eval_episodes=eval_episodes,
            deterministic=eval_deterministic,
            device=device,
            metrics_path=metrics_path,
            wandb_run=wandb_run,
            total_timesteps=0,
            phase="initial",
        )
        progress_bar.write(
            "stage1_max_subnet_eval phase=initial step=0 "
            f"ep_return={initial_eval_record['eval/ep_return']:.6g} "
            f"ep_return_std={initial_eval_record['eval/ep_return_std']:.6g} "
            f"ep_length={initial_eval_record['eval/ep_length']:.6g}"
        )
        maybe_save_best_checkpoint(initial_eval_record)

        # --- Training loop ---
        observation = train_env.reset()
        episode_starts = np.ones((train_env.num_envs,), dtype=np.bool_)

        while total_timesteps < target_timesteps:
            train_iteration += 1
            progress_remaining = 1.0 - float(total_timesteps) / float(
                max(1, target_timesteps)
            )
            actor_lr = (
                float(policy_head_lr_schedule(progress_remaining))
                if callable(policy_head_lr_schedule)
                else float(policy_head_lr_schedule)
            )
            backbone_lr = (
                float(policy_backbone_lr_schedule(progress_remaining))
                if callable(policy_backbone_lr_schedule)
                else float(policy_backbone_lr_schedule)
            )
            critic_lr = (
                float(critic_lr_schedule(progress_remaining))
                if callable(critic_lr_schedule)
                else float(critic_lr_schedule)
            )
            clip_range = (
                float(clip_range_schedule(progress_remaining))
                if callable(clip_range_schedule)
                else float(clip_range_schedule)
            )
            update_actor_optimizer_learning_rate(actor_optimizer, actor_lr, backbone_lr)
            update_optimizer_learning_rate(critic_optimizer, critic_lr)

            policy.set_max_arch()
            observation, episode_starts, rollout_metrics = collect_rollout(
                policy=policy,
                critic_model=critic_model,
                env=train_env,
                rollout_buffer=rollout_buffer,
                initial_observation=observation,
                initial_episode_starts=episode_starts,
                n_steps=int(ppo_config.n_steps),
                gamma=float(ppo_config.gamma),
                device=device,
            )
            total_timesteps += int(ppo_config.n_steps) * int(train_env.num_envs)

            actor_metrics = fixed_arch_actor_update(
                policy=policy,
                actor_optimizer=actor_optimizer,
                rollout_buffer=rollout_buffer,
                arch=max_arch,
                action_space=train_env.action_space,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                clip_range=clip_range,
                normalize_advantage=bool(ppo_config.normalize_advantage),
                ent_coef=float(ppo_config.ent_coef),
                max_grad_norm=float(ppo_config.max_grad_norm),
                target_kl=target_kl,
                ema_policy=ema_policy,
                z_dyn_coef=z_dyn_coef,
            )
            if ema_policy is not None and z_dyn_coef > 0.0:
                update_ema_model(ema_policy, policy, tau=float(ppo_config.ema_tau))
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
                "total_timesteps": int(total_timesteps),
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
            log_record(metrics_path, wandb_run, train_record, step=total_timesteps)

            progress_bar.update(
                max(0, min(total_timesteps, target_timesteps) - progress_bar.n)
            )
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

            # --- Periodic evaluation ---
            if total_timesteps >= next_eval_timestep:
                while next_eval_timestep <= total_timesteps:
                    next_eval_timestep += eval_freq
                eval_record = evaluate_and_record(
                    policy=policy,
                    train_env=train_env,
                    eval_env=eval_env,
                    arch_config=max_arch,
                    n_eval_episodes=eval_episodes,
                    deterministic=eval_deterministic,
                    device=device,
                    metrics_path=metrics_path,
                    wandb_run=wandb_run,
                    total_timesteps=total_timesteps,
                    phase="periodic",
                )
                progress_bar.write(
                    f"stage1_max_subnet_eval phase=periodic step={total_timesteps} "
                    f"ep_return={eval_record['eval/ep_return']:.6g} "
                    f"ep_return_std={eval_record['eval/ep_return_std']:.6g} "
                    f"ep_length={eval_record['eval/ep_length']:.6g}"
                )
                maybe_save_best_checkpoint(eval_record)

            # --- Save last checkpoint every iteration ---
            _save_max_subnet_checkpoint(
                last_checkpoint_path,
                args=args,
                ppo_config=ppo_config,
                policy=policy,
                ema_policy=ema_policy,
                critic_model=critic_model,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                total_timesteps=total_timesteps,
                iteration=train_iteration,
                stage_name=stage_name,
            )

        # --- Final evaluation ---
        final_eval_record = evaluate_and_record(
            policy=policy,
            train_env=train_env,
            eval_env=eval_env,
            arch_config=max_arch,
            n_eval_episodes=eval_episodes,
            deterministic=eval_deterministic,
            device=device,
            metrics_path=metrics_path,
            wandb_run=wandb_run,
            total_timesteps=total_timesteps,
            phase="final",
        )
        progress_bar.write(
            f"stage1_max_subnet_eval phase=final step={total_timesteps} "
            f"ep_return={final_eval_record['eval/ep_return']:.6g} "
            f"ep_return_std={final_eval_record['eval/ep_return_std']:.6g} "
            f"ep_length={final_eval_record['eval/ep_length']:.6g}"
        )
        maybe_save_best_checkpoint(final_eval_record)

        # --- Parameter stats ---
        policy.set_active_arch(max_arch)
        policy_backbone_params = int(policy.backbone.elastic_num_params)
        policy_head_params = count_parameters(actor_head_parameters(policy))
        policy_params = int(policy.elastic_num_params)
        trainable_policy_params = policy_head_params
        if any(p.requires_grad for p in policy.backbone.parameters()):
            trainable_policy_params += policy_backbone_params

        # --- Final last checkpoint ---
        _save_max_subnet_checkpoint(
            last_checkpoint_path,
            args=args,
            ppo_config=ppo_config,
            policy=policy,
            ema_policy=ema_policy,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            total_timesteps=total_timesteps,
            iteration=train_iteration,
            stage_name=stage_name,
        )

        # --- Manifest ---
        manifest = {
            "stage": stage_name,
            "arch_config": max_arch.to_dict(),
            "last_checkpoint": str(last_checkpoint_path),
            "best_checkpoint": str(best_checkpoint_path),
            "metrics": str(metrics_path),
            "search_space": str(search_space_path),
            "configured_total_timesteps": int(target_timesteps),
            "total_timesteps": int(total_timesteps),
            "policy_backbone_params": policy_backbone_params,
            "policy_head_params": policy_head_params,
            "policy_params": policy_params,
            "trainable_policy_params": trainable_policy_params,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "initial_eval": initial_eval_record,
            "final_eval": final_eval_record,
            "best_eval_ep_return": best_eval_ep_return,
            "best_eval": best_eval_record,
            "last_train": last_train_record,
            "notes": {
                "description": "Max subnet (supernet) trained from scratch with PPO, no distillation.",
                "rollout_policy": "max_subnet_only",
                "actor_update": "fixed_arch_ppo",
                "train_from_scratch": True,
            },
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        update_wandb_summary(
            wandb_run,
            {
                "total_timesteps": int(total_timesteps),
                "policy_backbone_params": policy_backbone_params,
                "policy_head_params": policy_head_params,
                "policy_params": policy_params,
                "trainable_policy_params": trainable_policy_params,
                "best_eval_ep_return": best_eval_ep_return,
            },
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
