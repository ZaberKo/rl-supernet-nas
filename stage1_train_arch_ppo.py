from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from tqdm.auto import tqdm

from checkpoint_utils import (
    build_policy_from_checkpoint,
    load_checkpoint,
    load_critic_from_checkpoint,
    save_checkpoint,
)
from env_utils import EVAL_SEED_OFFSET, make_vec_env_from_ppo_config
from ppo_utils import (
    FixedPolicySubnet,
    PolicySupernet,
    append_jsonl_record,
    build_sb3_critic_model,
    collect_candidate_rollout,
    configure_actor_optimizer,
    critic_update,
    evaluate_actor_subnet,
    fixed_arch_actor_update,
    prefixed_metrics,
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
    resolve_device,
    set_global_seeds,
)
from supernet_backbone import ArchConfig, SearchSpace
from trajectory_data import DynamicsRolloutBuffer
from wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb,
    update_wandb_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 single-architecture PPO finetune diagnostic initialized from a policy supernet checkpoint.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument(
        "--arch_config",
        default="arch_configs/max_arch.json",
        help="JSON file containing one ArchConfig object.",
    )
    parser.add_argument(
        "--supernet_checkpoint",
        default="runs/stage1_policy_supernet/policy_supernet_best.pt",
        help="Stage 1 policy-supernet checkpoint used to initialize the actor supernet and critic.",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/stage1_arch_ppo",
        help="Directory for PPO metrics, checkpoint, and manifest.",
    )

    parser.add_argument(
        "--critic_warmup_timesteps",
        type=int,
        default=0,
        help="Critic-only warmup timesteps before actor PPO finetune; 0 disables warmup.",
    )

    parser.add_argument(
        "--suffix",
        default="max-arch",
        help="Optional suffix to append to the stage name.",
    )
    args = parser.parse_args()
    if args.critic_warmup_timesteps < 0:
        raise ValueError("critic_warmup_timesteps must be non-negative.")
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
        raise ValueError(
            "Architecture layer config stage count does not match the search space."
        )

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
            raise ValueError(
                f"Stage {stage_index} must contain {max_depth} layer configs."
            )
        for layer_index, layer_config in enumerate(stage_layers):
            if int(layer_config.kernel_size) not in search_space.kernel_size_candidates:
                raise ValueError(
                    f"Stage {stage_index} layer {layer_index} kernel size is not in the search space."
                )
            if (
                int(layer_config.expand_ratio)
                not in search_space.expand_ratio_candidates
            ):
                raise ValueError(
                    f"Stage {stage_index} layer {layer_index} expand ratio is not in the search space."
                )


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
    total_env_timesteps: int,
    phase: str,
) -> dict[str, Any]:
    eval_metrics = evaluate_actor_subnet(
        policy=policy,
        eval_env=eval_env,
        n_eval_episodes=int(n_eval_episodes),
        deterministic=bool(deterministic),
        device=device,
        train_env=train_env,
    )
    record = {
        "type": "eval",
        "phase": phase,
        "total_timesteps": int(total_timesteps),
        "total_env_timesteps": int(total_env_timesteps),
        "eval/ep_return": float(eval_metrics["ep_return"]),
        "eval/ep_return_std": float(eval_metrics["ep_return_std"]),
        "eval/ep_length": float(eval_metrics["ep_length"]),
        "eval/ep_length_std": float(eval_metrics["ep_length_std"]),
    }
    log_record(metrics_path, wandb_run, record, step=int(total_timesteps))
    return record


def _save_arch_checkpoint(
    path: Path,
    args: argparse.Namespace,
    ppo_config: DictConfig,
    policy: PolicySupernet | FixedPolicySubnet,
    critic_model: PPO,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    arch_config: ArchConfig,
    source_stage: str | None,
    total_timesteps: int,
    critic_warmup_timesteps: int,
    total_env_timesteps: int,
    stage_name: str,
    ema_policy: PolicySupernet | FixedPolicySubnet | None = None,
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
        source_stage=source_stage,
        arch_config=arch_config.to_dict(),
        supernet_checkpoint=args.supernet_checkpoint,
        total_timesteps=total_timesteps,
        critic_warmup_timesteps=critic_warmup_timesteps,
        total_env_timesteps=total_env_timesteps,
        **extra,
    )


def run(
    args: argparse.Namespace,
    ppo_config: DictConfig,
    *,
    arch_config: ArchConfig | None = None,
    device: torch.device | None = None,
    output_dir: str | Path | None = None,
    wandb_init_fn: Callable[..., Any] | None = None,
    extra_config: dict[str, Any] | None = None,
    progress_bar_desc: str | None = None,
) -> dict[str, Any]:
    """Run single-architecture PPO finetune.

    When called from the multi-worker script (``stage1_train_archs_ppo``),
    callers pass *arch_config*, *device*, *output_dir*, and *wandb_init_fn*
    to override the defaults derived from ``args``.
    """
    output_dir_resolved = Path(output_dir) if output_dir is not None else Path(args.output_dir)
    output_dir_resolved.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir_resolved / "metrics.jsonl"
    search_space_path = output_dir_resolved / "search_space.json"
    last_checkpoint_path = output_dir_resolved / "policy_supernet_arch_ppo_last.pt"
    best_checkpoint_path = output_dir_resolved / "policy_supernet_arch_ppo_best.pt"

    stage_name = (
        f"stage1_train_arch_ppo_{args.suffix}"
        if getattr(args, "suffix", "")
        else "stage1_train_arch_ppo"
    )
    run_config = build_run_config(args, ppo_config)
    if extra_config:
        run_config.update(extra_config)
    if wandb_init_fn is not None:
        wandb_run = wandb_init_fn(stage_name, run_config, output_dir_resolved)
    else:
        wandb_run = init_wandb_run(stage_name, run_config, output_dir_resolved)
    train_env = None
    eval_env = None
    progress_bar = None

    try:
        search_space = SearchSpace()
        search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))
        if arch_config is None:
            arch_config = load_arch_config(args.arch_config)
        validate_arch_config(search_space, arch_config)

        train_seed = int(ppo_config.seed)
        eval_seed = train_seed + EVAL_SEED_OFFSET
        set_global_seeds(train_seed)
        if device is None:
            device = resolve_device(str(ppo_config.device))

        checkpoint = load_checkpoint(args.supernet_checkpoint, map_location=device)

        train_env = make_vec_env_from_ppo_config(
            ppo_config, seed=train_seed, n_envs=ppo_config.train_n_envs
        )
        eval_episodes = ppo_config.eval_episodes
        eval_deterministic = ppo_config.eval_deterministic
        eval_freq = ppo_config.eval_freq
        if eval_episodes > 0:
            eval_env = make_vec_env_from_ppo_config(
                ppo_config, seed=eval_seed, n_envs=ppo_config.eval_n_envs
            )

        supernet = build_policy_from_checkpoint(
            ppo_config=ppo_config,
            env=train_env,
            search_space=search_space,
            checkpoint=checkpoint,
            device=device,
        )
        supernet.set_sample_config(arch_config)
        policy = supernet.get_active_subnet().to(device)

        z_dyn_coef = float(ppo_config.z_dyn_coef)
        ema_policy: FixedPolicySubnet | None = None
        if z_dyn_coef > 0.0:
            ema_state_dict = checkpoint.get("ema_policy_state_dict")
            if ema_state_dict is not None:
                supernet.load_state_dict(ema_state_dict, strict=True)
                supernet.set_sample_config(arch_config)
                ema_policy = supernet.get_active_subnet().to(device)
            else:
                ema_policy = copy.deepcopy(policy)
            for parameter in ema_policy.parameters():
                parameter.requires_grad_(False)
        del supernet

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
            device=device,
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

        config_record: dict[str, Any] = {
            "type": "config",
            "stage": stage_name,
            "supernet_checkpoint": str(args.supernet_checkpoint),
            "candidate_timesteps": int(target_timesteps),
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "device": str(device),
            "loaded_critic": bool(loaded_critic),
            "observation_shape": list(train_env.observation_space.shape),
            "action_space": type(train_env.action_space).__name__,
        }
        if hasattr(args, "arch_config"):
            config_record["arch_config_path"] = str(args.arch_config)
        if extra_config:
            config_record.update(extra_config)
        append_jsonl_record(metrics_path, config_record)

        initial_eval_record = None
        final_eval_record = None
        total_timesteps = 0
        critic_warmup_actual_timesteps = 0
        total_env_timesteps = 0
        best_eval_ep_return: float | None = None
        best_eval_record: dict[str, Any] | None = None

        def maybe_save_best_checkpoint(
            record: dict[str, Any],
            current_total_timesteps: int,
            current_total_env_timesteps: int,
        ) -> None:
            nonlocal best_eval_ep_return, best_eval_record
            ep_return = float(record["eval/ep_return"])
            if best_eval_ep_return is None or ep_return > best_eval_ep_return:
                best_eval_ep_return = ep_return
                best_eval_record = dict(record)
                _save_arch_checkpoint(
                    best_checkpoint_path,
                    args=args,
                    ppo_config=ppo_config,
                    policy=policy,
                    critic_model=critic_model,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    arch_config=arch_config,
                    source_stage=checkpoint.get("stage"),
                    total_timesteps=current_total_timesteps,
                    critic_warmup_timesteps=critic_warmup_actual_timesteps,
                    total_env_timesteps=current_total_env_timesteps,
                    stage_name=stage_name,
                    ema_policy=ema_policy,
                )

        # ----- Critic warmup (independent phase) -----
        observation = train_env.reset()
        episode_starts = np.ones((train_env.num_envs,), dtype=np.bool_)
        warmup_target = max(0, int(args.critic_warmup_timesteps))
        if warmup_target > 0:
            warmup_critic_lr = (
                float(critic_lr_schedule(1.0))
                if callable(critic_lr_schedule)
                else float(critic_lr_schedule)
            )
            update_optimizer_learning_rate(critic_optimizer, warmup_critic_lr)
            warmup_bar = tqdm(
                total=warmup_target,
                desc=(progress_bar_desc or "arch_ppo") + "_warmup",
                unit="step",
                dynamic_ncols=True,
                disable=ppo_config.quiet,
            )
            warmup_iteration = 0
            while critic_warmup_actual_timesteps < warmup_target:
                warmup_iteration += 1
                observation, episode_starts, rollout_metrics = collect_candidate_rollout(
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
                critic_warmup_actual_timesteps += int(ppo_config.n_steps) * int(
                    train_env.num_envs
                )
                critic_metrics = critic_update(
                    critic_model=critic_model,
                    optimizer=critic_optimizer,
                    rollout_buffer=rollout_buffer,
                    n_epochs=int(ppo_config.n_epochs),
                    batch_size=int(ppo_config.batch_size),
                    max_grad_norm=float(ppo_config.max_grad_norm),
                )
                # Log warmup to file only (not wandb)
                warmup_record = {
                    "type": "critic_warmup",
                    "iteration": int(warmup_iteration),
                    "critic_warmup_timesteps": int(critic_warmup_actual_timesteps),
                    "critic_lr": float(warmup_critic_lr),
                    **prefixed_metrics("critic_warmup_rollout", rollout_metrics),
                    **prefixed_metrics("critic_warmup", critic_metrics),
                }
                append_jsonl_record(metrics_path, warmup_record)
                warmup_bar.update(
                    min(critic_warmup_actual_timesteps, warmup_target) - warmup_bar.n
                )
                warmup_bar.set_postfix(
                    {
                        "ret": f"{rollout_metrics['rollout/ep_return']:.3g}",
                        "critic": f"{critic_metrics['critic/loss']:.3g}",
                        "lr": f"{warmup_critic_lr:.2g}",
                    },
                    refresh=True,
                )
            warmup_bar.close()


        # ----- Main training phase -----
        if target_timesteps > 0:
            progress_bar = tqdm(
                total=target_timesteps,
                desc=progress_bar_desc or "stage1_arch_ppo",
                unit="step",
                dynamic_ncols=True,
                disable=ppo_config.quiet,
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
                total_timesteps=0,
                total_env_timesteps=total_env_timesteps,
                phase="initial",
            )
            if progress_bar is not None:
                progress_bar.write(
                    "stage1_arch_eval phase=initial step=0 "
                    f"ep_return={initial_eval_record['eval/ep_return']:.6g} "
                    f"ep_return_std={initial_eval_record['eval/ep_return_std']:.6g} "
                    f"ep_length={initial_eval_record['eval/ep_length']:.6g}"
                )
            maybe_save_best_checkpoint(initial_eval_record, 0, 0)

        target_kl = ppo_config.target_kl
        next_eval_timestep = eval_freq if eval_freq > 0 else 0
        train_iteration = 0
        last_train_record: dict[str, Any] = {}
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

            observation, episode_starts, rollout_metrics = collect_candidate_rollout(
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
            total_env_timesteps = critic_warmup_actual_timesteps + total_timesteps
            actor_metrics = fixed_arch_actor_update(
                policy=policy,
                actor_optimizer=actor_optimizer,
                rollout_buffer=rollout_buffer,
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
            log_record(metrics_path, wandb_run, train_record, step=total_timesteps)
            if progress_bar is not None:
                progress_bar.update(total_timesteps - progress_bar.n)
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

            if (
                eval_env is not None
                and next_eval_timestep > 0
                and total_timesteps >= next_eval_timestep
            ):
                while next_eval_timestep <= total_timesteps:
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
                    total_timesteps=total_timesteps,
                    total_env_timesteps=total_env_timesteps,
                    phase="periodic",
                )
                if progress_bar is not None:
                    progress_bar.write(
                        f"stage1_arch_eval phase=periodic step={total_timesteps} "
                        f"ep_return={eval_record['eval/ep_return']:.6g} "
                        f"ep_return_std={eval_record['eval/ep_return_std']:.6g} "
                        f"ep_length={eval_record['eval/ep_length']:.6g}"
                    )
                maybe_save_best_checkpoint(
                    eval_record, total_timesteps, total_env_timesteps
                )

            _save_arch_checkpoint(
                last_checkpoint_path,
                args=args,
                ppo_config=ppo_config,
                policy=policy,
                critic_model=critic_model,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                arch_config=arch_config,
                source_stage=checkpoint.get("stage"),
                total_timesteps=total_timesteps,
                critic_warmup_timesteps=critic_warmup_actual_timesteps,
                total_env_timesteps=total_env_timesteps,
                stage_name=stage_name,
                ema_policy=ema_policy,
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
                total_timesteps=total_timesteps,
                total_env_timesteps=total_env_timesteps,
                phase="final",
            )
            if progress_bar is not None:
                progress_bar.write(
                    f"stage1_arch_eval phase=final step={total_timesteps} "
                    f"ep_return={final_eval_record['eval/ep_return']:.6g} "
                    f"ep_return_std={final_eval_record['eval/ep_return_std']:.6g} "
                    f"ep_length={final_eval_record['eval/ep_length']:.6g}"
                )
            maybe_save_best_checkpoint(
                final_eval_record, total_timesteps, total_env_timesteps
            )

        param_stats = policy.policy_param_stats()
        policy_backbone_params = param_stats["policy_backbone_params"]
        policy_head_params = param_stats["policy_head_params"]
        policy_params = param_stats["policy_params"]
        trainable_policy_params = param_stats["trainable_policy_params"]

        _save_arch_checkpoint(
            last_checkpoint_path,
            args=args,
            ppo_config=ppo_config,
            policy=policy,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            arch_config=arch_config,
            source_stage=checkpoint.get("stage"),
            total_timesteps=total_timesteps,
            critic_warmup_timesteps=critic_warmup_actual_timesteps,
            total_env_timesteps=total_env_timesteps,
            stage_name=stage_name,
            ema_policy=ema_policy,
        )

        manifest: dict[str, Any] = {
            "stage": stage_name,
            "source_stage": checkpoint.get("stage"),
            "arch_config": arch_config.to_dict(),
            "supernet_checkpoint": str(args.supernet_checkpoint),
            "last_checkpoint": str(last_checkpoint_path),
            "best_checkpoint": str(best_checkpoint_path),
            "metrics": str(metrics_path),
            "search_space": str(search_space_path),
            "configured_candidate_timesteps": int(target_timesteps),
            "critic_warmup_actual_timesteps": int(critic_warmup_actual_timesteps),
            "total_env_timesteps": int(total_env_timesteps),
            "policy_backbone_params": policy_backbone_params,
            "policy_head_params": policy_head_params,
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
        if hasattr(args, "arch_config"):
            manifest["arch_config_path"] = str(args.arch_config)
        if extra_config:
            manifest.update(extra_config)
        manifest_path = output_dir_resolved / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        update_wandb_summary(
            wandb_run,
            {
                "total_timesteps": int(total_timesteps),
                "critic_warmup_actual_timesteps": int(critic_warmup_actual_timesteps),
                "total_env_timesteps": int(total_env_timesteps),
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
