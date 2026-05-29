from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3.common.vec_env import VecEnv
from tqdm.auto import tqdm

from checkpoint_utils import (
    build_policy_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from env_utils import EVAL_SEED_OFFSET, make_vec_env_from_ppo_config
from ppo_utils import (
    PolicySupernet,
    SB3CriticModel,
    append_jsonl_record,
    build_sb3_critic_model,
    configure_actor_optimizer,
    create_ema_policy,
    evaluate_actor_subnet,
    prepare_env_actions,
    require_monitor_episode_info,
    update_actor_optimizer_learning_rate,
    update_ema_model,
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
from stage1_train_policy_supernet import policy_kl_distillation_loss
from supernet_backbone import SearchSpace
from trajectory_data import (
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
        description=(
            "Stage 2: pretrain a fresh policy supernet by distilling a frozen "
            "teacher policy into max, min, and random sandwich subnets."
        ),
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument(
        "--teacher_checkpoint",
        required=True,
        help="Checkpoint from stage1_train_max_subnet_ppo.py used as the teacher.",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/stage2_policy_supernet_distill_pretrain",
        help="Directory for checkpoints, metrics, and manifest.",
    )
    parser.add_argument(
        "--random_subnets",
        type=int,
        default=2,
        help="Number of random subnets distilled per minibatch, in addition to max and min.",
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=1.0,
        help="Temperature for discrete policy KL distillation.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to the stage name for W&B and manifests.",
    )
    return parser.parse_args()


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


@dataclass
class TeacherRolloutData:
    """Collected teacher rollout data for distillation."""

    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    dynamics_masks: torch.Tensor


def collect_teacher_rollout(
    teacher_policy: PolicySupernet,
    env: VecEnv,
    initial_observation: np.ndarray,
    n_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, TeacherRolloutData, dict[str, float]]:
    teacher_policy.eval()
    teacher_policy.set_max_arch()

    num_envs = int(env.num_envs)
    observation = np.asarray(initial_observation)
    observation_batches: list[np.ndarray] = []
    next_observation_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    dynamics_mask_batches: list[np.ndarray] = []
    rollout_reward_sum = 0.0
    rollout_done_count = 0
    rollout_episode_returns: list[float] = []
    rollout_episode_lengths: list[float] = []

    with torch.no_grad():
        for _ in range(int(n_steps)):
            observation_batches.append(observation.copy())
            observation_tensor = torch.as_tensor(observation, device=device)
            action_tensor, _log_prob_tensor, _entropy_tensor = teacher_policy.act(
                observation_tensor, deterministic=False
            )
            stored_actions, env_actions = prepare_env_actions(
                env.action_space, action_tensor, num_envs
            )

            next_observation, raw_rewards, raw_dones, infos = env.step(env_actions)
            info_list = list(infos)
            reward_array = np.asarray(raw_rewards, dtype=np.float64)
            done_array = np.asarray(raw_dones, dtype=np.bool_)

            resolved_next_obs = resolve_terminal_next_observations(
                next_observation, info_list
            )
            next_observation_batches.append(resolved_next_obs)
            action_batches.append(stored_actions)
            terminated, _truncated = split_done_flags(raw_dones, info_list)
            dynamics_mask_batches.append((~terminated).astype(np.float32))

            rollout_reward_sum += float(reward_array.sum())
            rollout_done_count += int(done_array.sum())
            for env_index, done in enumerate(done_array):
                if not bool(done):
                    continue
                episode_info = require_monitor_episode_info(
                    info_list[env_index], "teacher rollout"
                )
                rollout_episode_returns.append(float(episode_info["r"]))
                rollout_episode_lengths.append(float(episode_info["l"]))

            observation = np.asarray(next_observation)

    rollout_observations = np.concatenate(observation_batches, axis=0)
    rollout_observation_tensor = torch.as_tensor(rollout_observations, device=device)

    rollout_next_observations = np.concatenate(next_observation_batches, axis=0)
    rollout_next_obs_tensor = torch.as_tensor(
        rollout_next_observations, device=device
    )
    rollout_actions = np.concatenate(action_batches, axis=0)
    rollout_actions_tensor = torch.as_tensor(rollout_actions, device=device)
    rollout_dynamics_masks = np.concatenate(dynamics_mask_batches, axis=0)
    rollout_dynamics_masks_tensor = torch.as_tensor(
        rollout_dynamics_masks, device=device
    )

    rollout_data = TeacherRolloutData(
        observations=rollout_observation_tensor,
        next_observations=rollout_next_obs_tensor,
        actions=rollout_actions_tensor,
        dynamics_masks=rollout_dynamics_masks_tensor,
    )

    ep_return = (
        float(np.mean(rollout_episode_returns)) if rollout_episode_returns else 0.0
    )
    ep_length = (
        float(np.mean(rollout_episode_lengths)) if rollout_episode_lengths else 0.0
    )
    metrics = {
        "rollout/reward_per_step": rollout_reward_sum
        / float(max(1, int(n_steps) * num_envs)),
        "rollout/ep_return": ep_return,
        "rollout/ep_length": ep_length,
        "rollout/done_count": float(rollout_done_count),
        "rollout/dynamics_mask_mean": float(rollout_dynamics_masks_tensor.mean()),
    }
    return observation, rollout_data, metrics


def sandwich_distill_update(
    student_policy: PolicySupernet,
    teacher_policy: PolicySupernet,
    actor_optimizer: torch.optim.Optimizer,
    rollout_data: TeacherRolloutData,
    search_space: SearchSpace,
    action_space,
    n_epochs: int,
    batch_size: int,
    max_grad_norm: float,
    random_subnets: int,
    temperature: float,
    ema_policy: PolicySupernet | None = None,
    z_dyn_coef: float = 0.0,
) -> dict[str, float]:
    student_policy.train()
    teacher_policy.eval()
    use_dynamics = ema_policy is not None and z_dyn_coef > 0.0
    if use_dynamics:
        ema_policy.eval()

    observations = rollout_data.observations
    max_arch = search_space.max_arch()
    min_arch = search_space.min_arch()
    num_samples = int(observations.size(0))
    update_count = 0
    loss_sum = 0.0
    max_loss_sum = 0.0
    min_loss_sum = 0.0
    random_loss_sum = 0.0
    random_loss_count = 0
    dynamics_loss_sum = 0.0
    arch_count_sum = 0.0

    for _ in range(int(n_epochs)):
        permutation = torch.randperm(num_samples, device=observations.device)
        for start_index in range(0, num_samples, int(batch_size)):
            batch_indices = permutation[start_index : start_index + int(batch_size)]
            batch_observations = observations.index_select(0, batch_indices)

            with torch.no_grad():
                teacher_policy.set_sample_config(max_arch)
                teacher_params = teacher_policy.distribution_params(batch_observations)
                teacher_params = {
                    key: value.detach() for key, value in teacher_params.items()
                }

            if use_dynamics:
                batch_next_observations = rollout_data.next_observations.index_select(
                    0, batch_indices
                )
                batch_actions = rollout_data.actions.index_select(0, batch_indices)
                batch_dynamics_masks = rollout_data.dynamics_masks.index_select(
                    0, batch_indices
                )

            sampled_arches = [("max", max_arch), ("min", min_arch)]
            for _sample_index in range(int(random_subnets)):
                sampled_arches.append(("random", search_space.sample_arch()))

            actor_optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=observations.device)
            batch_max_loss = 0.0
            batch_min_loss = 0.0
            batch_random_loss_sum = 0.0
            batch_random_loss_count = 0
            batch_dynamics_loss_sum = 0.0

            for arch_name, arch in sampled_arches:
                student_policy.set_sample_config(arch)
                student_features = student_policy.encode(batch_observations)
                student_params = student_policy.distribution_params_from_features(
                    student_features
                )
                distill_loss = policy_kl_distillation_loss(
                    action_space=action_space,
                    teacher_params=teacher_params,
                    student_params=student_params,
                    temperature=temperature,
                )
                subnet_loss = distill_loss

                if use_dynamics:
                    ema_policy.set_sample_config(arch)
                    dyn_loss = student_policy.compute_dynamics_loss(
                        ema_policy=ema_policy,
                        start_features=student_features,
                        next_observations=batch_next_observations,
                        actions=batch_actions,
                        sample_weights=batch_dynamics_masks,
                    )
                    subnet_loss = subnet_loss + float(z_dyn_coef) * dyn_loss
                    batch_dynamics_loss_sum += float(dyn_loss.detach().cpu())

                batch_loss = batch_loss + subnet_loss / float(len(sampled_arches))
                loss_value = float(distill_loss.detach().cpu())
                if arch_name == "max":
                    batch_max_loss = loss_value
                elif arch_name == "min":
                    batch_min_loss = loss_value
                else:
                    batch_random_loss_sum += loss_value
                    batch_random_loss_count += 1

            batch_loss.backward()
            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in student_policy.parameters()
                        if parameter.requires_grad
                    ],
                    float(max_grad_norm),
                )
            actor_optimizer.step()

            update_count += 1
            loss_sum += float(batch_loss.detach().cpu())
            max_loss_sum += batch_max_loss
            min_loss_sum += batch_min_loss
            random_loss_sum += batch_random_loss_sum
            random_loss_count += batch_random_loss_count
            dynamics_loss_sum += batch_dynamics_loss_sum / float(len(sampled_arches))
            arch_count_sum += float(len(sampled_arches))

    denominator = float(max(1, update_count))
    random_denominator = float(max(1, random_loss_count))
    result = {
        "actor/loss": loss_sum / denominator,
        "actor/policy_distill_loss": loss_sum / denominator,
        "actor/max_policy_distill_loss": max_loss_sum / denominator,
        "actor/min_policy_distill_loss": min_loss_sum / denominator,
        "actor/random_policy_distill_loss": random_loss_sum / random_denominator,
        "actor/update_count": float(update_count),
        "actor/sampled_arch_count": arch_count_sum / denominator,
    }
    if use_dynamics:
        result["actor/dynamics_loss"] = dynamics_loss_sum / denominator
    return result


def _save_distill_checkpoint(
    path: Path,
    args: argparse.Namespace,
    ppo_config: DictConfig,
    policy: PolicySupernet,
    critic_model: SB3CriticModel,
    critic_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    total_timesteps: int,
    iteration: int,
    stage_name: str,
    ema_policy: PolicySupernet | None = None,
) -> None:
    extra_kwargs: dict[str, Any] = {}
    if ema_policy is not None:
        extra_kwargs["ema_policy_state_dict"] = ema_policy.state_dict()
    save_checkpoint(
        path,
        stage=stage_name,
        args=args,
        ppo_config=ppo_config,
        policy_state_dict=policy.state_dict(),
        critic_policy_state_dict=critic_model.state_dict(),
        actor_optimizer_state_dict=actor_optimizer.state_dict(),
        critic_optimizer_state_dict=critic_optimizer.state_dict(),
        total_timesteps=total_timesteps,
        iteration=iteration,
        teacher_checkpoint=str(args.teacher_checkpoint),
        **extra_kwargs,
    )


def evaluate_and_log_sandwich(
    policy: PolicySupernet,
    train_env: VecEnv,
    eval_env: VecEnv,
    search_space: SearchSpace,
    ppo_config: DictConfig,
    device: torch.device,
    metrics_path: Path,
    wandb_run: Any,
    total_timesteps: int,
    iteration: int,
    phase: str,
) -> dict[str, Any]:
    policy.set_sample_config(search_space.max_arch())
    max_subnet_eval = evaluate_actor_subnet(
        policy=policy,
        eval_env=eval_env,
        n_eval_episodes=int(ppo_config.eval_episodes),
        deterministic=bool(ppo_config.eval_deterministic),
        device=device,
        train_env=train_env,
    )
    policy.set_sample_config(search_space.min_arch())
    min_subnet_eval = evaluate_actor_subnet(
        policy=policy,
        eval_env=eval_env,
        n_eval_episodes=int(ppo_config.eval_episodes),
        deterministic=bool(ppo_config.eval_deterministic),
        device=device,
        train_env=train_env,
    )
    record = {
        "type": "eval",
        "phase": phase,
        "iteration": int(iteration),
        "total_timesteps": int(total_timesteps),
        "eval/max_subnet_ep_return": float(max_subnet_eval["ep_return"]),
        "eval/max_subnet_ep_return_std": float(max_subnet_eval["ep_return_std"]),
        "eval/max_subnet_ep_length": float(max_subnet_eval["ep_length"]),
        "eval/max_subnet_ep_length_std": float(max_subnet_eval["ep_length_std"]),
        "eval/min_subnet_ep_return": float(min_subnet_eval["ep_return"]),
        "eval/min_subnet_ep_return_std": float(min_subnet_eval["ep_return_std"]),
        "eval/min_subnet_ep_length": float(min_subnet_eval["ep_length"]),
        "eval/min_subnet_ep_length_std": float(min_subnet_eval["ep_length_std"]),
    }
    log_record(metrics_path, wandb_run, record, step=int(total_timesteps))
    return record


def run(args: argparse.Namespace, ppo_config: DictConfig) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    search_space_path = output_dir / "search_space.json"
    last_checkpoint_path = output_dir / "policy_supernet_distill_last.pt"
    best_checkpoint_path = output_dir / "policy_supernet_distill_best.pt"

    stage_name = (
        f"stage2_pretrain_policy_supernet_distill_{args.suffix}"
        if getattr(args, "suffix", "")
        else "stage2_pretrain_policy_supernet_distill"
    )
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run(stage_name, run_config, output_dir)
    train_env = None
    eval_env = None
    progress_bar = None

    try:
        set_global_seeds(int(ppo_config.seed))
        device = resolve_device(str(ppo_config.device))
        search_space = SearchSpace()
        search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))

        train_env = make_vec_env_from_ppo_config(
            ppo_config, seed=int(ppo_config.seed), n_envs=int(ppo_config.train_n_envs)
        )
        eval_env = make_vec_env_from_ppo_config(
            ppo_config,
            seed=int(ppo_config.seed) + EVAL_SEED_OFFSET,
            n_envs=int(ppo_config.eval_n_envs),
        )
        if int(ppo_config.eval_freq) <= 0 or int(ppo_config.eval_episodes) <= 0:
            raise ValueError(
                "stage2 distill pretraining requires positive ppo.eval_freq and ppo.eval_episodes."
            )

        teacher_checkpoint = load_checkpoint(args.teacher_checkpoint, map_location=device)
        teacher_policy = build_policy_from_checkpoint(
            ppo_config=ppo_config,
            env=train_env,
            search_space=search_space,
            checkpoint=teacher_checkpoint,
            device=device,
        )
        teacher_policy.set_max_arch()
        teacher_policy.eval()
        for parameter in teacher_policy.parameters():
            parameter.requires_grad_(False)

        policy = PolicySupernet(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            search_space=search_space,
            features_dim=int(ppo_config.features_dim),
            policy_net_arch=list(ppo_config.policy_net_arch),
            activation_fn=resolve_activation_fn(ppo_config.activation_fn),
            log_std_init=ppo_config.log_std_init,
            ortho_init=bool(ppo_config.ortho_init),
            projection_dim=int(ppo_config.projection_dim),
            predictor_hidden_dim=int(ppo_config.predictor_hidden_dim),
        ).to(device)
        policy.load_state_dict(teacher_policy.state_dict())

        z_dyn_coef = float(ppo_config.z_dyn_coef)
        ema_policy: PolicySupernet | None = None
        if z_dyn_coef > 0.0:
            ema_policy = create_ema_policy(policy, device)

        policy_head_lr_schedule = parse_schedule_value(ppo_config.policy_head_lr)
        policy_backbone_lr_schedule = parse_schedule_value(
            ppo_config.policy_backbone_lr
        )
        critic_lr_schedule = parse_schedule_value(ppo_config.critic_lr)
        actor_optimizer = configure_actor_optimizer(
            policy=policy,
            head_lr=policy_head_lr_schedule,
            backbone_lr=policy_backbone_lr_schedule,
        )
        critic_model = build_sb3_critic_model(
            ppo_config=ppo_config,
            env=train_env,
        )
        critic_lr = (
            float(critic_lr_schedule(1.0))
            if callable(critic_lr_schedule)
            else float(critic_lr_schedule)
        )
        critic_optimizer = torch.optim.Adam(
            critic_model.parameters(), lr=critic_lr, eps=1e-5,
        )

        target_timesteps = max(0, int(ppo_config.total_timesteps))
        eval_freq = int(ppo_config.eval_freq)
        next_eval_timestep = min(eval_freq, target_timesteps)
        total_timesteps = 0
        iteration = 0
        best_eval_max_subnet_ep_return: float | None = None
        best_eval_record: dict[str, Any] | None = None
        final_eval_record: dict[str, Any] | None = None
        last_train_record: dict[str, Any] | None = None

        append_jsonl_record(
            metrics_path,
            {
                "type": "config",
                "stage": stage_name,
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "configured_total_timesteps": int(target_timesteps),
                "device": str(device),
                "action_space": type(train_env.action_space).__name__,
                "observation_shape": list(train_env.observation_space.shape),
                "random_subnets": int(args.random_subnets),
                "distill_temperature": float(args.distill_temperature),
                "z_dyn_coef": z_dyn_coef,
            },
        )

        progress_bar = tqdm(
            total=target_timesteps,
            desc="stage2_policy_distill",
            unit="step",
            dynamic_ncols=True,
            disable=bool(ppo_config.quiet),
        )

        def maybe_save_best_checkpoint(record: dict[str, Any]) -> None:
            nonlocal best_eval_max_subnet_ep_return, best_eval_record
            ep_return = float(record["eval/max_subnet_ep_return"])
            if (
                best_eval_max_subnet_ep_return is None
                or ep_return > best_eval_max_subnet_ep_return
            ):
                best_eval_max_subnet_ep_return = ep_return
                best_eval_record = dict(record)
                _save_distill_checkpoint(
                    best_checkpoint_path,
                    args=args,
                    ppo_config=ppo_config,
                    policy=policy,
                    critic_model=critic_model,
                    critic_optimizer=critic_optimizer,
                    actor_optimizer=actor_optimizer,
                    total_timesteps=total_timesteps,
                    iteration=iteration,
                    stage_name=stage_name,
                    ema_policy=ema_policy,
                )

        initial_eval_record = evaluate_and_log_sandwich(
            policy=policy,
            train_env=train_env,
            eval_env=eval_env,
            search_space=search_space,
            ppo_config=ppo_config,
            device=device,
            metrics_path=metrics_path,
            wandb_run=wandb_run,
            total_timesteps=0,
            iteration=0,
            phase="initial",
        )
        maybe_save_best_checkpoint(initial_eval_record)

        observation = train_env.reset()
        while total_timesteps < target_timesteps:
            iteration += 1
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
            update_actor_optimizer_learning_rate(actor_optimizer, actor_lr, backbone_lr)

            observation, rollout_data, rollout_metrics = collect_teacher_rollout(
                teacher_policy=teacher_policy,
                env=train_env,
                initial_observation=observation,
                n_steps=int(ppo_config.n_steps),
                device=device,
            )
            total_timesteps += int(ppo_config.n_steps) * int(train_env.num_envs)

            actor_metrics = sandwich_distill_update(
                student_policy=policy,
                teacher_policy=teacher_policy,
                actor_optimizer=actor_optimizer,
                rollout_data=rollout_data,
                search_space=search_space,
                action_space=train_env.action_space,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                max_grad_norm=float(ppo_config.max_grad_norm),
                random_subnets=int(args.random_subnets),
                temperature=float(args.distill_temperature),
                ema_policy=ema_policy,
                z_dyn_coef=z_dyn_coef,
            )
            if ema_policy is not None and z_dyn_coef > 0.0:
                update_ema_model(ema_policy, policy, tau=float(ppo_config.ema_tau))

            train_record = {
                "type": "train",
                "iteration": int(iteration),
                "total_timesteps": int(total_timesteps),
                "progress_remaining": float(progress_remaining),
                "actor_lr": float(actor_lr),
                "backbone_lr": float(backbone_lr),
                **rollout_metrics,
                **actor_metrics,
            }
            last_train_record = dict(train_record)
            log_record(metrics_path, wandb_run, train_record, step=total_timesteps)

            progress_bar.update(
                max(0, min(total_timesteps, target_timesteps) - progress_bar.n)
            )
            progress_bar.set_postfix(
                {
                    "ret": f"{rollout_metrics['rollout/ep_return']:.3g}",
                    "loss": f"{actor_metrics['actor/loss']:.3g}",
                    "max": f"{actor_metrics['actor/max_policy_distill_loss']:.3g}",
                    "min": f"{actor_metrics['actor/min_policy_distill_loss']:.3g}",
                    "lr": f"{actor_lr:.2g}",
                },
                refresh=True,
            )

            if total_timesteps >= next_eval_timestep:
                while next_eval_timestep <= total_timesteps:
                    next_eval_timestep += eval_freq
                eval_record = evaluate_and_log_sandwich(
                    policy=policy,
                    train_env=train_env,
                    eval_env=eval_env,
                    search_space=search_space,
                    ppo_config=ppo_config,
                    device=device,
                    metrics_path=metrics_path,
                    wandb_run=wandb_run,
                    total_timesteps=total_timesteps,
                    iteration=iteration,
                    phase="periodic",
                )
                final_eval_record = dict(eval_record)
                progress_bar.write(
                    f"stage2_distill_eval step={total_timesteps} "
                    f"max_subnet_ep_return={eval_record['eval/max_subnet_ep_return']:.6g} "
                    f"min_subnet_ep_return={eval_record['eval/min_subnet_ep_return']:.6g}"
                )
                maybe_save_best_checkpoint(eval_record)

            _save_distill_checkpoint(
                last_checkpoint_path,
                args=args,
                ppo_config=ppo_config,
                policy=policy,
                critic_model=critic_model,
                critic_optimizer=critic_optimizer,
                actor_optimizer=actor_optimizer,
                total_timesteps=total_timesteps,
                iteration=iteration,
                stage_name=stage_name,
                ema_policy=ema_policy,
            )

        final_eval_record = evaluate_and_log_sandwich(
            policy=policy,
            train_env=train_env,
            eval_env=eval_env,
            search_space=search_space,
            ppo_config=ppo_config,
            device=device,
            metrics_path=metrics_path,
            wandb_run=wandb_run,
            total_timesteps=total_timesteps,
            iteration=iteration,
            phase="final",
        )
        maybe_save_best_checkpoint(final_eval_record)

        _save_distill_checkpoint(
            last_checkpoint_path,
            args=args,
            ppo_config=ppo_config,
            policy=policy,
            critic_model=critic_model,
            critic_optimizer=critic_optimizer,
            actor_optimizer=actor_optimizer,
            total_timesteps=total_timesteps,
            iteration=iteration,
            stage_name=stage_name,
            ema_policy=ema_policy,
        )

        manifest = {
            "stage": stage_name,
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "last_checkpoint": str(last_checkpoint_path),
            "best_checkpoint": str(best_checkpoint_path),
            "metrics": str(metrics_path),
            "search_space": str(search_space_path),
            "configured_total_timesteps": int(target_timesteps),
            "total_timesteps": int(total_timesteps),
            "max_arch": search_space.max_arch().to_dict(),
            "min_arch": search_space.min_arch().to_dict(),
            "best_eval_max_subnet_ep_return": best_eval_max_subnet_ep_return,
            "best_eval": best_eval_record,
            "last_eval": final_eval_record,
            "last_train": last_train_record,
            "notes": {
                "rollout_policy": "frozen_teacher_max_subnet",
                "actor_update": "sandwich_policy_kl_distillation"
                if z_dyn_coef <= 0.0
                else "sandwich_policy_kl_distillation_plus_latent_dynamics",
                "subnet_objective": "max_min_random_all_policy_distillation"
                if z_dyn_coef <= 0.0
                else "max_min_random_policy_distillation_plus_latent_dynamics",
                "critic": "initialized_only_for_checkpoint_compatibility",
                "z_dyn_coef": z_dyn_coef,
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
                "best_eval_max_subnet_ep_return": best_eval_max_subnet_ep_return,
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
