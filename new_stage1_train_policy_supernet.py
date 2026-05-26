from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import (
    VecEnv,
    sync_envs_normalization,
    unwrap_vec_normalize,
)
from tqdm.auto import tqdm

from config_utils import (
    add_ppo_config_args,
    build_run_config,
    load_ppo_config,
    ppo_config_to_dict,
)
from env_utils import EVAL_SEED_OFFSET
from ppo_utils import (
    append_jsonl_record,
    make_vec_env_from_ppo_config,
    parse_hidden_sizes,
    parse_optional_float,
    parse_schedule_value,
    resolve_activation_fn,
    resolve_device,
)
from representation_losses import (
    LatentDynamicsPredictor,
    ProjectionHead,
    encode_action_batch,
    get_action_dim,
)
from supernet_backbone import (
    ArchConfig,
    SearchSpace,
    SupernetCNNBackbone,
    infer_input_channels,
)
from trajectory_data import resolve_terminal_next_observations, split_done_flags
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="New stage 1: train a policy supernet with PPO max-subnet updates and subnet distillation.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument(
        "--output_dir",
        default="runs/new_stage1_policy_supernet",
        help="Directory for checkpoints, metrics, and manifest.",
    )
    parser.add_argument(
        "--random_subnets",
        type=int,
        default=2,
        help="Number of random subnets distilled per PPO iteration, in addition to the min subnet.",
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=1.0,
        help="Temperature for discrete policy KL distillation.",
    )
    return parser.parse_args()


def build_mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    activation_fn: type[nn.Module] | None,
) -> tuple[nn.Module, int]:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    activation_class = activation_fn or nn.Tanh
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation_class())
        last_dim = int(hidden_dim)
    if not layers:
        return nn.Identity(), last_dim
    return nn.Sequential(*layers), last_dim


def maybe_orthogonal_init(module: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    for item in module.modules():
        if isinstance(item, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(item.weight, gain=math.sqrt(2.0))
            if item.bias is not None:
                nn.init.constant_(item.bias, 0.0)


class PolicySupernet(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        search_space: SearchSpace,
        features_dim: int,
        policy_net_arch: tuple[int, ...],
        activation_fn: type[nn.Module] | None,
        log_std_init: float | None,
        ortho_init: bool,
        projection_dim: int,
        predictor_hidden_dim: int,
    ):
        super().__init__()
        if (
            not isinstance(observation_space, spaces.Box)
            or len(observation_space.shape) != 3
        ):
            raise TypeError("PolicySupernet expects image Box observations.")
        if not isinstance(action_space, (spaces.Discrete, spaces.Box)):
            raise TypeError("Only Discrete and Box action spaces are supported.")

        self.action_space = action_space
        self.search_space = search_space
        self.features_dim = int(features_dim)
        self.backbone = SupernetCNNBackbone(
            input_channels=infer_input_channels(tuple(observation_space.shape)),
            search_space=search_space,
            feature_dim=self.features_dim,
        )
        self.policy_net, latent_dim = build_mlp(
            input_dim=self.features_dim,
            hidden_sizes=policy_net_arch,
            activation_fn=activation_fn,
        )
        self.latent_dim = int(latent_dim)

        if isinstance(action_space, spaces.Discrete):
            self.action_kind = "discrete"
            self.action_dim = int(action_space.n)
            self.action_net = nn.Linear(self.latent_dim, self.action_dim)
            self.log_std = None
        else:
            self.action_kind = "box"
            self.action_shape = tuple(int(value) for value in action_space.shape)
            self.action_dim = int(np.prod(self.action_shape, dtype=np.int64))
            self.action_net = nn.Linear(self.latent_dim, self.action_dim)
            initial_log_std = 0.0 if log_std_init is None else float(log_std_init)
            self.log_std = nn.Parameter(
                torch.ones(self.action_dim, dtype=torch.float32) * initial_log_std
            )

        maybe_orthogonal_init(self.policy_net, ortho_init)
        maybe_orthogonal_init(self.action_net, ortho_init)
        self.projection = ProjectionHead(features_dim, projection_dim)
        self.predictor = LatentDynamicsPredictor(
            latent_dim=projection_dim,
            action_dim=get_action_dim(action_space),
            hidden_dim=predictor_hidden_dim,
        )

    def set_active_arch(self, arch: ArchConfig) -> None:
        self.backbone.set_sample_config(arch)

    def set_max_arch(self) -> None:
        self.backbone.set_max_arch()

    def set_min_arch(self) -> None:
        self.backbone.set_min_arch()

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.backbone(observations)

    def distribution_params_from_features(
        self, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        action_output = self.action_net(self.policy_net(features))
        if self.action_kind == "discrete":
            return {"logits": action_output}
        if self.log_std is None:
            raise RuntimeError("Continuous policy is missing log_std.")
        return {
            "mean": action_output,
            "log_std": self.log_std.view(1, -1).expand_as(action_output),
        }

    def distribution_params(
        self, observations: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.distribution_params_from_features(self.encode(observations))

    def distribution_from_params(self, params: Mapping[str, torch.Tensor]):
        if self.action_kind == "discrete":
            return torch.distributions.Categorical(logits=params["logits"])
        std = torch.exp(params["log_std"])
        base_dist = torch.distributions.Normal(params["mean"], std)
        return torch.distributions.Independent(base_dist, 1)

    def act(
        self,
        observations: torch.Tensor,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.distribution_params(observations)
        distribution = self.distribution_from_params(params)
        if self.action_kind == "discrete":
            if deterministic:
                actions = torch.argmax(params["logits"], dim=-1)
            else:
                actions = distribution.sample()
        else:
            actions = params["mean"] if deterministic else distribution.sample()
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return actions, log_prob, entropy

    def evaluate_actions_from_params(
        self,
        params: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution_from_params(params)
        if self.action_kind == "discrete":
            prepared_actions = actions.long().view(-1)
        else:
            prepared_actions = actions.float().view(actions.size(0), self.action_dim)
        log_prob = distribution.log_prob(prepared_actions)
        entropy = distribution.entropy()
        return log_prob, entropy

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.evaluate_actions_from_params(
            self.distribution_params(observations), actions
        )

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    def project_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return self.project_features(self.encode(observations))


def build_sb3_critic_model(
    ppo_config: DictConfig,
    env: VecEnv,
    learning_rate: Any,
) -> PPO:
    policy_kwargs: dict[str, Any] = {
        "net_arch": {
            "pi": [],
            "vf": list(parse_hidden_sizes(ppo_config.value_net_arch)),
        },
        "features_extractor_kwargs": {"features_dim": ppo_config.features_dim},
    }
    activation_fn = resolve_activation_fn(ppo_config.activation_fn)
    if activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_fn
    if ppo_config.ortho_init is not None:
        policy_kwargs["ortho_init"] = ppo_config.ortho_init
    log_std_init = parse_optional_float(ppo_config.log_std_init)
    if log_std_init is not None:
        policy_kwargs["log_std_init"] = log_std_init

    return PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=parse_schedule_value(ppo_config.clip_range),
        normalize_advantage=ppo_config.normalize_advantage,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        target_kl=parse_optional_float(ppo_config.target_kl),
        stats_window_size=ppo_config.stats_window_size,
        tensorboard_log=ppo_config.tensorboard_log,
        policy_kwargs=policy_kwargs,
        use_sde=ppo_config.use_sde and isinstance(env.action_space, spaces.Box),
        sde_sample_freq=ppo_config.sde_sample_freq,
        seed=ppo_config.seed,
        device=str(ppo_config.device),
        verbose=0 if ppo_config.quiet else 1,
    )


def predict_critic_values(
    critic_model: PPO, observations: torch.Tensor
) -> torch.Tensor:
    return critic_model.policy.predict_values(observations).flatten()


@dataclass(frozen=True)
class DynamicsRolloutSamples:
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor
    dynamics_masks: torch.Tensor


class DynamicsRolloutBuffer(RolloutBuffer):
    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=self.observation_space.dtype,
        )
        self.dynamics_masks = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

    def add_transition(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        dynamics_mask: np.ndarray,
    ) -> None:
        self.next_observations[self.pos] = np.array(next_obs)
        self.dynamics_masks[self.pos] = np.array(dynamics_mask, dtype=np.float32)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: int | None = None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for tensor_name in (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "next_observations",
                "dynamics_masks",
            ):
                self.__dict__[tensor_name] = self.swap_and_flatten(
                    self.__dict__[tensor_name]
                )
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_index = 0
        while start_index < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_index : start_index + batch_size])
            start_index += batch_size

    def _get_samples(self, batch_indices: np.ndarray) -> DynamicsRolloutSamples:
        data = (
            self.observations[batch_indices],
            self.actions[batch_indices].astype(np.float32, copy=False),
            self.values[batch_indices].flatten(),
            self.log_probs[batch_indices].flatten(),
            self.advantages[batch_indices].flatten(),
            self.returns[batch_indices].flatten(),
            self.next_observations[batch_indices],
            self.dynamics_masks[batch_indices].flatten(),
        )
        return DynamicsRolloutSamples(*tuple(map(self.to_torch, data)))


def bootstrap_time_limit_rewards(
    rewards: np.ndarray,
    dones: np.ndarray,
    infos: list[dict[str, Any]],
    critic_model: PPO,
    device: torch.device,
    gamma: float,
) -> np.ndarray:
    adjusted_rewards = np.asarray(rewards, dtype=np.float32).copy()
    terminal_observations = []
    terminal_indices = []
    for env_index, info in enumerate(infos):
        if not bool(dones[env_index]):
            continue
        if not bool(info.get("TimeLimit.truncated", False)):
            continue
        terminal_observation = info.get("terminal_observation")
        if terminal_observation is None:
            continue
        terminal_indices.append(env_index)
        terminal_observations.append(np.asarray(terminal_observation))

    if not terminal_observations:
        return adjusted_rewards

    terminal_tensor = torch.as_tensor(np.stack(terminal_observations), device=device)
    with torch.no_grad():
        terminal_values = (
            predict_critic_values(critic_model, terminal_tensor).detach().cpu().numpy()
        )
    for offset, env_index in enumerate(terminal_indices):
        adjusted_rewards[env_index] += float(gamma) * float(terminal_values[offset])
    return adjusted_rewards


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
    current_rollout_returns = np.zeros(num_envs, dtype=np.float64)
    current_rollout_lengths = np.zeros(num_envs, dtype=np.int64)
    last_dones = episode_starts

    with torch.no_grad():
        for _ in range(n_steps):
            observation_tensor = torch.as_tensor(observation, device=device)
            action_tensor, log_prob_tensor, _ = policy.act(
                observation_tensor, deterministic=False
            )
            value_tensor = predict_critic_values(critic_model, observation_tensor)

            if isinstance(env.action_space, spaces.Discrete):
                stored_actions = action_tensor.detach().cpu().numpy().reshape((-1, 1))
                env_actions = stored_actions.reshape(-1)
            else:
                action_shape = tuple(int(value) for value in env.action_space.shape)
                stored_actions = (
                    action_tensor.detach()
                    .cpu()
                    .numpy()
                    .reshape((num_envs, *action_shape))
                    .astype(np.float32)
                )
                env_actions = np.clip(
                    stored_actions, env.action_space.low, env.action_space.high
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
            reward_array = np.asarray(raw_rewards, dtype=np.float64)
            done_array = np.asarray(raw_dones, dtype=np.bool_)
            current_rollout_returns += reward_array
            current_rollout_lengths += 1
            rollout_done_count += int(done_array.sum())
            for env_index, done in enumerate(done_array):
                if not bool(done):
                    continue
                episode_info = (
                    info_list[env_index].get("episode")
                    if isinstance(info_list[env_index], Mapping)
                    else None
                )
                if isinstance(episode_info, Mapping) and "r" in episode_info:
                    episode_return = float(episode_info["r"])
                else:
                    episode_return = float(current_rollout_returns[env_index])
                if isinstance(episode_info, Mapping) and "l" in episode_info:
                    episode_length = float(episode_info["l"])
                else:
                    episode_length = float(current_rollout_lengths[env_index])
                rollout_episode_returns.append(episode_return)
                rollout_episode_lengths.append(episode_length)
                current_rollout_returns[env_index] = 0.0
                current_rollout_lengths[env_index] = 0
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


def update_ema_model(
    ema_policy: PolicySupernet,
    online_policy: PolicySupernet,
    tau: float,
) -> None:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("ema_tau must be in [0, 1].")
    with torch.no_grad():
        for ema_param, online_param in zip(
            ema_policy.parameters(), online_policy.parameters(), strict=True
        ):
            ema_param.mul_(float(tau)).add_(
                online_param.detach(), alpha=1.0 - float(tau)
            )
        for ema_buffer, online_buffer in zip(
            ema_policy.buffers(), online_policy.buffers(), strict=True
        ):
            ema_buffer.copy_(online_buffer)


def compute_dynamics_loss(
    online_policy: PolicySupernet,
    ema_policy: PolicySupernet,
    arch: ArchConfig,
    start_features: torch.Tensor,
    next_observations: torch.Tensor,
    actions: torch.Tensor,
    action_space: spaces.Space,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    if sample_weights is not None:
        active_mask = sample_weights.to(device=start_features.device).view(-1) > 0.0
        if not bool(active_mask.any()):
            return torch.zeros((), dtype=torch.float32, device=start_features.device)
        start_features = start_features[active_mask]
        next_observations = next_observations[active_mask]
        actions = actions[active_mask]
        sample_weights = sample_weights.to(device=start_features.device)[active_mask]

    start_latent = online_policy.project_features(start_features)
    action_features = encode_action_batch(actions, action_space).to(
        device=start_features.device
    )
    predicted_next_latent = start_latent + online_policy.predictor(
        start_latent, action_features
    )
    with torch.no_grad():
        ema_policy.set_active_arch(arch)
        target_next_latent = ema_policy.project_observations(next_observations)

    per_sample_loss = 2.0 - 2.0 * F.cosine_similarity(
        predicted_next_latent,
        target_next_latent.detach(),
        dim=-1,
        eps=1e-6,
    )
    if sample_weights is None:
        return per_sample_loss.mean()
    prepared_weights = sample_weights.to(
        dtype=per_sample_loss.dtype, device=per_sample_loss.device
    ).view_as(per_sample_loss)
    return (
        per_sample_loss * prepared_weights
    ).sum() / prepared_weights.sum().clamp_min(1.0)


def policy_kl_distillation_loss(
    action_space: spaces.Space,
    teacher_params: Mapping[str, torch.Tensor],
    student_params: Mapping[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    if isinstance(action_space, spaces.Discrete):
        if temperature <= 0.0:
            raise ValueError("distill_temperature must be positive.")
        teacher_log_probs = F.log_softmax(
            teacher_params["logits"] / float(temperature), dim=-1
        )
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(
            student_params["logits"] / float(temperature), dim=-1
        )
        return (
            F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            * float(temperature) ** 2
        )

    if isinstance(action_space, spaces.Box):
        teacher_mean = teacher_params["mean"].detach()
        teacher_log_std = teacher_params["log_std"].detach()
        student_mean = student_params["mean"]
        student_log_std = student_params["log_std"]
        teacher_var = torch.exp(2.0 * teacher_log_std)
        student_var = torch.exp(2.0 * student_log_std)
        mean_delta = student_mean - teacher_mean
        kl_per_dim = (
            student_log_std
            - teacher_log_std
            + (teacher_var + mean_delta.pow(2)) / (2.0 * student_var)
            - 0.5
        )
        return kl_per_dim.sum(dim=-1).mean()

    raise TypeError(f"Unsupported action space: {type(action_space).__name__}")


def sandwich_actor_update(
    policy: PolicySupernet,
    ema_policy: PolicySupernet,
    actor_optimizer: torch.optim.Optimizer,
    rollout_buffer: DynamicsRolloutBuffer,
    search_space: SearchSpace,
    action_space: spaces.Space,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    normalize_advantage: bool,
    ent_coef: float,
    max_grad_norm: float,
    beta_dyn: float,
    target_kl: float | None,
    random_subnets: int,
    temperature: float,
) -> dict[str, float]:
    policy.train()
    ema_policy.eval()

    loss_sum = 0.0
    max_loss_sum = 0.0
    policy_loss_sum = 0.0
    entropy_loss_sum = 0.0
    max_dynamic_loss_sum = 0.0
    subnet_loss_sum = 0.0
    policy_distill_loss_sum = 0.0
    subnet_dynamic_loss_sum = 0.0
    approx_kl_sum = 0.0
    clip_fraction_sum = 0.0
    update_count = 0
    continue_training = True
    max_arch = search_space.max_arch()
    min_arch = search_space.min_arch()

    for _ in range(int(n_epochs)):
        if not continue_training:
            break
        for rollout_data in rollout_buffer.get(batch_size):
            batch_observations = rollout_data.observations
            batch_actions = rollout_data.actions
            if isinstance(action_space, spaces.Discrete):
                batch_actions = batch_actions.long().flatten()
            batch_old_log_probs = rollout_data.old_log_prob
            batch_advantages = rollout_data.advantages
            if normalize_advantage and batch_advantages.numel() > 1:
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )

            actor_optimizer.zero_grad(set_to_none=True)

            policy.set_active_arch(max_arch)
            max_features = policy.encode(batch_observations)
            max_params = policy.distribution_params_from_features(max_features)
            new_log_probs, entropy = policy.evaluate_actions_from_params(
                max_params, batch_actions
            )
            log_ratio = new_log_probs - batch_old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped_loss = batch_advantages * ratio
            clipped_loss = batch_advantages * torch.clamp(
                ratio, 1.0 - clip_range, 1.0 + clip_range
            )
            policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
            entropy_loss = -entropy.mean()

            max_dyn_loss = compute_dynamics_loss(
                online_policy=policy,
                ema_policy=ema_policy,
                arch=max_arch,
                start_features=max_features,
                next_observations=rollout_data.next_observations,
                actions=batch_actions,
                action_space=action_space,
                sample_weights=rollout_data.dynamics_masks,
            )
            max_loss = (
                policy_loss
                + float(ent_coef) * entropy_loss
                + float(beta_dyn) * max_dyn_loss
            )
            max_loss_value = float(max_loss.detach().cpu())
            policy_loss_value = float(policy_loss.detach().cpu())
            entropy_loss_value = float(entropy_loss.detach().cpu())
            max_dynamic_loss_value = float(max_dyn_loss.detach().cpu())
            approx_kl = torch.mean(((ratio - 1.0) - log_ratio).detach())
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1.0) > clip_range).float().detach()
            )
            max_loss.backward()

            with torch.no_grad():
                ema_policy.set_max_arch()
                teacher_features = ema_policy.encode(batch_observations)
                teacher_params = ema_policy.distribution_params_from_features(
                    teacher_features
                )
                teacher_params = {
                    key: value.detach() for key, value in teacher_params.items()
                }

            sampled_arches = [min_arch]
            for _sample_index in range(max(0, int(random_subnets))):
                sampled_arches.append(search_space.sample_arch())

            subnet_loss_values = []
            policy_distill_loss_values = []
            subnet_dyn_loss_values = []
            subnet_scale = 1.0 / float(len(sampled_arches))
            for arch in sampled_arches:
                policy.set_active_arch(arch)
                subnet_features = policy.encode(batch_observations)
                student_params = policy.distribution_params_from_features(
                    subnet_features
                )
                policy_distill_loss = policy_kl_distillation_loss(
                    action_space=action_space,
                    teacher_params=teacher_params,
                    student_params=student_params,
                    temperature=temperature,
                )
                subnet_dyn_loss = compute_dynamics_loss(
                    online_policy=policy,
                    ema_policy=ema_policy,
                    arch=arch,
                    start_features=subnet_features,
                    next_observations=rollout_data.next_observations,
                    actions=batch_actions,
                    action_space=action_space,
                    sample_weights=rollout_data.dynamics_masks,
                )
                subnet_loss = policy_distill_loss + float(beta_dyn) * subnet_dyn_loss
                subnet_loss_values.append(float(subnet_loss.detach().cpu()))
                policy_distill_loss_values.append(
                    float(policy_distill_loss.detach().cpu())
                )
                subnet_dyn_loss_values.append(float(subnet_dyn_loss.detach().cpu()))
                (subnet_loss * subnet_scale).backward()

            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float(max_grad_norm)
                )
            actor_optimizer.step()

            subnet_loss_value = float(np.mean(subnet_loss_values))
            policy_distill_loss_value = float(np.mean(policy_distill_loss_values))
            subnet_dynamic_loss_value = float(np.mean(subnet_dyn_loss_values))
            update_count += 1
            loss_sum += max_loss_value + subnet_loss_value
            max_loss_sum += max_loss_value
            policy_loss_sum += policy_loss_value
            entropy_loss_sum += entropy_loss_value
            max_dynamic_loss_sum += max_dynamic_loss_value
            subnet_loss_sum += subnet_loss_value
            policy_distill_loss_sum += policy_distill_loss_value
            subnet_dynamic_loss_sum += subnet_dynamic_loss_value
            approx_kl_sum += float(approx_kl.cpu())
            clip_fraction_sum += float(clip_fraction.cpu())

            if target_kl is not None and float(approx_kl.cpu()) > 1.5 * float(
                target_kl
            ):
                continue_training = False
                break

    denominator = float(max(1, update_count))
    return {
        "actor/loss": loss_sum / denominator,
        "actor/max_loss": max_loss_sum / denominator,
        "actor/policy_loss": policy_loss_sum / denominator,
        "actor/entropy_loss": entropy_loss_sum / denominator,
        "actor/max_dynamic_loss": max_dynamic_loss_sum / denominator,
        "actor/subnet_loss": subnet_loss_sum / denominator,
        "actor/policy_distill_loss": policy_distill_loss_sum / denominator,
        "actor/subnet_dynamic_loss": subnet_dynamic_loss_sum / denominator,
        "actor/approx_kl": approx_kl_sum / denominator,
        "actor/clip_fraction": clip_fraction_sum / denominator,
    }


def critic_update(
    critic_model: PPO,
    optimizer: torch.optim.Optimizer,
    rollout_buffer: DynamicsRolloutBuffer,
    n_epochs: int,
    batch_size: int,
    max_grad_norm: float,
) -> dict[str, float]:
    critic_model.policy.train()
    loss_sum = 0.0
    update_count = 0
    for _ in range(int(n_epochs)):
        for rollout_data in rollout_buffer.get(batch_size):
            values = predict_critic_values(critic_model, rollout_data.observations)
            loss = F.mse_loss(values, rollout_data.returns)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    critic_model.policy.parameters(), float(max_grad_norm)
                )
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            update_count += 1
    denominator = float(max(1, update_count))
    return {
        "critic/loss": loss_sum / denominator,
    }


def evaluate_actor_subnet(
    policy: PolicySupernet,
    train_env: VecEnv,
    eval_env: VecEnv,
    arch: ArchConfig,
    n_eval_episodes: int,
    deterministic: bool,
    device: torch.device,
) -> dict[str, float]:
    sync_envs_normalization(train_env, eval_env)
    eval_vec_normalize = unwrap_vec_normalize(eval_env)
    if eval_vec_normalize is not None:
        eval_vec_normalize.training = False
        eval_vec_normalize.norm_reward = False

    policy.eval()
    policy.set_active_arch(arch)
    observations = eval_env.reset()
    current_returns = np.zeros(eval_env.num_envs, dtype=np.float64)
    current_lengths = np.zeros(eval_env.num_envs, dtype=np.int64)
    episode_returns: list[float] = []
    episode_lengths: list[float] = []
    with torch.no_grad():
        while len(episode_returns) < n_eval_episodes:
            observation_tensor = torch.as_tensor(observations, device=device)
            actions, _, _ = policy.act(observation_tensor, deterministic=deterministic)
            if isinstance(eval_env.action_space, spaces.Discrete):
                env_actions = actions.detach().cpu().numpy().reshape(-1)
            else:
                action_shape = tuple(
                    int(value) for value in eval_env.action_space.shape
                )
                env_actions = (
                    actions.detach().cpu().numpy().reshape((-1, *action_shape))
                )
                env_actions = np.clip(
                    env_actions, eval_env.action_space.low, eval_env.action_space.high
                )
            observations, rewards, dones, infos = eval_env.step(env_actions)
            current_returns += np.asarray(rewards, dtype=np.float64)
            current_lengths += 1
            for env_index, done in enumerate(dones):
                if not bool(done):
                    continue
                episode_info = (
                    infos[env_index].get("episode")
                    if isinstance(infos[env_index], dict)
                    else None
                )
                if isinstance(episode_info, Mapping) and "r" in episode_info:
                    episode_return = float(episode_info["r"])
                else:
                    episode_return = float(current_returns[env_index])
                if isinstance(episode_info, Mapping) and "l" in episode_info:
                    episode_length = float(episode_info["l"])
                else:
                    episode_length = float(current_lengths[env_index])
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                current_returns[env_index] = 0.0
                current_lengths[env_index] = 0
                if len(episode_returns) >= n_eval_episodes:
                    break

    return {
        "ep_return": float(np.mean(episode_returns)),
        "ep_return_std": float(np.std(episode_returns)),
        "ep_length": float(np.mean(episode_lengths)),
        "ep_length_std": float(np.std(episode_lengths)),
    }


def save_supernet_checkpoint(
    path: Path,
    args: argparse.Namespace,
    ppo_config: DictConfig,
    policy: PolicySupernet,
    ema_policy: PolicySupernet,
    critic_model: PPO,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    search_space: SearchSpace,
    total_timesteps: int,
    iteration: int,
) -> None:
    torch.save(
        {
            "stage": "new_stage1_policy_supernet",
            "total_timesteps": int(total_timesteps),
            "iteration": int(iteration),
            "policy_state_dict": policy.state_dict(),
            "ema_policy_state_dict": ema_policy.state_dict(),
            "critic_policy_state_dict": critic_model.policy.state_dict(),
            "backbone_state_dict": policy.backbone.state_dict(),
            "policy_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "search_space": search_space.to_dict(),
            "max_arch": search_space.max_arch().to_dict(),
            "min_arch": search_space.min_arch().to_dict(),
            "features_dim": int(ppo_config.features_dim),
            "projection_dim": int(ppo_config.projection_dim),
            "predictor_hidden_dim": int(ppo_config.predictor_hidden_dim),
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
    last_checkpoint_path = output_dir / "policy_supernet_last.pt"
    best_checkpoint_path = output_dir / "policy_supernet_best.pt"

    random.seed(int(ppo_config.seed))
    np.random.seed(int(ppo_config.seed))
    torch.manual_seed(int(ppo_config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(ppo_config.seed))
    device = resolve_device(str(ppo_config.device))
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run(
        "new_stage1_train_policy_supernet", run_config, output_dir
    )
    search_space = SearchSpace()
    search_space_path.write_text(json.dumps(search_space.to_dict(), indent=2))

    total_timesteps_target = int(ppo_config.total_timesteps)
    eval_freq = int(ppo_config.eval_freq)
    eval_episodes = int(ppo_config.eval_episodes)
    if eval_freq <= 0 or eval_episodes <= 0:
        raise ValueError(
            "new_stage1 requires positive ppo.eval_freq and ppo.eval_episodes."
        )

    train_env = make_vec_env_from_ppo_config(
        ppo_config, seed=int(ppo_config.seed), n_envs=ppo_config.train_n_envs
    )
    eval_env = make_vec_env_from_ppo_config(
        ppo_config,
        seed=int(ppo_config.seed) + EVAL_SEED_OFFSET,
        n_envs=ppo_config.eval_n_envs,
    )
    progress_bar = None
    try:
        policy = PolicySupernet(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            search_space=search_space,
            features_dim=ppo_config.features_dim,
            policy_net_arch=parse_hidden_sizes(ppo_config.policy_net_arch),
            activation_fn=resolve_activation_fn(ppo_config.activation_fn),
            log_std_init=parse_optional_float(ppo_config.log_std_init),
            ortho_init=ppo_config.ortho_init,
            projection_dim=ppo_config.projection_dim,
            predictor_hidden_dim=ppo_config.predictor_hidden_dim,
        ).to(device)
        if ppo_config.beta_dyn > 0.0:
            ema_policy = copy.deepcopy(policy).to(device)
            for parameter in ema_policy.parameters():
                parameter.requires_grad_(False)
        else:
            ema_policy = policy
        critic_lr_schedule = parse_schedule_value(ppo_config.critic_lr)
        policy_head_lr_schedule = parse_schedule_value(ppo_config.policy_head_lr)
        policy_backbone_lr_schedule = parse_schedule_value(
            ppo_config.policy_backbone_lr
        )
        clip_range_schedule = parse_schedule_value(ppo_config.clip_range)

        critic_model = build_sb3_critic_model(
            ppo_config=ppo_config,
            env=train_env,
            learning_rate=critic_lr_schedule,
        )

        initial_head_lr = (
            float(policy_head_lr_schedule(1.0))
            if callable(policy_head_lr_schedule)
            else float(policy_head_lr_schedule)
        )
        initial_backbone_lr = (
            float(policy_backbone_lr_schedule(1.0))
            if callable(policy_backbone_lr_schedule)
            else float(policy_backbone_lr_schedule)
        )

        backbone_params = list(policy.backbone.parameters())
        head_params = [
            p for n, p in policy.named_parameters() if not n.startswith("backbone.")
        ]
        actor_optimizer = torch.optim.Adam(
            [
                {
                    "params": backbone_params,
                    "lr": initial_backbone_lr,
                    "group_name": "backbone",
                },
                {"params": head_params, "lr": initial_head_lr, "group_name": "head"},
            ],
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=0.0,
        )
        critic_optimizer = critic_model.policy.optimizer
        rollout_buffer = DynamicsRolloutBuffer(
            buffer_size=ppo_config.n_steps,
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            device=device,
            gae_lambda=ppo_config.gae_lambda,
            gamma=ppo_config.gamma,
            n_envs=train_env.num_envs,
        )

        observation = train_env.reset()
        episode_starts = np.ones((train_env.num_envs,), dtype=np.bool_)
        total_timesteps = 0
        iteration = 0
        next_eval_timestep = min(eval_freq, total_timesteps_target)
        best_eval_max_subnet_ep_return: float | None = None
        final_eval_record: dict[str, Any] = {}
        best_eval_record: dict[str, Any] = {}

        append_jsonl_record(
            metrics_path,
            {
                "type": "config",
                "total_timesteps_target": total_timesteps_target,
                "device": str(device),
                "action_space": type(train_env.action_space).__name__,
                "observation_shape": list(train_env.observation_space.shape),
                "dynamics_loss": "normalized_cosine_distance",
                "actor_use_sde_ignored": ppo_config.use_sde,
            },
        )
        progress_bar = tqdm(
            total=ppo_config.total_timesteps,
            desc="new_stage1_policy_supernet",
            unit="step",
            dynamic_ncols=True,
            disable=ppo_config.quiet,
        )

        while total_timesteps < total_timesteps_target:
            iteration += 1
            progress_remaining = 1.0 - float(total_timesteps) / float(
                max(1, total_timesteps_target)
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
            for param_group in actor_optimizer.param_groups:
                if param_group.get("group_name") == "backbone":
                    param_group["lr"] = backbone_lr
                else:
                    param_group["lr"] = actor_lr
            for param_group in critic_optimizer.param_groups:
                param_group["lr"] = critic_lr

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
            progress_bar.update(
                max(0, min(total_timesteps, total_timesteps_target) - progress_bar.n)
            )

            actor_metrics = sandwich_actor_update(
                policy=policy,
                ema_policy=ema_policy,
                actor_optimizer=actor_optimizer,
                rollout_buffer=rollout_buffer,
                search_space=search_space,
                action_space=train_env.action_space,
                n_epochs=ppo_config.n_epochs,
                batch_size=ppo_config.batch_size,
                clip_range=clip_range,
                normalize_advantage=ppo_config.normalize_advantage,
                ent_coef=ppo_config.ent_coef,
                max_grad_norm=ppo_config.max_grad_norm,
                beta_dyn=ppo_config.beta_dyn,
                target_kl=parse_optional_float(ppo_config.target_kl),
                random_subnets=args.random_subnets,
                temperature=args.distill_temperature,
            )
            if ppo_config.beta_dyn > 0.0:
                update_ema_model(ema_policy, policy, tau=ppo_config.ema_tau)
            critic_metrics = critic_update(
                critic_model=critic_model,
                optimizer=critic_optimizer,
                rollout_buffer=rollout_buffer,
                n_epochs=ppo_config.n_epochs,
                batch_size=ppo_config.batch_size,
                max_grad_norm=ppo_config.max_grad_norm,
            )

            if total_timesteps >= next_eval_timestep:
                while next_eval_timestep <= total_timesteps:
                    next_eval_timestep += eval_freq
                max_subnet_eval = evaluate_actor_subnet(
                    policy=policy,
                    train_env=train_env,
                    eval_env=eval_env,
                    arch=search_space.max_arch(),
                    n_eval_episodes=eval_episodes,
                    deterministic=bool(ppo_config.eval_deterministic),
                    device=device,
                )
                min_subnet_eval = evaluate_actor_subnet(
                    policy=policy,
                    train_env=train_env,
                    eval_env=eval_env,
                    arch=search_space.min_arch(),
                    n_eval_episodes=eval_episodes,
                    deterministic=bool(ppo_config.eval_deterministic),
                    device=device,
                )
                is_best_max_subnet = (
                    best_eval_max_subnet_ep_return is None
                    or max_subnet_eval["ep_return"] > best_eval_max_subnet_ep_return
                )
                if is_best_max_subnet:
                    best_eval_max_subnet_ep_return = float(max_subnet_eval["ep_return"])
                eval_record = {
                    "type": "eval",
                    "iteration": int(iteration),
                    "total_timesteps": int(total_timesteps),
                    "eval/max_subnet_ep_return": float(max_subnet_eval["ep_return"]),
                    "eval/max_subnet_ep_return_std": float(
                        max_subnet_eval["ep_return_std"]
                    ),
                    "eval/max_subnet_ep_length": float(max_subnet_eval["ep_length"]),
                    "eval/max_subnet_ep_length_std": float(
                        max_subnet_eval["ep_length_std"]
                    ),
                    "eval/min_subnet_ep_return": float(min_subnet_eval["ep_return"]),
                    "eval/min_subnet_ep_return_std": float(
                        min_subnet_eval["ep_return_std"]
                    ),
                    "eval/min_subnet_ep_length": float(min_subnet_eval["ep_length"]),
                    "eval/min_subnet_ep_length_std": float(
                        min_subnet_eval["ep_length_std"]
                    ),
                    "eval/best_max_subnet_ep_return": float(
                        best_eval_max_subnet_ep_return
                    ),
                    "eval/is_best_max_subnet": bool(is_best_max_subnet),
                }
                final_eval_record = dict(eval_record)
                append_jsonl_record(metrics_path, eval_record)
                log_wandb(
                    wandb_run,
                    {
                        key: value
                        for key, value in eval_record.items()
                        if isinstance(value, (int, float, bool))
                    },
                    step=total_timesteps,
                )
                progress_bar.write(
                    f"new_stage1_eval step={total_timesteps} "
                    f"max_subnet_ep_return={max_subnet_eval['ep_return']:.6g} "
                    f"min_subnet_ep_return={min_subnet_eval['ep_return']:.6g} "
                    f"is_best_max_subnet={is_best_max_subnet}"
                )
                if is_best_max_subnet:
                    best_eval_record = dict(eval_record)
                    save_supernet_checkpoint(
                        best_checkpoint_path,
                        args=args,
                        ppo_config=ppo_config,
                        policy=policy,
                        ema_policy=ema_policy,
                        critic_model=critic_model,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        search_space=search_space,
                        total_timesteps=total_timesteps,
                        iteration=iteration,
                    )

            record = {
                "type": "train",
                "iteration": int(iteration),
                "total_timesteps": int(total_timesteps),
                "progress_remaining": float(progress_remaining),
                "actor_lr": float(actor_lr),
                "critic_lr": float(critic_lr),
                "clip_range": float(clip_range),
                "beta_dyn": float(ppo_config.beta_dyn),
                **rollout_metrics,
                **actor_metrics,
                **critic_metrics,
            }
            append_jsonl_record(metrics_path, record)
            log_wandb(
                wandb_run,
                {
                    key: value
                    for key, value in record.items()
                    if isinstance(value, (int, float, bool))
                },
                step=total_timesteps,
            )
            progress_bar.set_postfix(
                {
                    "ret": f"{rollout_metrics['rollout/ep_return']:.3g}",
                    "actor": f"{actor_metrics['actor/loss']:.3g}",
                    "distill": f"{actor_metrics['actor/policy_distill_loss']:.3g}",
                    "dyn": f"{actor_metrics['actor/subnet_dynamic_loss']:.3g}",
                    "critic": f"{critic_metrics['critic/loss']:.3g}",
                    "lr": f"{actor_lr:.2g}",
                },
                refresh=True,
            )

            save_supernet_checkpoint(
                last_checkpoint_path,
                args=args,
                ppo_config=ppo_config,
                policy=policy,
                ema_policy=ema_policy,
                critic_model=critic_model,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                search_space=search_space,
                total_timesteps=total_timesteps,
                iteration=iteration,
            )

        manifest = {
            "stage": "new_stage1_policy_supernet",
            "last_checkpoint": str(last_checkpoint_path),
            "best_checkpoint": str(best_checkpoint_path),
            "metrics": str(metrics_path),
            "search_space": str(search_space_path),
            "total_timesteps": int(total_timesteps),
            "configured_total_timesteps": int(total_timesteps_target),
            "max_arch": search_space.max_arch().to_dict(),
            "min_arch": search_space.min_arch().to_dict(),
            "best_eval_max_subnet_ep_return": best_eval_max_subnet_ep_return,
            "best_eval": best_eval_record,
            "last_eval": final_eval_record,
            "checkpoint_fields": [
                "policy_state_dict",
                "ema_policy_state_dict",
                "critic_policy_state_dict",
                "backbone_state_dict",
                "search_space",
            ],
            "notes": {
                "rollout_policy": "max_subnet_only",
                "critic": "independent_no_distillation",
                "actor_update": "sandwich_max_ppo_plus_subnet_distillation",
                "subnet_objective": "policy_kl_distillation_plus_latent_dynamics_in_joint_actor_step",
                "latent_dynamics_loss": "normalized_cosine_distance",
                "latent_dynamics_transition": "residual_delta",
                "latent_target": "ema_same_active_subnet",
                "sde": "ignored_diag_gaussian_used"
                if ppo_config.use_sde
                else "not_requested",
            },
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        finish_wandb_run(wandb_run)
        return manifest
    finally:
        if progress_bar is not None:
            progress_bar.close()
        train_env.close()
        eval_env.close()


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    print(json.dumps(run(args, ppo_config), indent=2))


if __name__ == "__main__":
    main()
