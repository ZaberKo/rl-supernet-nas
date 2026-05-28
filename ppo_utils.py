from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecEnv,
    sync_envs_normalization,
    unwrap_vec_normalize,
)

from representation_losses import (
    LatentDynamicsPredictor,
    ProjectionHead,
    encode_action_batch,
    get_action_dim,
    latent_dynamics_loss,
)
from setup_utils import (
    parse_schedule_value,
    prefixed_metrics,
    resolve_activation_fn,
)
from supernet_backbone import (
    ArchConfig,
    SearchSpace,
    SupernetCNNBackbone,
)
from trajectory_data import (
    DynamicsRolloutBuffer,
    resolve_terminal_next_observations,
    split_done_flags,
)


def jsonable_metric_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_metric_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_metric_key(key: str) -> str:
    if key.startswith("time/"):
        return key.removeprefix("time/")
    return key


def append_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a") as metrics_file:
        metrics_file.write(json.dumps(jsonable_metric_value(record)) + "\n")



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
        policy_net_arch: list[int],
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
            input_channels=int(observation_space.shape[0]),
            search_space=search_space,
            feature_dim=self.features_dim,
        )
        self.policy_net, latent_dim = build_mlp(
            input_dim=self.features_dim,
            hidden_sizes=tuple(policy_net_arch),
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

    @property
    def elastic_num_params(self) -> int:
        total = int(self.backbone.elastic_num_params)
        total += sum(parameter.numel() for parameter in self.policy_net.parameters())
        total += sum(parameter.numel() for parameter in self.action_net.parameters())
        if self.log_std is not None:
            total += self.log_std.numel()
        return total


def build_sb3_critic_model(
    ppo_config: DictConfig,
    env: VecEnv,
    learning_rate: Any,
    device: str | torch.device | None = None,
) -> PPO:
    policy_kwargs: dict[str, Any] = {
        "net_arch": {
            "pi": [],
            "vf": list(ppo_config.value_net_arch),
        },
        "features_extractor_kwargs": {"features_dim": ppo_config.features_dim},
    }
    activation_fn = resolve_activation_fn(ppo_config.activation_fn)
    if activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_fn
    if ppo_config.ortho_init is not None:
        policy_kwargs["ortho_init"] = ppo_config.ortho_init
    log_std_init = ppo_config.log_std_init
    if log_std_init is not None:
        policy_kwargs["log_std_init"] = log_std_init

    if device is None:
        device = str(ppo_config.device)

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
        target_kl=ppo_config.target_kl,
        stats_window_size=ppo_config.stats_window_size,
        tensorboard_log=ppo_config.tensorboard_log,
        policy_kwargs=policy_kwargs,
        use_sde=ppo_config.use_sde and isinstance(env.action_space, spaces.Box),
        sde_sample_freq=ppo_config.sde_sample_freq,
        seed=ppo_config.seed,
        device=str(device),
        verbose=0 if ppo_config.quiet else 1,
    )


def predict_critic_values(
    critic_model: PPO, observations: torch.Tensor
) -> torch.Tensor:
    return critic_model.policy.predict_values(observations).flatten()


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


def actor_head_parameters(policy: PolicySupernet) -> list[torch.nn.Parameter]:
    parameters = list(policy.policy_net.parameters()) + list(
        policy.action_net.parameters()
    )
    if policy.log_std is not None:
        parameters.append(policy.log_std)
    return parameters


def configure_actor_optimizer(
    policy: PolicySupernet,
    head_lr: Any,
    backbone_lr: Any,
) -> torch.optim.Optimizer:
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    head_params = actor_head_parameters(policy)
    for parameter in head_params:
        parameter.requires_grad_(True)

    head_lr_start = float(head_lr(1.0)) if callable(head_lr) else float(head_lr)
    backbone_lr_start = (
        float(backbone_lr(1.0)) if callable(backbone_lr) else float(backbone_lr)
    )
    optimizer_kwargs = {
        "betas": (0.9, 0.999),
        "eps": 1e-5,
        "weight_decay": 0.0,
    }

    backbone_params = list(policy.backbone.parameters())
    if float(backbone_lr_start) <= 0.0:
        return torch.optim.Adam(head_params, lr=head_lr_start, **optimizer_kwargs)

    for parameter in backbone_params:
        parameter.requires_grad_(True)

    return torch.optim.Adam(
        [
            {
                "params": backbone_params,
                "lr": backbone_lr_start,
                "group_name": "backbone",
            },
            {"params": head_params, "lr": head_lr_start, "group_name": "head"},
        ],
        **optimizer_kwargs,
    )


def update_actor_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    actor_lr: float,
    backbone_lr: float,
) -> None:
    for param_group in optimizer.param_groups:
        if param_group.get("group_name") == "backbone":
            param_group["lr"] = float(backbone_lr)
        else:
            param_group["lr"] = float(actor_lr)


def update_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(learning_rate)


def prepare_env_actions(
    action_space: spaces.Space, action_tensor: torch.Tensor, num_envs: int
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(action_space, spaces.Discrete):
        stored_actions = action_tensor.detach().cpu().numpy().reshape((-1, 1))
        return stored_actions, stored_actions.reshape(-1)

    if isinstance(action_space, spaces.Box):
        action_shape = tuple(int(value) for value in action_space.shape)
        stored_actions = (
            action_tensor.detach()
            .cpu()
            .numpy()
            .reshape((num_envs, *action_shape))
            .astype(np.float32)
        )
        env_actions = np.clip(stored_actions, action_space.low, action_space.high)
        return stored_actions, env_actions

    raise TypeError(f"Unsupported action space: {type(action_space).__name__}")


def collect_candidate_rollout(
    policy: PolicySupernet,
    critic_model: PPO,
    env: VecEnv,
    arch: ArchConfig,
    rollout_buffer: DynamicsRolloutBuffer,
    initial_observation: np.ndarray,
    initial_episode_starts: np.ndarray,
    n_steps: int,
    gamma: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    policy.eval()
    critic_model.policy.eval()
    policy.set_active_arch(arch)
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
        for _ in range(int(n_steps)):
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
                gamma=float(gamma),
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
        / float(max(1, int(n_steps) * num_envs)),
        "rollout/ep_return": ep_return,
        "rollout/ep_length": ep_length,
        "rollout/done_count": float(rollout_done_count),
        "rollout/advantage_mean": float(rollout_buffer.advantages.mean()),
        "rollout/advantage_std": float(rollout_buffer.advantages.std()),
    }
    return observation, episode_starts, metrics


def fixed_arch_actor_update(
    policy: PolicySupernet,
    actor_optimizer: torch.optim.Optimizer,
    rollout_buffer: DynamicsRolloutBuffer,
    arch: ArchConfig,
    action_space: spaces.Space,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    normalize_advantage: bool,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    ema_policy: PolicySupernet | None = None,
    z_dyn_coef: float = 0.0,
) -> dict[str, float]:
    policy.train()
    use_dynamics = ema_policy is not None and z_dyn_coef > 0.0
    if use_dynamics:
        ema_policy.eval()

    loss_sum = 0.0
    policy_loss_sum = 0.0
    entropy_loss_sum = 0.0
    dynamics_loss_sum = 0.0
    approx_kl_sum = 0.0
    clip_fraction_sum = 0.0
    update_count = 0
    continue_training = True

    for _ in range(int(n_epochs)):
        if not continue_training:
            break
        for rollout_data in rollout_buffer.get(int(batch_size)):
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
            policy.set_active_arch(arch)

            if use_dynamics:
                features = policy.encode(rollout_data.observations)
                params = policy.distribution_params_from_features(features)
                new_log_probs, entropy = policy.evaluate_actions_from_params(
                    params, batch_actions
                )
            else:
                new_log_probs, entropy = policy.evaluate_actions(
                    rollout_data.observations, batch_actions
                )

            log_ratio = new_log_probs - batch_old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped_loss = batch_advantages * ratio
            clipped_loss = batch_advantages * torch.clamp(
                ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)
            )
            policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
            entropy_loss = -entropy.mean()

            if use_dynamics:
                dyn_loss = compute_dynamics_loss(
                    online_policy=policy,
                    ema_policy=ema_policy,
                    arch=arch,
                    start_features=features,
                    next_observations=rollout_data.next_observations,
                    actions=batch_actions,
                    action_space=action_space,
                    sample_weights=rollout_data.dynamics_masks,
                )
                loss = (
                    policy_loss
                    + float(ent_coef) * entropy_loss
                    + float(z_dyn_coef) * dyn_loss
                )
                dynamics_loss_value = float(dyn_loss.detach().cpu())
            else:
                loss = policy_loss + float(ent_coef) * entropy_loss
                dynamics_loss_value = 0.0

            approx_kl = torch.mean(((ratio - 1.0) - log_ratio).detach())
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1.0) > float(clip_range)).float().detach()
            )
            loss.backward()

            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in policy.parameters()
                        if parameter.requires_grad
                    ],
                    float(max_grad_norm),
                )
            actor_optimizer.step()

            update_count += 1
            loss_sum += float(loss.detach().cpu())
            policy_loss_sum += float(policy_loss.detach().cpu())
            entropy_loss_sum += float(entropy_loss.detach().cpu())
            dynamics_loss_sum += dynamics_loss_value
            approx_kl_sum += float(approx_kl.cpu())
            clip_fraction_sum += float(clip_fraction.cpu())

            if target_kl is not None and float(approx_kl.cpu()) > 1.5 * float(
                target_kl
            ):
                continue_training = False
                break

    denominator = float(max(1, update_count))
    metrics = {
        "actor/loss": loss_sum / denominator,
        "actor/policy_loss": policy_loss_sum / denominator,
        "actor/entropy_loss": entropy_loss_sum / denominator,
        "actor/approx_kl": approx_kl_sum / denominator,
        "actor/clip_fraction": clip_fraction_sum / denominator,
    }
    if use_dynamics:
        metrics["actor/dynamics_loss"] = dynamics_loss_sum / denominator
    return metrics




def evaluate_actor_subnet(
    policy: PolicySupernet,
    eval_env: VecEnv,
    arch: ArchConfig,
    n_eval_episodes: int,
    deterministic: bool,
    device: torch.device,
    *,
    train_env: VecEnv | None = None,
) -> dict[str, float]:
    """Evaluate a subnet on *eval_env* and return episode statistics.

    When *train_env* is provided, ``sync_envs_normalization`` is called
    automatically to copy observation/reward normalisation statistics from
    the training environment before evaluation.
    """
    if train_env is not None:
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
            _stored_actions, env_actions = prepare_env_actions(
                eval_env.action_space, actions, int(eval_env.num_envs)
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


def critic_warmup(
    policy: PolicySupernet,
    critic_model: PPO,
    env: VecEnv,
    arch: ArchConfig,
    rollout_buffer: DynamicsRolloutBuffer,
    initial_observation: np.ndarray,
    initial_episode_starts: np.ndarray,
    target_timesteps: int,
    ppo_config: Any,
    critic_lr_schedule: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]:
    observation = np.asarray(initial_observation)
    episode_starts = np.asarray(initial_episode_starts, dtype=np.bool_)
    actual_timesteps = 0
    last_metrics: dict[str, float] = {}
    target_timesteps = max(0, int(target_timesteps))

    while actual_timesteps < target_timesteps:
        progress_remaining = 1.0 - float(actual_timesteps) / float(
            max(1, target_timesteps)
        )
        critic_lr = (
            float(critic_lr_schedule(progress_remaining))
            if callable(critic_lr_schedule)
            else float(critic_lr_schedule)
        )
        update_optimizer_learning_rate(critic_model.policy.optimizer, critic_lr)
        observation, episode_starts, rollout_metrics = collect_candidate_rollout(
            policy=policy,
            critic_model=critic_model,
            env=env,
            arch=arch,
            rollout_buffer=rollout_buffer,
            initial_observation=observation,
            initial_episode_starts=episode_starts,
            n_steps=int(ppo_config.n_steps),
            gamma=float(ppo_config.gamma),
            device=device,
        )
        actual_timesteps += int(ppo_config.n_steps) * int(env.num_envs)
        critic_metrics = critic_update(
            critic_model=critic_model,
            optimizer=critic_model.policy.optimizer,
            rollout_buffer=rollout_buffer,
            n_epochs=int(ppo_config.n_epochs),
            batch_size=int(ppo_config.batch_size),
            max_grad_norm=float(ppo_config.max_grad_norm),
        )
        last_metrics = {
            "progress_remaining": float(progress_remaining),
            "critic_lr": float(critic_lr),
            **prefixed_metrics("critic_warmup_rollout", rollout_metrics),
            **prefixed_metrics("critic_warmup", critic_metrics),
        }

    return observation, episode_starts, actual_timesteps, last_metrics


def count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def create_ema_policy(
    policy: PolicySupernet,
    device: torch.device,
    checkpoint_ema_state_dict: dict[str, Any] | None = None,
) -> PolicySupernet:
    """Create an EMA copy of *policy*.

    If *checkpoint_ema_state_dict* is provided, the EMA model is initialised
    from those weights; otherwise it is a ``deepcopy`` of *policy*.
    """
    ema_policy = copy.deepcopy(policy).to(device)
    if checkpoint_ema_state_dict is not None:
        ema_policy.load_state_dict(checkpoint_ema_state_dict, strict=True)
    for parameter in ema_policy.parameters():
        parameter.requires_grad_(False)
    return ema_policy


def update_ema_model(
    ema_policy: PolicySupernet,
    online_policy: PolicySupernet,
    tau: float,
) -> None:
    """Exponential moving-average update: ``ema ← tau * ema + (1-tau) * online``."""
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
    """Normalised cosine-distance latent dynamics loss.

    ``online_policy`` predicts the next latent from *start_features* + *actions*;
    the target comes from ``ema_policy`` encoding *next_observations*.
    """
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

    return latent_dynamics_loss(
        predictions=predicted_next_latent,
        teacher_targets=target_next_latent,
        sample_weights=sample_weights,
    )
