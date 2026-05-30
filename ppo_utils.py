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
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import NatureCNN
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


class PolicyBase(nn.Module):
    """Base class for policy networks with shared forward API.

    Subclasses must set the following attributes in ``__init__``:

    Attributes:
        backbone: CNN feature extractor module.
        policy_net: MLP applied after backbone.
        action_net: Linear head producing action logits or mean.
        log_std: Learnable log-std for continuous actions, or None.
        projection: Projection head for latent dynamics.
        predictor: Latent dynamics predictor.
        action_space: Gymnasium action space.
        action_kind: ``"discrete"`` or ``"box"``.
        action_dim: Number of action dimensions.
    """

    backbone: nn.Module
    policy_net: nn.Module
    action_net: nn.Module
    log_std: nn.Parameter | None
    projection: nn.Module
    predictor: nn.Module
    action_space: spaces.Space
    action_kind: str
    action_dim: int

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.backbone(observations))

    def distribution_params_from_features(
        self, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        action_output = self.action_net(features)
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

    def compute_dynamics_loss(
        self,
        ema_policy: PolicyBase,
        start_features: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalised cosine-distance latent dynamics loss.

        ``self`` (online policy) predicts the next latent from
        ``start_features`` + ``actions``; the target comes from
        ``ema_policy`` encoding ``next_observations``.

        Args:
            ema_policy: EMA target policy for computing target latents.
            start_features: Encoded features of current observations.
            next_observations: Raw next-step observations.
            actions: Actions taken at current step.
            sample_weights: Per-sample weights (dynamics mask); entries
                <= 0 are filtered out.

        Returns:
            Scalar loss tensor.
        """
        if sample_weights is not None:
            active_mask = sample_weights.to(device=start_features.device).view(-1) > 0.0
            if not bool(active_mask.any()):
                return torch.zeros((), dtype=torch.float32, device=start_features.device)
            start_features = start_features[active_mask]
            next_observations = next_observations[active_mask]
            actions = actions[active_mask]
            sample_weights = sample_weights.to(device=start_features.device)[active_mask]

        start_latent = self.project_features(start_features)
        action_features = encode_action_batch(actions, self.action_space).to(
            device=start_features.device
        )
        predicted_next_latent = start_latent + self.predictor(
            start_latent, action_features
        )
        with torch.no_grad():
            target_next_latent = ema_policy.project_observations(next_observations)

        return latent_dynamics_loss(
            predictions=predicted_next_latent,
            teacher_targets=target_next_latent,
            sample_weights=sample_weights,
        )


class PolicySupernet(PolicyBase):
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
        self.projection = ProjectionHead(self.latent_dim, projection_dim)
        self.predictor = LatentDynamicsPredictor(
            latent_dim=projection_dim,
            action_dim=get_action_dim(action_space),
            hidden_dim=predictor_hidden_dim,
        )

    def set_sample_config(self, arch: ArchConfig) -> None:
        self.backbone.set_sample_config(arch)

    def set_max_arch(self) -> None:
        self.backbone.set_max_arch()

    def set_min_arch(self) -> None:
        self.backbone.set_min_arch()

    def get_active_subnet(self) -> FixedPolicySubnet:
        """Extract an independent fixed-architecture subnet.

        Returns a ``FixedPolicySubnet`` with copies of the currently active
        weights.  The returned module can be trained independently without
        affecting the supernet.  Call ``set_sample_config`` before this method
        to configure which sub-network to extract.
        """
        return FixedPolicySubnet(
            backbone=self.backbone.get_active_subnet(),
            policy_net=copy.deepcopy(self.policy_net),
            action_net=copy.deepcopy(self.action_net),
            log_std=(
                nn.Parameter(self.log_std.data.clone())
                if self.log_std is not None
                else None
            ),
            projection=copy.deepcopy(self.projection),
            predictor=copy.deepcopy(self.predictor),
            action_space=self.action_space,
            arch_config=self.backbone.active_arch,
        )

    @property
    def elastic_num_params(self) -> int:
        total = int(self.backbone.elastic_num_params)
        total += sum(parameter.numel() for parameter in self.policy_net.parameters())
        total += sum(parameter.numel() for parameter in self.action_net.parameters())
        if self.log_std is not None:
            total += self.log_std.numel()
        if hasattr(self, "projection") and self.projection is not None:
            total += sum(parameter.numel() for parameter in self.projection.parameters())
        if hasattr(self, "predictor") and self.predictor is not None:
            total += sum(parameter.numel() for parameter in self.predictor.parameters())
        return total

    def policy_param_stats(self) -> dict[str, int]:
        """Compute parameter statistics for the **active** sub-network.

        Uses ``elastic_num_params`` for the backbone so that only the
        layers selected by the current ``set_sample_config`` are counted.

        Returns:
            Dict with keys:
                - ``policy_backbone_params``: active backbone + policy_net.
                - ``policy_head_params``: action_net + log_std.
                - ``policy_params``: total (backbone + head + projection
                  + predictor).
                - ``trainable_policy_params``: subset with
                  ``requires_grad=True``.
        """
        backbone_count = (
            int(self.backbone.elastic_num_params)
            + sum(p.numel() for p in self.policy_net.parameters())
        )
        head_params: list[torch.nn.Parameter] = list(self.action_net.parameters())
        if self.log_std is not None:
            head_params.append(self.log_std)
        head_count = sum(p.numel() for p in head_params)

        aux_params: list[torch.nn.Parameter] = []
        if hasattr(self, "projection") and self.projection is not None:
            aux_params += list(self.projection.parameters())
        if hasattr(self, "predictor") and self.predictor is not None:
            aux_params += list(self.predictor.parameters())
        aux_count = sum(p.numel() for p in aux_params)

        # For trainable count, backbone params must be enumerated explicitly
        # since elastic_num_params is a scalar, not a param list.
        all_real_params = (
            list(self.backbone.parameters())
            + list(self.policy_net.parameters())
            + head_params
            + aux_params
        )
        trainable_count = sum(
            p.numel() for p in all_real_params if p.requires_grad
        )

        return {
            "policy_backbone_params": backbone_count,
            "policy_head_params": head_count,
            "policy_params": backbone_count + head_count + aux_count,
            "trainable_policy_params": trainable_count,
        }


class FixedPolicySubnet(PolicyBase):
    """A fixed-architecture policy subnet extracted from a ``PolicySupernet``.

    This module has the same forward API as ``PolicySupernet`` but contains
    only the active sub-network weights (independent copies, not shared).
    It is produced by ``PolicySupernet.get_active_subnet()`` and is intended
    for independent fine-tuning without modifying the source supernet.
    """

    def __init__(
        self,
        backbone: nn.Module,
        policy_net: nn.Module,
        action_net: nn.Module,
        log_std: nn.Parameter | None,
        projection: nn.Module,
        predictor: nn.Module,
        action_space: spaces.Space,
        arch_config: ArchConfig,
    ):
        super().__init__()
        self.backbone = backbone
        self.policy_net = policy_net
        self.action_net = action_net
        if log_std is not None:
            self.log_std = log_std
        else:
            self.log_std = None
        self.projection = projection
        self.predictor = predictor
        self.action_space = action_space
        self.arch_config = arch_config

        if isinstance(action_space, spaces.Discrete):
            self.action_kind = "discrete"
            self.action_dim = int(action_space.n)
        else:
            self.action_kind = "box"
            self.action_shape = tuple(int(v) for v in action_space.shape)
            self.action_dim = int(np.prod(self.action_shape, dtype=np.int64))

    def policy_param_stats(self) -> dict[str, int]:
        """Compute parameter statistics for this fixed subnet.

        Returns:
            Dict with keys:
                - ``policy_backbone_params``: backbone + policy_net count.
                - ``policy_head_params``: action_net + log_std count.
                - ``policy_params``: total (backbone + head + projection
                  + predictor).
                - ``trainable_policy_params``: subset with
                  ``requires_grad=True``.
        """
        backbone_params = list(self.backbone.parameters()) + list(
            self.policy_net.parameters()
        )
        head_params: list[torch.nn.Parameter] = list(self.action_net.parameters())
        if self.log_std is not None:
            head_params.append(self.log_std)
        aux_params: list[torch.nn.Parameter] = []
        if hasattr(self, "projection") and self.projection is not None:
            aux_params += list(self.projection.parameters())
        if hasattr(self, "predictor") and self.predictor is not None:
            aux_params += list(self.predictor.parameters())

        backbone_count = sum(p.numel() for p in backbone_params)
        head_count = sum(p.numel() for p in head_params)
        aux_count = sum(p.numel() for p in aux_params)
        all_params = backbone_params + head_params + aux_params
        trainable_count = sum(p.numel() for p in all_params if p.requires_grad)

        return {
            "policy_backbone_params": backbone_count,
            "policy_head_params": head_count,
            "policy_params": backbone_count + head_count + aux_count,
            "trainable_policy_params": trainable_count,
        }


class _CriticMlp(nn.Module):
    """Container whose ``value_net`` attribute matches the
    ``mlp_extractor.value_net`` key prefix in SB3's ActorCriticPolicy
    state dict, enabling checkpoint compatibility."""

    def __init__(
        self,
        features_dim: int,
        value_net_arch: list[int],
        activation_fn: type[nn.Module],
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = features_dim
        for dim in value_net_arch:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(activation_fn())
            last_dim = dim
        self.value_net = nn.Sequential(*layers)
        self.latent_dim_vf = last_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class SB3CriticModel(nn.Module):
    """Standalone value-function model using SB3's NatureCNN backbone.

    Sub-module names (``features_extractor``, ``mlp_extractor.value_net``,
    ``value_net``) are chosen to match the critic keys of SB3's
    ``ActorCriticCnnPolicy`` state dict so that checkpoints saved by the
    old ``PPO``-based code can be loaded with ``strict=False``.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int,
        value_net_arch: list[int],
        activation_fn: type[nn.Module],
        ortho_init: bool,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.features_extractor = NatureCNN(
            observation_space, features_dim=features_dim,
        )
        self.mlp_extractor = _CriticMlp(
            features_dim=features_dim,
            value_net_arch=value_net_arch,
            activation_fn=activation_fn,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if ortho_init:
            self._ortho_init()

    def _ortho_init(self) -> None:
        """Apply orthogonal initialization (same convention as SB3)."""
        module_gains: dict[nn.Module, float] = {
            self.features_extractor: math.sqrt(2.0),
            self.mlp_extractor: math.sqrt(2.0),
            self.value_net: 1.0,
        }
        for module, gain in module_gains.items():
            for m in module.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.orthogonal_(m.weight, gain=gain)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        preprocessed = preprocess_obs(
            obs, self.observation_space, normalize_images=True,
        )
        features = self.features_extractor(preprocessed)
        latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)


def build_sb3_critic_model(
    ppo_config: DictConfig,
    env: VecEnv,
    device: str | torch.device | None = None,
) -> SB3CriticModel:
    activation_fn = resolve_activation_fn(ppo_config.activation_fn)
    if activation_fn is None:
        activation_fn = nn.Tanh
    ortho_init = ppo_config.ortho_init if ppo_config.ortho_init is not None else True

    if device is None:
        device = str(ppo_config.device)

    model = SB3CriticModel(
        observation_space=env.observation_space,
        features_dim=int(ppo_config.features_dim),
        value_net_arch=list(ppo_config.value_net_arch),
        activation_fn=activation_fn,
        ortho_init=bool(ortho_init),
    )
    return model.to(device)


def predict_critic_values(
    critic_model: SB3CriticModel, observations: torch.Tensor
) -> torch.Tensor:
    return critic_model.predict_values(observations).flatten()


def bootstrap_time_limit_rewards(
    rewards: np.ndarray,
    dones: np.ndarray,
    infos: list[dict[str, Any]],
    critic_model: SB3CriticModel,
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
    critic_model: SB3CriticModel,
    optimizer: torch.optim.Optimizer,
    rollout_buffer: DynamicsRolloutBuffer,
    n_epochs: int,
    batch_size: int,
    max_grad_norm: float,
) -> dict[str, float]:
    critic_model.train()
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
                    critic_model.parameters(), float(max_grad_norm)
                )
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            update_count += 1
    denominator = float(max(1, update_count))
    return {
        "critic/loss": loss_sum / denominator,
    }


def actor_head_parameters(policy: PolicyBase) -> list[torch.nn.Parameter]:
    parameters = list(policy.action_net.parameters())
    if hasattr(policy, "projection") and policy.projection is not None:
        parameters += list(policy.projection.parameters())
    if hasattr(policy, "predictor") and policy.predictor is not None:
        parameters += list(policy.predictor.parameters())
    if policy.log_std is not None:
        parameters.append(policy.log_std)
    return parameters


def configure_actor_optimizer(
    policy: PolicyBase,
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

    backbone_params = list(policy.backbone.parameters()) + list(
        policy.policy_net.parameters()
    )
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


def require_monitor_episode_info(info: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    episode_info = info.get("episode") if isinstance(info, Mapping) else None
    if (
        not isinstance(episode_info, Mapping)
        or "r" not in episode_info
        or "l" not in episode_info
    ):
        raise RuntimeError(
            f"Missing VecMonitor episode info during {context}. "
            "Wrap the VecEnv with stable_baselines3.common.vec_env.VecMonitor "
            "before recording episode returns."
        )
    return episode_info


def collect_candidate_rollout(
    policy: PolicyBase,
    critic_model: SB3CriticModel,
    env: VecEnv,
    rollout_buffer: DynamicsRolloutBuffer,
    initial_observation: np.ndarray,
    initial_episode_starts: np.ndarray,
    n_steps: int,
    gamma: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    policy.eval()
    critic_model.eval()
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
        / float(max(1, int(n_steps) * num_envs)),
        "rollout/ep_return": ep_return,
        "rollout/ep_length": ep_length,
        "rollout/done_count": float(rollout_done_count),
        "rollout/advantage_mean": float(rollout_buffer.advantages.mean()),
        "rollout/advantage_std": float(rollout_buffer.advantages.std()),
    }
    return observation, episode_starts, metrics


def fixed_arch_actor_update(
    policy: PolicyBase,
    actor_optimizer: torch.optim.Optimizer,
    rollout_buffer: DynamicsRolloutBuffer,
    action_space: spaces.Space,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    normalize_advantage: bool,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    ema_policy: PolicyBase | None = None,
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
                dyn_loss = policy.compute_dynamics_loss(
                    ema_policy=ema_policy,
                    start_features=features,
                    next_observations=rollout_data.next_observations,
                    actions=batch_actions,
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
    policy: PolicyBase,
    eval_env: VecEnv,
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
    observations = eval_env.reset()
    n_eval_episodes = int(n_eval_episodes)
    if n_eval_episodes <= 0:
        raise ValueError("n_eval_episodes must be positive.")
    n_envs = int(eval_env.num_envs)
    episode_counts = np.zeros(n_envs, dtype=np.int64)
    episode_count_targets = np.asarray(
        [(n_eval_episodes + env_index) // n_envs for env_index in range(n_envs)],
        dtype=np.int64,
    )
    episode_returns: list[float] = []
    episode_lengths: list[float] = []
    with torch.no_grad():
        while (episode_counts < episode_count_targets).any():
            observation_tensor = torch.as_tensor(observations, device=device)
            actions, _, _ = policy.act(observation_tensor, deterministic=deterministic)
            _stored_actions, env_actions = prepare_env_actions(
                eval_env.action_space, actions, int(eval_env.num_envs)
            )
            observations, rewards, dones, infos = eval_env.step(env_actions)
            for env_index, done in enumerate(dones):
                if episode_counts[env_index] >= episode_count_targets[env_index]:
                    continue
                if not bool(done):
                    continue
                episode_info = require_monitor_episode_info(
                    infos[env_index], "policy evaluation"
                )
                episode_returns.append(float(episode_info["r"]))
                episode_lengths.append(float(episode_info["l"]))
                episode_counts[env_index] += 1

    return {
        "ep_return": float(np.mean(episode_returns)),
        "ep_return_std": float(np.std(episode_returns)),
        "ep_length": float(np.mean(episode_lengths)),
        "ep_length_std": float(np.std(episode_lengths)),
    }


def critic_warmup(
    policy: PolicyBase,
    critic_model: SB3CriticModel,
    critic_optimizer: torch.optim.Optimizer,
    env: VecEnv,
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
        update_optimizer_learning_rate(critic_optimizer, critic_lr)
        observation, episode_starts, rollout_metrics = collect_candidate_rollout(
            policy=policy,
            critic_model=critic_model,
            env=env,
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
            optimizer=critic_optimizer,
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
    policy: PolicyBase,
    device: torch.device,
    checkpoint_ema_state_dict: dict[str, Any] | None = None,
) -> PolicyBase:
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
    ema_policy: PolicyBase,
    online_policy: PolicyBase,
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


