from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv

from supernet_backbone import (
    ArchConfig,
    SearchSpace,
    SupernetCNNBackbone,
    infer_input_channels,
    load_backbone_from_backbone_checkpoint,
)

ScheduleValue = float | Callable[[float], float]


class SupernetFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        arch_config: dict[str, Any] | ArchConfig | None = None,
        backbone_checkpoint_path: str | None = None,
        map_location: str | torch.device = "cpu",
    ):
        super().__init__(observation_space, features_dim)
        search_space = SearchSpace()
        arch = _resolve_arch_config(arch_config, search_space)
        self.backbone = SupernetCNNBackbone(
            input_channels=infer_input_channels(tuple(observation_space.shape)),
            search_space=search_space,
            feature_dim=features_dim,
        )
        self.backbone.set_sample_config(arch)
        if backbone_checkpoint_path is not None:
            load_backbone_from_backbone_checkpoint(
                self.backbone, backbone_checkpoint_path, map_location=map_location
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.backbone(observations)


def _resolve_arch_config(
    arch_config: dict[str, Any] | ArchConfig | None,
    search_space: SearchSpace,
) -> ArchConfig:
    if arch_config is None:
        return search_space.max_arch()
    if isinstance(arch_config, ArchConfig):
        return arch_config
    return ArchConfig.from_dict(arch_config)


def build_ppo_model(
    env: VecEnv,
    arch_config: ArchConfig,
    features_dim: int = 256,
    backbone_checkpoint_path: str | None = None,
    learning_rate: float | Callable[[float], float] = 3e-4,
    n_steps: int = 128,
    batch_size: int = 64,
    n_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float | Callable[[float], float] = 0.2,
    clip_range_vf: float | Callable[[float], float] | None = None,
    normalize_advantage: bool = True,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    stats_window_size: int = 100,
    tensorboard_log: str | None = None,
    policy_net_arch: Sequence[int] = (),
    value_net_arch: Sequence[int] = (),
    log_std_init: float | None = None,
    ortho_init: bool | None = None,
    activation_fn: type[nn.Module] | None = None,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    seed: int | None = None,
    device: str = "auto",
    verbose: int = 1,
) -> PPO:
    policy_kwargs = {
        "features_extractor_class": SupernetFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "arch_config": arch_config.to_dict(),
            "backbone_checkpoint_path": backbone_checkpoint_path,
        },
        "share_features_extractor": True,
        "net_arch": {"pi": list(policy_net_arch), "vf": list(value_net_arch)},
    }
    if log_std_init is not None:
        policy_kwargs["log_std_init"] = float(log_std_init)
    if ortho_init is not None:
        policy_kwargs["ortho_init"] = bool(ortho_init)
    if activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_fn
    return PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        normalize_advantage=normalize_advantage,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        stats_window_size=stats_window_size,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        seed=seed,
        device=device,
        verbose=verbose,
    )


def configure_policy_optimizer(
    model: PPO,
    head_lr: ScheduleValue,
    backbone_lr: float,
) -> None:
    policy = model.policy
    backbone = policy.features_extractor.backbone
    backbone_param_ids = {id(param) for param in backbone.parameters()}
    head_lr_start = float(head_lr(1.0)) if callable(head_lr) else float(head_lr)

    if backbone_lr <= 0.0:
        for param in backbone.parameters():
            param.requires_grad = False
        policy.optimizer = policy.optimizer_class(
            [param for param in policy.parameters() if param.requires_grad],
            lr=head_lr_start,
            **policy.optimizer_kwargs,
        )
        return

    for param in backbone.parameters():
        param.requires_grad = True
    backbone_params = list(backbone.parameters())
    head_params = [
        param
        for param in policy.parameters()
        if param.requires_grad and id(param) not in backbone_param_ids
    ]
    policy.optimizer = policy.optimizer_class(
        [
            {
                "params": backbone_params,
                "lr": float(backbone_lr),
                "group_name": "backbone",
            },
            {"params": head_params, "lr": head_lr_start, "group_name": "head"},
        ],
        **policy.optimizer_kwargs,
    )

    def update_group_learning_rates(self: PPO, optimizers: Any) -> None:
        head_lr_value = float(self.lr_schedule(self._current_progress_remaining))
        backbone_lr_value = float(backbone_lr)
        self.logger.record("train/learning_rate", head_lr_value)
        self.logger.record("train/backbone_learning_rate", backbone_lr_value)
        optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]
        for optimizer in optimizer_list:
            for param_group in optimizer.param_groups:
                if param_group.get("group_name") == "backbone":
                    param_group["lr"] = backbone_lr_value
                else:
                    param_group["lr"] = head_lr_value

    model._update_learning_rate = MethodType(update_group_learning_rates, model)


def save_policy_backbone(
    policy: ActorCriticPolicy,
    path: str | Path,
    search_space: SearchSpace,
    extra: dict[str, Any] | None = None,
) -> None:
    state_dict: dict[str, Any] = {
        "backbone_state_dict": policy.features_extractor.backbone.state_dict(),
        "search_space": search_space.to_dict(),
        "active_arch": policy.features_extractor.backbone.active_arch.to_dict(),
    }
    if extra:
        state_dict.update(extra)
    torch.save(state_dict, Path(path))


def save_ppo_supernet_checkpoint(
    model: PPO,
    path: str | Path,
    search_space: SearchSpace,
    extra: dict[str, Any] | None = None,
) -> None:
    policy = model.policy
    state_dict: dict[str, Any] = {
        "policy_state_dict": policy.state_dict(),
        "search_space": search_space.to_dict(),
        "active_arch": policy.features_extractor.backbone.active_arch.to_dict(),
        "policy_class": policy.__class__.__name__,
        "features_extractor_class": policy.features_extractor.__class__.__name__,
        "num_timesteps": int(model.num_timesteps),
    }
    if extra:
        state_dict.update(extra)
    torch.save(state_dict, Path(path))
