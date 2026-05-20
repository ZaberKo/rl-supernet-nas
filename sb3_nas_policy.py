from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv

from supernet_backbone import ArchConfig, SearchSpace, SupernetCNNBackbone, infer_input_channels


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
            payload = torch.load(Path(backbone_checkpoint_path), map_location=map_location)
            self.backbone.load_state_dict(payload.get("backbone_state_dict", payload), strict=False)

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
    learning_rate: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 64,
    n_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    clip_range_vf: float | None = None,
    normalize_advantage: bool = True,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    stats_window_size: int = 100,
    tensorboard_log: str | None = None,
    policy_net_arch: Sequence[int] = (),
    value_net_arch: Sequence[int] = (),
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
        seed=seed,
        device=device,
        verbose=verbose,
    )


def configure_policy_optimizer(
    policy: ActorCriticPolicy,
    head_lr: float,
    backbone_lr: float,
) -> None:
    backbone = policy.features_extractor.backbone
    backbone_param_ids = {id(param) for param in backbone.parameters()}

    if backbone_lr <= 0.0:
        for param in backbone.parameters():
            param.requires_grad = False
        policy.optimizer = policy.optimizer_class(
            [param for param in policy.parameters() if param.requires_grad],
            lr=head_lr,
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
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        **policy.optimizer_kwargs,
    )


def save_policy_backbone(
    policy: ActorCriticPolicy,
    path: str | Path,
    search_space: SearchSpace,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "backbone_state_dict": policy.features_extractor.backbone.state_dict(),
        "search_space": search_space.to_dict(),
        "active_arch": policy.features_extractor.backbone.active_arch.to_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path))
