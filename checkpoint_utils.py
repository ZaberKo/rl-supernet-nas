import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from ppo_utils import (
    PolicySupernet,
    resolve_activation_fn,
)
from setup_utils import ppo_config_to_dict
from supernet_backbone import SearchSpace


def load_checkpoint(
    path: str | Path, map_location: str | torch.device
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint data must be a mapping.")
    return dict(checkpoint)


def save_checkpoint(
    path: Path,
    *,
    stage: str,
    args: argparse.Namespace,
    ppo_config: DictConfig,
    policy_state_dict: dict[str, Any],
    critic_policy_state_dict: dict[str, Any],
    actor_optimizer_state_dict: dict[str, Any],
    critic_optimizer_state_dict: dict[str, Any],
    **extra: Any,
) -> None:
    """Unified checkpoint save for all stages.

    Common fields are explicit parameters; stage-specific fields go in ``**extra``.
    """
    torch.save(
        {
            "stage": stage,
            "policy_state_dict": policy_state_dict,
            "critic_policy_state_dict": critic_policy_state_dict,
            "actor_optimizer_state_dict": actor_optimizer_state_dict,
            "critic_optimizer_state_dict": critic_optimizer_state_dict,
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
            **extra,
        },
        path,
    )


def build_policy_from_checkpoint(
    ppo_config: Any,
    env: VecEnv,
    search_space: SearchSpace,
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> PolicySupernet:
    """Build a PolicySupernet from ``ppo_config`` and load weights from *checkpoint*.

    All network hyper-parameters are read directly from ``ppo_config``;
    the checkpoint only supplies the ``policy_state_dict``.
    """
    policy = PolicySupernet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        search_space=search_space,
        features_dim=int(ppo_config.features_dim),
        policy_net_arch=list(ppo_config.policy_net_arch or []),
        activation_fn=resolve_activation_fn(ppo_config.activation_fn),
        log_std_init=ppo_config.log_std_init,
        ortho_init=bool(ppo_config.ortho_init),
        projection_dim=int(ppo_config.projection_dim),
        predictor_hidden_dim=int(ppo_config.predictor_hidden_dim),
    ).to(device)

    state_dict = checkpoint.get("policy_state_dict")
    if not isinstance(state_dict, Mapping):
        raise KeyError("Checkpoint does not contain policy_state_dict.")
    policy.load_state_dict(state_dict, strict=True)
    return policy


def load_critic_from_checkpoint(
    critic_model: PPO,
    checkpoint: Mapping[str, Any],
) -> bool:
    state_dict = checkpoint.get("critic_policy_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain critic_policy_state_dict.")
    if not isinstance(state_dict, Mapping):
        raise TypeError("critic_policy_state_dict must be a mapping.")
    critic_model.policy.load_state_dict(state_dict, strict=True)
    return True
