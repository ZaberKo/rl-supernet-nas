from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from config_utils import ppo_config_to_dict
from ppo_utils import parse_hidden_sizes, parse_optional_float, resolve_activation_fn
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


def build_network_ppo_config(
    checkpoint_ppo: Mapping[str, Any], runtime_ppo_config: Any, seed: int
) -> DictConfig:
    values = ppo_config_to_dict(runtime_ppo_config)
    for key in (
        "features_dim",
        "policy_net_arch",
        "value_net_arch",
        "activation_fn",
        "ortho_init",
        "log_std_init",
    ):
        if key in checkpoint_ppo:
            values[key] = checkpoint_ppo[key]
    values["seed"] = int(seed)
    return OmegaConf.create(values)


def validate_checkpoint_search_space(
    checkpoint: Mapping[str, Any], search_space: SearchSpace
) -> None:
    checkpoint_search_space = checkpoint.get("search_space")
    if checkpoint_search_space is None:
        return
    if checkpoint_search_space != search_space.to_dict():
        raise ValueError(
            "Checkpoint search_space does not match the current SearchSpace defaults."
        )


def build_policy_from_checkpoint(
    ppo_config: Any,
    train_env: VecEnv,
    search_space: SearchSpace,
    checkpoint: Mapping[str, Any],
    device: torch.device,
):
    from new_stage1_train_policy_supernet import PolicySupernet

    checkpoint_args = checkpoint.get("args", {})
    checkpoint_ppo = checkpoint.get("ppo_config", {})

    projection_dim = ppo_config.projection_dim
    if projection_dim <= 0:
        projection_dim = checkpoint.get(
            "projection_dim", checkpoint_args.get("projection_dim", 128)
        )

    predictor_hidden_dim = ppo_config.predictor_hidden_dim
    if predictor_hidden_dim <= 0:
        predictor_hidden_dim = checkpoint.get(
            "predictor_hidden_dim", checkpoint_args.get("predictor_hidden_dim", 512)
        )

    def get_ppo_val(key: str) -> Any:
        return checkpoint_ppo.get(key, getattr(ppo_config, key))

    policy = PolicySupernet(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        search_space=search_space,
        features_dim=int(checkpoint.get("features_dim", get_ppo_val("features_dim"))),
        policy_net_arch=parse_hidden_sizes(get_ppo_val("policy_net_arch")),
        activation_fn=resolve_activation_fn(get_ppo_val("activation_fn")),
        log_std_init=parse_optional_float(get_ppo_val("log_std_init")),
        ortho_init=bool(get_ppo_val("ortho_init")),
        projection_dim=int(projection_dim),
        predictor_hidden_dim=int(predictor_hidden_dim),
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
