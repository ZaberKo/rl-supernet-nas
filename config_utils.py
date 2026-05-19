from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULT_PPO_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

DEFAULT_SHARED_CONFIG: dict[str, dict[str, Any]] = {
    "env": {
        "env_id": "CartPole-v1",
        "seed": 0,
        "image_size": 64,
        "native_image_env": False,
        "vector_env_type": "dummy",
    },
    "ppo": {
        "train_n_envs": 1,
        "eval_n_envs": 1,
        "eval_deterministic": True,
        "total_timesteps": 1024,
        "features_dim": 256,
        "learning_rate": 3e-4,
        "head_lr": 3e-4,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "stats_window_size": 100,
        "tensorboard_log": None,
        "policy_hidden_sizes": "",
        "device": "auto",
        "progress_bar": False,
        "quiet": False,
    },
}


def add_ppo_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ppo_config", default=str(DEFAULT_PPO_CONFIG_PATH), help="Path to the shared env/PPO YAML config.")
    parser.add_argument("--ppo_config_override", action="append", default=[], help="OmegaConf dotlist override for env/PPO config, for example ppo.total_timesteps=10000.")


def load_ppo_config(args: argparse.Namespace) -> argparse.Namespace:
    config = OmegaConf.merge(
        OmegaConf.create(DEFAULT_SHARED_CONFIG),
        OmegaConf.load(args.ppo_config),
    )
    overrides = list(getattr(args, "ppo_config_override", []) or [])
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    merged = OmegaConf.merge(config.get("env", {}), config.get("ppo", {}))
    values = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("PPO config must contain mapping values.")
    for key, value in values.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args
