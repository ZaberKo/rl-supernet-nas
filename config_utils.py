from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_PPO_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def add_ppo_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ppo_config", default=str(DEFAULT_PPO_CONFIG_PATH), help="Path to the shared env/PPO YAML config.")
    parser.add_argument("--ppo_config_override", action="append", default=[], help="OmegaConf dotlist override for env/PPO config, for example ppo.total_timesteps=10000.")


def load_ppo_config(args: argparse.Namespace) -> argparse.Namespace:
    config = OmegaConf.load(args.ppo_config)
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
