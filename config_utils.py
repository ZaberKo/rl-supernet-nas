from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_PPO_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def add_ppo_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ppo_config",
        default=str(DEFAULT_PPO_CONFIG_PATH),
        help="Path to the shared env/PPO YAML config.",
    )
    parser.add_argument(
        "--ppo_config_override",
        action="append",
        default=[],
        help="OmegaConf dotlist override for env/PPO config, for example ppo.total_timesteps=10000.",
    )


def load_ppo_config(args: argparse.Namespace) -> DictConfig:
    config = OmegaConf.load(args.ppo_config)
    overrides = list(getattr(args, "ppo_config_override", []) or [])
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    ppo_config = OmegaConf.merge(config.get("env", {}), config.get("ppo", {}))
    values = OmegaConf.to_container(ppo_config, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("PPO config must contain mapping values.")
    return ppo_config


def ppo_config_to_dict(ppo_config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ppo_config, dict):
        return dict(ppo_config)
    values = OmegaConf.to_container(ppo_config, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("PPO config must contain mapping values.")
    return values


def build_run_config(
    args: argparse.Namespace, ppo_config: DictConfig
) -> dict[str, Any]:
    return {
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
