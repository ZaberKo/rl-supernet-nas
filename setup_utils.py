from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

ScheduleValue = float | Callable[[float], float]


class LinearSchedule:
    def __init__(self, initial_value: float | str):
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        progress = min(1.0, max(0.0, float(progress_remaining)))
        return progress * self.initial_value

    def __repr__(self) -> str:
        return f"LinearSchedule(initial_value={self.initial_value})"


def parse_schedule_value(value: Any) -> ScheduleValue:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("lin_"):
            return LinearSchedule(text.removeprefix("lin_"))
        return float(text)
    return float(value)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_ppo_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ppo_config",
        default="config.yaml",
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


def prefixed_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def resolve_activation_fn(value: str | None) -> type[nn.Module] | None:
    if value is None:
        return None
    name = str(value).strip().lower()
    if name in {"", "none", "null"}:
        return None
    activations: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation_fn: {value}")
    return activations[name]


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        name = value.strip().lower()
        if name in {"", "none", "null", "off"}:
            return None
    return float(value)


# ---------------------------------------------------------------------------
# Ray worker resource scheduling
# ---------------------------------------------------------------------------

_GPU_EPS = 1e-5


@dataclass(frozen=True)
class RayWorkerConfig:
    """Computed Ray actor resource configuration."""

    num_workers: int
    """Total number of Ray actors to create."""

    num_gpus: int
    """Number of GPUs detected on the system."""

    workers_per_gpu: int
    """Maximum number of workers that may share a single GPU."""

    gpu_fraction: float
    """Per-actor ``num_gpus`` value for ``ray.remote()``."""

    def summary(self) -> str:
        return (
            f"workers={self.num_workers}, num_gpus={self.num_gpus}, "
            f"workers_per_gpu={self.workers_per_gpu}, "
            f"gpu_fraction={self.gpu_fraction:.6f}"
        )


def compute_ray_worker_config(num_workers: int) -> RayWorkerConfig:
    """Derive Ray actor GPU scheduling from a total worker count.

    * ``workers_per_gpu = ceil(num_workers / num_gpus)``
    * ``gpu_fraction = (1 - eps) / workers_per_gpu``

    The ``(1 - eps)`` ensures that ``workers_per_gpu`` actors fit on one GPU
    without floating-point over-subscription.
    """
    num_workers = max(1, int(num_workers))
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        workers_per_gpu = math.ceil(num_workers / num_gpus)
        gpu_fraction = (1.0 - _GPU_EPS) / workers_per_gpu
    else:
        workers_per_gpu = num_workers
        gpu_fraction = 0.0
    return RayWorkerConfig(
        num_workers=num_workers,
        num_gpus=num_gpus,
        workers_per_gpu=workers_per_gpu,
        gpu_fraction=gpu_fraction,
    )
