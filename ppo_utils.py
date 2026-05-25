from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import KVWriter, filter_excluded_keys
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization, unwrap_vec_normalize

from env_utils import make_atari_vec_env, make_box2d_vec_env
from sb3_nas_policy import build_ppo_model, configure_policy_optimizer
from supernet_backbone import ArchConfig

ScheduleValue = float | Callable[[float], float]
MetricLogFn = Callable[[dict[str, Any], int], None]


class LinearSchedule:
    def __init__(self, initial_value: float | str):
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        progress = min(1.0, max(0.0, float(progress_remaining)))
        return progress * self.initial_value

    def __repr__(self) -> str:
        return f"LinearSchedule(initial_value={self.initial_value})"


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    return values


def parse_optional_float(text: str | None) -> float | None:
    if text is None:
        return None
    if str(text).lower() in {"none", "null", "off"}:
        return None
    return float(text)


def parse_schedule_value(value: Any) -> ScheduleValue:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("lin_"):
            return LinearSchedule(text.removeprefix("lin_"))
        return float(text)
    return float(value)


def parse_optional_schedule_value(value: Any) -> ScheduleValue | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "off"}:
        return None
    return parse_schedule_value(value)


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


def metric_log_step(metrics: Mapping[str, Any], fallback_step: int) -> int:
    value = metrics.get("total_timesteps", fallback_step)
    if isinstance(value, (int, float)):
        return int(value)
    return int(fallback_step)


class JsonlTrainingMetricsWriter(KVWriter):
    def __init__(self, path: Path, log_fn: MetricLogFn | None = None):
        self.path = path
        self.file = path.open("a")
        self.log_fn = log_fn

    def write(self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0) -> None:
        values = filter_excluded_keys(key_values, key_excluded, "json")
        metrics = {normalize_metric_key(key): jsonable_metric_value(value) for key, value in values.items()}
        if not metrics:
            return
        metrics.setdefault("total_timesteps", int(step))
        record = {"type": "train", **metrics}
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        if self.log_fn is not None:
            self.log_fn(metrics, metric_log_step(metrics, int(step)))

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, metrics_path: Path, log_fn: MetricLogFn | None = None):
        super().__init__()
        self.metrics_path = metrics_path
        self.log_fn = log_fn
        self.writer: JsonlTrainingMetricsWriter | None = None

    def _on_training_start(self) -> None:
        self.writer = JsonlTrainingMetricsWriter(self.metrics_path, self.log_fn)
        self.model.logger.output_formats.append(self.writer)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self.model.logger.name_to_value:
            self.model.logger.dump(self.num_timesteps)
        if self.writer is not None:
            self.writer.close()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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



def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "off"}:
        return None
    return int(value)



def make_vec_env_from_ppo_config(
    ppo_config: Any,
    seed: int | None = None,
    n_envs: int | None = None,
) -> VecEnv:
    common_kwargs = dict(
        env_id=ppo_config.env_id,
        n_envs=ppo_config.train_n_envs if n_envs is None else n_envs,
        seed=ppo_config.seed if seed is None else seed,
        image_size=ppo_config.image_size,
        vector_env_type=ppo_config.vector_env_type,
    )

    if getattr(ppo_config, "atari_wrapper", None) is not None:
        return make_atari_vec_env(
            **common_kwargs,
            **OmegaConf.to_container(ppo_config.atari_wrapper, resolve=True),
        )
    elif getattr(ppo_config, "box2d_wrapper", None) is not None:
        return make_box2d_vec_env(
            **common_kwargs,
            **OmegaConf.to_container(ppo_config.box2d_wrapper, resolve=True),
        )
    else:
        raise ValueError("Exactly one of env.atari_wrapper or env.box2d_wrapper must be configured.")


def get_vision_spaces_from_ppo_config(ppo_config: Any, seed: int | None = None) -> tuple[Any, Any]:
    env = make_vec_env_from_ppo_config(ppo_config, seed=seed, n_envs=1)
    try:
        return env.observation_space, env.action_space
    finally:
        env.close()


def parse_hidden_sizes(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return parse_int_tuple(value) if value else ()
    return tuple(int(item) for item in value)


def build_ppo_model_from_config(
    ppo_config: Any,
    env: VecEnv,
    arch_config: ArchConfig,
    backbone_checkpoint_path: str | None = None,
    learning_rate_attr: str = "learning_rate",
    model_seed: int | None = None,
) -> PPO:
    return build_ppo_model(
        env=env,
        arch_config=arch_config,
        features_dim=ppo_config.features_dim,
        backbone_checkpoint_path=backbone_checkpoint_path,
        learning_rate=parse_schedule_value(getattr(ppo_config, learning_rate_attr)),
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=parse_schedule_value(ppo_config.clip_range),
        clip_range_vf=parse_optional_schedule_value(ppo_config.clip_range_vf),
        normalize_advantage=ppo_config.normalize_advantage,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        target_kl=ppo_config.target_kl,
        stats_window_size=ppo_config.stats_window_size,
        tensorboard_log=ppo_config.tensorboard_log,
        policy_net_arch=parse_hidden_sizes(getattr(ppo_config, "policy_net_arch", ())),
        value_net_arch=parse_hidden_sizes(getattr(ppo_config, "value_net_arch", ())),
        log_std_init=parse_optional_float(getattr(ppo_config, "log_std_init", None)),
        ortho_init=getattr(ppo_config, "ortho_init", None),
        activation_fn=resolve_activation_fn(getattr(ppo_config, "activation_fn", None)),
        use_sde=bool(getattr(ppo_config, "use_sde", False)),
        sde_sample_freq=int(getattr(ppo_config, "sde_sample_freq", -1)),
        seed=ppo_config.seed if model_seed is None else model_seed,
        device=ppo_config.device,
        verbose=0 if ppo_config.quiet else 1,
    )


def learn_ppo(
    model: PPO,
    total_timesteps: int,
    callback: BaseCallback | None = None,
    run_id: str | None = None,
) -> None:
    if total_timesteps > 0:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )


def steps_for_transition_budget(transitions: int, n_envs: int) -> int:
    if transitions <= 0:
        return 0
    return int(math.ceil(transitions / max(1, n_envs)))


def evaluate_ppo_model(
    model: PPO,
    eval_env: VecEnv,
    n_eval_episodes: int,
    deterministic: bool,
) -> tuple[float, float]:
    train_env = model.get_env()
    if train_env is not None:
        sync_envs_normalization(train_env, eval_env)
    eval_vec_normalize = unwrap_vec_normalize(eval_env)
    if eval_vec_normalize is not None:
        eval_vec_normalize.training = False
        eval_vec_normalize.norm_reward = False
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        warn=False,
    )
    return float(mean_reward), float(std_reward)


def finetune_and_evaluate_arch(
    args: argparse.Namespace,
    ppo_config: Any,
    arch_config: ArchConfig,
    train_seed: int,
    eval_seed: int,
) -> tuple[PPO, float, float]:
    train_env = make_vec_env_from_ppo_config(ppo_config, seed=train_seed, n_envs=ppo_config.train_n_envs)
    eval_env = make_vec_env_from_ppo_config(ppo_config, seed=eval_seed, n_envs=ppo_config.eval_n_envs)
    try:
        checkpoint_path = args.supernet_checkpoint if Path(args.supernet_checkpoint).exists() else None
        model = build_ppo_model_from_config(
            ppo_config=ppo_config,
            env=train_env,
            arch_config=arch_config,
            backbone_checkpoint_path=checkpoint_path,
            learning_rate_attr="head_lr",
            model_seed=train_seed,
        )
        configure_policy_optimizer(
            model,
            head_lr=parse_schedule_value(ppo_config.head_lr),
            backbone_lr=args.supernet_backbone_lr,
        )
        learn_ppo(
            model,
            total_timesteps=args.candidate_timesteps,
            run_id=f"finetune_{train_seed}",
        )
        mean_reward, std_reward = evaluate_ppo_model(
            model,
            eval_env,
            n_eval_episodes=ppo_config.eval_episodes,
            deterministic=ppo_config.eval_deterministic,
        )
        return model, mean_reward, std_reward
    finally:
        train_env.close()
        eval_env.close()
