from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization, unwrap_vec_normalize

from env_utils import make_atari_vec_env, make_box2d_vec_env
from sb3_nas_policy import build_ppo_model, configure_policy_optimizer
from supernet_backbone import ArchConfig

ScheduleValue = float | Callable[[float], float]


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


def mapping_to_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not hasattr(value, "items"):
        raise TypeError(f"{name} must be a mapping.")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if hasattr(item, "items"):
            result[str(key)] = mapping_to_dict(item, f"{name}.{key}")
        else:
            result[str(key)] = item
    return result


def get_env_wrapper_config(ppo_config: Any) -> tuple[str, dict[str, Any]]:
    atari_wrapper = getattr(ppo_config, "atari_wrapper", None)
    box2d_wrapper = getattr(ppo_config, "box2d_wrapper", None)
    has_atari = atari_wrapper is not None
    has_box2d = box2d_wrapper is not None
    if has_atari == has_box2d:
        raise ValueError("Exactly one of env.atari_wrapper or env.box2d_wrapper must be configured.")
    if has_atari:
        return "atari", mapping_to_dict(atari_wrapper, "env.atari_wrapper")
    return "box2d", mapping_to_dict(box2d_wrapper, "env.box2d_wrapper")


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "off"}:
        return None
    return int(value)


def get_train_n_envs(ppo_config: Any) -> int:
    return int(getattr(ppo_config, "train_n_envs", 1))


def get_eval_n_envs(ppo_config: Any) -> int:
    return int(getattr(ppo_config, "eval_n_envs", 1))


def make_vec_env_from_ppo_config(
    ppo_config: Any,
    seed: int | None = None,
    n_envs: int | None = None,
) -> VecEnv:
    wrapper_kind, wrapper_config = get_env_wrapper_config(ppo_config)
    common_kwargs = dict(
        env_id=ppo_config.env_id,
        n_envs=get_train_n_envs(ppo_config) if n_envs is None else n_envs,
        seed=ppo_config.seed if seed is None else seed,
        image_size=int(getattr(ppo_config, "image_size", 64)),
        vector_env_type=getattr(ppo_config, "vector_env_type", "dummy"),
        frame_stack=int(wrapper_config.get("frame_stack", 1)),
        max_episode_steps=parse_optional_int(wrapper_config.get("max_episode_steps", None)),
        env_kwargs=mapping_to_dict(wrapper_config.get("env_kwargs", None), f"env.{wrapper_kind}_wrapper.env_kwargs"),
        normalize_observation=bool(wrapper_config.get("normalize_observation", False)),
        normalize_reward=bool(wrapper_config.get("normalize_reward", False)),
        normalize_clip_obs=float(wrapper_config.get("normalize_clip_obs", 10.0)),
        normalize_gamma=float(wrapper_config.get("normalize_gamma", getattr(ppo_config, "gamma", 0.99))),
    )
    if wrapper_kind == "atari":
        return make_atari_vec_env(**common_kwargs)
    return make_box2d_vec_env(
        **common_kwargs,
        frame_skip=int(wrapper_config.get("frame_skip", 1)),
        grayscale_observation=bool(wrapper_config.get("grayscale_observation", False)),
    )


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
    progress_bar: bool = False,
) -> None:
    if total_timesteps > 0:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
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
    train_env = make_vec_env_from_ppo_config(ppo_config, seed=train_seed, n_envs=get_train_n_envs(ppo_config))
    eval_env = make_vec_env_from_ppo_config(ppo_config, seed=eval_seed, n_envs=get_eval_n_envs(ppo_config))
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
            progress_bar=ppo_config.progress_bar,
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
