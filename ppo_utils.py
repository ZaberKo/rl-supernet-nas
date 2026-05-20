from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from env_utils import make_vision_vec_env
from sb3_nas_policy import build_ppo_model, configure_policy_optimizer
from supernet_backbone import ArchConfig


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


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def use_render_observation(ppo_config: Any) -> bool:
    if str(getattr(ppo_config, "atari_wrapper", "none")).lower() != "none":
        return False
    return not bool(ppo_config.native_image_env)


def get_train_n_envs(ppo_config: Any) -> int:
    return int(getattr(ppo_config, "train_n_envs", 1))


def get_eval_n_envs(ppo_config: Any) -> int:
    return int(getattr(ppo_config, "eval_n_envs", 1))


def make_vec_env_from_ppo_config(
    ppo_config: Any,
    seed: int | None = None,
    n_envs: int | None = None,
) -> VecEnv:
    return make_vision_vec_env(
        env_id=ppo_config.env_id,
        n_envs=get_train_n_envs(ppo_config) if n_envs is None else n_envs,
        seed=ppo_config.seed if seed is None else seed,
        image_size=ppo_config.image_size,
        use_render_observation=use_render_observation(ppo_config),
        vector_env_type=getattr(ppo_config, "vector_env_type", "dummy"),
        frame_stack=int(getattr(ppo_config, "frame_stack", 1)),
        atari_wrapper=str(getattr(ppo_config, "atari_wrapper", "none")),
    )


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
        learning_rate=float(getattr(ppo_config, learning_rate_attr)),
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        clip_range_vf=ppo_config.clip_range_vf,
        normalize_advantage=ppo_config.normalize_advantage,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        target_kl=ppo_config.target_kl,
        stats_window_size=ppo_config.stats_window_size,
        tensorboard_log=ppo_config.tensorboard_log,
        policy_net_arch=parse_hidden_sizes(getattr(ppo_config, "policy_net_arch", ())),
        value_net_arch=parse_hidden_sizes(getattr(ppo_config, "value_net_arch", ())),
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
            model.policy,
            head_lr=ppo_config.head_lr,
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
            n_eval_episodes=args.eval_episodes,
            deterministic=ppo_config.eval_deterministic,
        )
        return model, mean_reward, std_reward
    finally:
        train_env.close()
        eval_env.close()
