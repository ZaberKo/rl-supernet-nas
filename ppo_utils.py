from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

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


def use_render_observation(args: argparse.Namespace) -> bool:
    return not bool(args.native_image_env)


def get_train_n_envs(args: argparse.Namespace) -> int:
    return int(getattr(args, "train_n_envs", 1))


def get_eval_n_envs(args: argparse.Namespace) -> int:
    return int(getattr(args, "eval_n_envs", 1))


def make_vec_env_from_args(
    args: argparse.Namespace,
    seed: int | None = None,
    n_envs: int | None = None,
) -> VecEnv:
    return make_vision_vec_env(
        env_id=args.env_id,
        n_envs=get_train_n_envs(args) if n_envs is None else n_envs,
        seed=args.seed if seed is None else seed,
        image_size=args.image_size,
        use_render_observation=use_render_observation(args),
        vector_env_type=getattr(args, "vector_env_type", "dummy"),
    )


def parse_policy_hidden_sizes(args: argparse.Namespace) -> tuple[int, ...]:
    text = getattr(args, "policy_hidden_sizes", "")
    return parse_int_tuple(text) if text else ()


def build_ppo_model_from_args(
    args: argparse.Namespace,
    env: VecEnv,
    arch_config: ArchConfig,
    backbone_checkpoint_path: str | None = None,
    learning_rate_attr: str = "learning_rate",
    model_seed: int | None = None,
) -> PPO:
    return build_ppo_model(
        env=env,
        arch_config=arch_config,
        features_dim=args.features_dim,
        backbone_checkpoint_path=backbone_checkpoint_path,
        learning_rate=float(getattr(args, learning_rate_attr)),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        normalize_advantage=args.normalize_advantage,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        stats_window_size=args.stats_window_size,
        tensorboard_log=args.tensorboard_log,
        policy_hidden_sizes=parse_policy_hidden_sizes(args),
        seed=args.seed if model_seed is None else model_seed,
        device=args.device,
        verbose=0 if args.quiet else 1,
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
    arch_config: ArchConfig,
    train_seed: int,
    eval_seed: int,
) -> tuple[PPO, float, float]:
    train_env = make_vec_env_from_args(args, seed=train_seed, n_envs=get_train_n_envs(args))
    eval_env = make_vec_env_from_args(args, seed=eval_seed, n_envs=get_eval_n_envs(args))
    try:
        checkpoint_path = args.supernet_checkpoint if Path(args.supernet_checkpoint).exists() else None
        model = build_ppo_model_from_args(
            args=args,
            env=train_env,
            arch_config=arch_config,
            backbone_checkpoint_path=checkpoint_path,
            learning_rate_attr="head_lr",
            model_seed=train_seed,
        )
        configure_policy_optimizer(
            model.policy,
            head_lr=args.head_lr,
            backbone_lr=args.supernet_backbone_lr,
        )
        learn_ppo(
            model,
            total_timesteps=args.candidate_timesteps,
            progress_bar=args.progress_bar,
        )
        mean_reward, std_reward = evaluate_ppo_model(
            model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=args.eval_deterministic,
        )
        return model, mean_reward, std_reward
    finally:
        train_env.close()
        eval_env.close()
