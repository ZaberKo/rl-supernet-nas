from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import KVWriter, filter_excluded_keys
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    VecEnv,
    sync_envs_normalization,
    unwrap_vec_normalize,
)

from env_utils import make_vec_env_from_ppo_config
from ppo_utils import jsonable_metric_value, normalize_metric_key
from setup_utils import parse_schedule_value, resolve_activation_fn
from supernet_backbone import (
    ArchConfig,
    SearchSpace,
    SupernetCNNBackbone,
    load_backbone_from_backbone_checkpoint,
)

ScheduleValue = float | Callable[[float], float]


class SupernetFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        arch_config: dict[str, Any] | ArchConfig | None = None,
        backbone_checkpoint_path: str | None = None,
        map_location: str | torch.device = "cpu",
    ):
        super().__init__(observation_space, features_dim)
        search_space = SearchSpace()
        arch = _resolve_arch_config(arch_config, search_space)
        self.backbone = SupernetCNNBackbone(
            input_channels=int(observation_space.shape[0]),
            search_space=search_space,
            feature_dim=features_dim,
        )
        self.backbone.set_sample_config(arch)
        if backbone_checkpoint_path is not None:
            load_backbone_from_backbone_checkpoint(
                self.backbone, backbone_checkpoint_path, map_location=map_location
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.backbone(observations)


def _resolve_arch_config(
    arch_config: dict[str, Any] | ArchConfig | None,
    search_space: SearchSpace,
) -> ArchConfig:
    if arch_config is None:
        return search_space.max_arch()
    if isinstance(arch_config, ArchConfig):
        return arch_config
    return ArchConfig.from_dict(arch_config)


def build_ppo_model(
    env: VecEnv,
    arch_config: ArchConfig,
    features_dim: int = 256,
    backbone_checkpoint_path: str | None = None,
    learning_rate: float | Callable[[float], float] = 3e-4,
    n_steps: int = 128,
    batch_size: int = 64,
    n_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float | Callable[[float], float] = 0.2,
    clip_range_vf: float | Callable[[float], float] | None = None,
    normalize_advantage: bool = True,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    stats_window_size: int = 100,
    tensorboard_log: str | None = None,
    policy_net_arch: Sequence[int] = (),
    value_net_arch: Sequence[int] = (),
    log_std_init: float | None = None,
    ortho_init: bool | None = None,
    activation_fn: type[nn.Module] | None = None,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    seed: int | None = None,
    device: str = "auto",
    verbose: int = 1,
) -> PPO:
    policy_kwargs = {
        "features_extractor_class": SupernetFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "arch_config": arch_config.to_dict(),
            "backbone_checkpoint_path": backbone_checkpoint_path,
        },
        "share_features_extractor": True,
        "net_arch": {"pi": list(policy_net_arch), "vf": list(value_net_arch)},
    }
    if log_std_init is not None:
        policy_kwargs["log_std_init"] = float(log_std_init)
    if ortho_init is not None:
        policy_kwargs["ortho_init"] = bool(ortho_init)
    if activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_fn
    return PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        normalize_advantage=normalize_advantage,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        stats_window_size=stats_window_size,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        seed=seed,
        device=device,
        verbose=verbose,
    )


def configure_policy_optimizer(
    model: PPO,
    head_lr: ScheduleValue,
    backbone_lr: float,
) -> None:
    policy = model.policy
    backbone = policy.features_extractor.backbone
    backbone_param_ids = {id(param) for param in backbone.parameters()}
    head_lr_start = float(head_lr(1.0)) if callable(head_lr) else float(head_lr)

    if backbone_lr <= 0.0:
        for param in backbone.parameters():
            param.requires_grad = False
        policy.optimizer = policy.optimizer_class(
            [param for param in policy.parameters() if param.requires_grad],
            lr=head_lr_start,
            **policy.optimizer_kwargs,
        )
        return

    for param in backbone.parameters():
        param.requires_grad = True
    backbone_params = list(backbone.parameters())
    head_params = [
        param
        for param in policy.parameters()
        if param.requires_grad and id(param) not in backbone_param_ids
    ]
    policy.optimizer = policy.optimizer_class(
        [
            {
                "params": backbone_params,
                "lr": float(backbone_lr),
                "group_name": "backbone",
            },
            {"params": head_params, "lr": head_lr_start, "group_name": "head"},
        ],
        **policy.optimizer_kwargs,
    )

    def update_group_learning_rates(self: PPO, optimizers: Any) -> None:
        head_lr_value = float(self.lr_schedule(self._current_progress_remaining))
        backbone_lr_value = float(backbone_lr)
        self.logger.record("train/learning_rate", head_lr_value)
        self.logger.record("train/backbone_learning_rate", backbone_lr_value)
        optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]
        for optimizer in optimizer_list:
            for param_group in optimizer.param_groups:
                if param_group.get("group_name") == "backbone":
                    param_group["lr"] = backbone_lr_value
                else:
                    param_group["lr"] = head_lr_value

    model._update_learning_rate = MethodType(update_group_learning_rates, model)


MetricLogFn = Callable[[dict[str, Any], int], None]

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

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        values = filter_excluded_keys(key_values, key_excluded, "json")
        metrics = {
            normalize_metric_key(key): jsonable_metric_value(value)
            for key, value in values.items()
        }
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


def build_ppo_model_from_config(
    ppo_config: Any,
    env: VecEnv,
    arch_config: ArchConfig,
    backbone_checkpoint_path: str | None = None,
    model_seed: int | None = None,
) -> PPO:
    return build_ppo_model(
        env=env,
        arch_config=arch_config,
        features_dim=ppo_config.features_dim,
        backbone_checkpoint_path=backbone_checkpoint_path,
        learning_rate=parse_schedule_value(ppo_config.policy_head_lr),
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=parse_schedule_value(ppo_config.clip_range),
        clip_range_vf=parse_schedule_value(ppo_config.clip_range_vf) if ppo_config.clip_range_vf is not None else None,
        normalize_advantage=ppo_config.normalize_advantage,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        target_kl=ppo_config.target_kl,
        stats_window_size=ppo_config.stats_window_size,
        tensorboard_log=ppo_config.tensorboard_log,
        policy_net_arch=list(ppo_config.policy_net_arch),
        value_net_arch=list(ppo_config.value_net_arch),
        log_std_init=ppo_config.log_std_init,
        ortho_init=ppo_config.ortho_init,
        activation_fn=resolve_activation_fn(ppo_config.activation_fn),
        use_sde=bool(ppo_config.use_sde),
        sde_sample_freq=int(ppo_config.sde_sample_freq),
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
            reset_num_timesteps=False,
            progress_bar=progress_bar,
        )


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
    train_env = make_vec_env_from_ppo_config(
        ppo_config, seed=train_seed, n_envs=ppo_config.train_n_envs
    )
    eval_env = make_vec_env_from_ppo_config(
        ppo_config, seed=eval_seed, n_envs=ppo_config.eval_n_envs
    )
    try:
        checkpoint_path = (
            args.supernet_checkpoint
            if Path(args.supernet_checkpoint).exists()
            else None
        )
        head_lr = parse_schedule_value(ppo_config.policy_head_lr)
        backbone_lr_arg = getattr(args, "supernet_backbone_lr", None)
        if backbone_lr_arg is None:
            backbone_lr_schedule = parse_schedule_value(ppo_config.policy_backbone_lr)
            backbone_lr = (
                float(backbone_lr_schedule(1.0))
                if callable(backbone_lr_schedule)
                else float(backbone_lr_schedule)
            )
        else:
            backbone_lr = float(backbone_lr_arg)
        candidate_timesteps = getattr(args, "candidate_timesteps", None)
        if candidate_timesteps is None:
            candidate_timesteps = ppo_config.total_timesteps
        model = build_ppo_model_from_config(
            ppo_config=ppo_config,
            env=train_env,
            arch_config=arch_config,
            backbone_checkpoint_path=checkpoint_path,
            model_seed=train_seed,
        )
        configure_policy_optimizer(
            model,
            head_lr=head_lr,
            backbone_lr=backbone_lr,
        )
        learn_ppo(
            model,
            total_timesteps=max(0, int(candidate_timesteps)),
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
