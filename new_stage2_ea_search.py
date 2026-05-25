from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.multiprocessing as mp
from evox.core import Problem
from evox.operators.selection import non_dominate_rank
from evox.workflows import EvalMonitor, StdWorkflow
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from config_utils import add_ppo_config_args, build_run_config, load_ppo_config, ppo_config_to_dict
from ea_codec import GeneCodec
from new_stage1_train_policy_supernet import (
    DynamicsRolloutBuffer,
    PolicySupernet,
    bootstrap_time_limit_rewards,
    build_sb3_critic_model,
    critic_update,
    evaluate_actor_subnet,
    predict_critic_values,
)
from nsga2_search import DiscreteNSGA2
from ppo_utils import (
    get_eval_n_envs,
    get_train_n_envs,
    make_vec_env_from_ppo_config,
    parse_hidden_sizes,
    parse_optional_float,
    parse_schedule_value,
    resolve_activation_fn,
    resolve_device,
)
from supernet_backbone import ArchConfig, SearchSpace
from trajectory_data import resolve_terminal_next_observations, split_done_flags
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="New stage 2 NSGA-II subnet search initialized from a policy supernet.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--output_dir", default="runs/new_stage2_ea_search", help="Directory for NSGA-II records, search space, and manifest.")
    parser.add_argument("--supernet_checkpoint", default="runs/new_stage1_policy_supernet/policy_supernet_best.pt", help="New stage1 policy-supernet checkpoint used to initialize subnet candidates.")
    parser.add_argument("--population_size", type=int, default=6, help="NSGA-II population size.")
    parser.add_argument("--generations", type=int, default=3, help="Number of NSGA-II generations to evaluate.")
    parser.add_argument("--candidate_timesteps", type=int, default=1024, help="PPO finetune timesteps for each subnet candidate.")
    parser.add_argument("--eval_workers", type=int, default=1, help="Torch multiprocessing workers for parallel subnet evaluation.")
    parser.add_argument("--supernet_backbone_lr", type=float, default=0.0, help="Backbone learning rate during candidate PPO finetune; <=0 freezes the inherited backbone.")
    parser.add_argument("--critic_learning_rate", default="", help="Critic learning rate schedule. Empty reuses ppo.learning_rate.")
    parser.add_argument("--critic_warmup_timesteps", type=int, default=0, help="Critic-only warmup timesteps on current subnet rollouts before actor PPO finetune; 0 disables warmup.")
    parser.add_argument("--projection_dim", type=int, default=0, help="Policy projection dimension. 0 reads the value from the checkpoint.")
    parser.add_argument("--predictor_hidden_dim", type=int, default=0, help="Dynamics predictor hidden dimension. 0 reads the value from the checkpoint args.")
    parser.add_argument("--save_full_history", action="store_true", help="Store full EvoX monitor history in memory for debugging.")
    args = parser.parse_args()
    if args.critic_warmup_timesteps < 0:
        raise ValueError("critic_warmup_timesteps must be non-negative.")
    args.eval_call_seed_stride = 10_000
    args.candidate_seed_stride = 100
    args.eval_seed_offset = 50
    args.mp_start_method = "spawn"
    args.worker_torch_threads = 1
    return args


def build_initial_population(args: argparse.Namespace, codec: GeneCodec) -> list[list[int]]:
    if args.population_size <= 0:
        raise ValueError("population_size must be positive.")
    population: list[list[int]] = [codec.max_gene()]
    while len(population) < args.population_size:
        population.append(codec.sample_gene())
    return population[: args.population_size]


def tensor_to_genes(pop: torch.Tensor) -> list[list[int]]:
    return [[int(round(value)) for value in row.tolist()] for row in pop.detach().cpu()]


def build_generation_records(
    generation: int,
    pop: torch.Tensor,
    fit: torch.Tensor,
    codec: GeneCodec,
    problem: "NewPolicySubnetProblem",
) -> list[dict[str, Any]]:
    genes = tensor_to_genes(pop)
    fit_cpu = fit.detach().cpu()
    rank = non_dominate_rank(fit_cpu)
    records = []
    worker_records = {tuple(record["gene"]): record for record in problem.last_records}
    for index, gene in enumerate(genes):
        worker_record = worker_records.get(tuple(gene), {})
        objectives = [float(value) for value in fit_cpu[index].tolist()]
        pareto_rank = int(rank[index].item())
        records.append(
            {
                "gen": generation,
                "generation": generation,
                "individual_index": index,
                "candidate_index": index,
                "gene": gene,
                "arch": codec.gene_to_arch(gene).to_dict(),
                "objectives": {
                    "negative_return": objectives[0],
                    "params": objectives[1],
                },
                "return": -objectives[0],
                "params": objectives[1],
                "pareto_rank": pareto_rank,
                "is_pareto": bool(pareto_rank == 0),
                "is_pareto_front": bool(pareto_rank == 0),
                "worker_record": worker_record,
            }
        )
    return records


def write_generation(records_path: Path, records: list[dict[str, Any]]) -> None:
    with records_path.open("a") as records_file:
        for record in records:
            records_file.write(json.dumps(record) + "\n")


def generation_summary(generation: int, records: list[dict[str, Any]], cache_hits: int) -> dict[str, float | int]:
    if not records:
        return {
            "gen": generation,
            "candidates": 0,
            "pareto": 0,
            "best_return": 0.0,
            "min_params": 0.0,
            "cache_hits": cache_hits,
        }
    return {
        "gen": generation,
        "candidates": len(records),
        "pareto": sum(1 for record in records if bool(record["is_pareto"])),
        "best_return": max(float(record["return"]) for record in records),
        "min_params": min(float(record["params"]) for record in records),
        "cache_hits": cache_hits,
    }


def format_generation_log(generation: int, records: list[dict[str, Any]], cache_hits: int) -> str:
    summary = generation_summary(generation, records, cache_hits)
    return (
        f"gen={int(summary['gen'])} candidates={int(summary['candidates'])} "
        f"pareto={int(summary['pareto'])} best_return={float(summary['best_return']):.6g} "
        f"min_params={float(summary['min_params']):.0f} cache_hits={int(summary['cache_hits'])}"
    )


def log_generation(
    log_path: Path,
    generation: int,
    records: list[dict[str, Any]],
    cache_hits: int,
    wandb_run: Any = None,
) -> None:
    message = format_generation_log(generation, records, cache_hits)
    print(message, flush=True)
    with log_path.open("a") as log_file:
        log_file.write(message + "\n")
    summary = generation_summary(generation, records, cache_hits)
    log_wandb(wandb_run, summary, step=generation)


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Policy-supernet checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("Policy-supernet checkpoint payload must be a mapping.")
    return dict(payload)


def checkpoint_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        return {}
    return dict(value)


def checkpoint_arg(payload: Mapping[str, Any], key: str, default: Any) -> Any:
    args = checkpoint_mapping(payload, "args")
    return args.get(key, default)


def checkpoint_ppo_value(payload: Mapping[str, Any], runtime_ppo_config: Any, key: str, default: Any = None) -> Any:
    checkpoint_ppo_config = checkpoint_mapping(payload, "ppo_config")
    if key in checkpoint_ppo_config:
        return checkpoint_ppo_config[key]
    return getattr(runtime_ppo_config, key, default)


def build_network_ppo_config(payload: Mapping[str, Any], runtime_ppo_config: Any, seed: int) -> DictConfig:
    values = ppo_config_to_dict(runtime_ppo_config)
    checkpoint_ppo_config = checkpoint_mapping(payload, "ppo_config")
    for key in (
        "features_dim",
        "policy_net_arch",
        "value_net_arch",
        "activation_fn",
        "ortho_init",
        "log_std_init",
    ):
        if key in checkpoint_ppo_config:
            values[key] = checkpoint_ppo_config[key]
    values["seed"] = int(seed)
    return OmegaConf.create(values)


def resolve_policy_constructor_values(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    ppo_config: Any,
) -> dict[str, Any]:
    projection_dim = int(args.projection_dim)
    if projection_dim <= 0:
        projection_dim = int(payload.get("projection_dim", checkpoint_arg(payload, "projection_dim", 128)))

    predictor_hidden_dim = int(args.predictor_hidden_dim)
    if predictor_hidden_dim <= 0:
        predictor_hidden_dim = int(checkpoint_arg(payload, "predictor_hidden_dim", 512))

    return {
        "features_dim": int(payload.get("features_dim", checkpoint_ppo_value(payload, ppo_config, "features_dim"))),
        "policy_net_arch": parse_hidden_sizes(checkpoint_ppo_value(payload, ppo_config, "policy_net_arch", ())),
        "activation_fn": resolve_activation_fn(checkpoint_ppo_value(payload, ppo_config, "activation_fn", None)),
        "log_std_init": parse_optional_float(checkpoint_ppo_value(payload, ppo_config, "log_std_init", None)),
        "ortho_init": bool(checkpoint_ppo_value(payload, ppo_config, "ortho_init", False)),
        "projection_dim": projection_dim,
        "predictor_hidden_dim": predictor_hidden_dim,
    }


def validate_checkpoint_search_space(payload: Mapping[str, Any], search_space: SearchSpace) -> None:
    checkpoint_search_space = payload.get("search_space")
    if checkpoint_search_space is None:
        return
    if checkpoint_search_space != search_space.to_dict():
        raise ValueError("Checkpoint search_space does not match the current SearchSpace defaults.")


def build_policy_from_checkpoint(
    args: argparse.Namespace,
    ppo_config: Any,
    train_env: VecEnv,
    search_space: SearchSpace,
    checkpoint_payload: Mapping[str, Any],
    device: torch.device,
) -> PolicySupernet:
    constructor_values = resolve_policy_constructor_values(checkpoint_payload, args, ppo_config)
    policy = PolicySupernet(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        search_space=search_space,
        features_dim=int(constructor_values["features_dim"]),
        policy_net_arch=constructor_values["policy_net_arch"],
        activation_fn=constructor_values["activation_fn"],
        log_std_init=constructor_values["log_std_init"],
        ortho_init=bool(constructor_values["ortho_init"]),
        projection_dim=int(constructor_values["projection_dim"]),
        predictor_hidden_dim=int(constructor_values["predictor_hidden_dim"]),
    ).to(device)

    state_dict = checkpoint_payload.get("policy_state_dict")
    if not isinstance(state_dict, Mapping):
        raise KeyError("Checkpoint does not contain policy_state_dict.")
    policy.load_state_dict(state_dict, strict=True)
    return policy


def load_critic_from_checkpoint(
    critic_model: PPO,
    checkpoint_payload: Mapping[str, Any],
) -> bool:
    state_dict = checkpoint_payload.get("critic_policy_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain critic_policy_state_dict.")
    if not isinstance(state_dict, Mapping):
        raise TypeError("critic_policy_state_dict must be a mapping.")
    critic_model.policy.load_state_dict(state_dict, strict=True)
    return True


def actor_head_parameters(policy: PolicySupernet) -> list[torch.nn.Parameter]:
    parameters = list(policy.policy_net.parameters()) + list(policy.action_net.parameters())
    if policy.log_std is not None:
        parameters.append(policy.log_std)
    return parameters


def configure_actor_optimizer(
    policy: PolicySupernet,
    actor_lr: Any,
    backbone_lr: float,
) -> torch.optim.Optimizer:
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    head_params = actor_head_parameters(policy)
    for parameter in head_params:
        parameter.requires_grad_(True)

    actor_lr_start = float(actor_lr(1.0)) if callable(actor_lr) else float(actor_lr)
    optimizer_kwargs = {
        "betas": (0.9, 0.999),
        "eps": 1e-5,
        "weight_decay": 0.0,
    }

    if float(backbone_lr) <= 0.0:
        return torch.optim.Adam(head_params, lr=actor_lr_start, **optimizer_kwargs)

    backbone_params = list(policy.backbone.parameters())
    for parameter in backbone_params:
        parameter.requires_grad_(True)

    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": float(backbone_lr), "group_name": "backbone"},
            {"params": head_params, "lr": actor_lr_start, "group_name": "head"},
        ],
        **optimizer_kwargs,
    )


def update_actor_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    actor_lr: float,
    backbone_lr: float,
) -> None:
    for param_group in optimizer.param_groups:
        if param_group.get("group_name") == "backbone":
            param_group["lr"] = float(backbone_lr)
        else:
            param_group["lr"] = float(actor_lr)


def update_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(learning_rate)


def prepare_env_actions(action_space: spaces.Space, action_tensor: torch.Tensor, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(action_space, spaces.Discrete):
        stored_actions = action_tensor.detach().cpu().numpy().reshape((-1, 1))
        return stored_actions, stored_actions.reshape(-1)

    if isinstance(action_space, spaces.Box):
        action_shape = tuple(int(value) for value in action_space.shape)
        stored_actions = action_tensor.detach().cpu().numpy().reshape((num_envs, *action_shape)).astype(np.float32)
        env_actions = np.clip(stored_actions, action_space.low, action_space.high)
        return stored_actions, env_actions

    raise TypeError(f"Unsupported action space: {type(action_space).__name__}")


def collect_candidate_rollout(
    policy: PolicySupernet,
    critic_model: PPO,
    env: VecEnv,
    arch: ArchConfig,
    rollout_buffer: DynamicsRolloutBuffer,
    initial_observation: np.ndarray,
    initial_episode_starts: np.ndarray,
    n_steps: int,
    gamma: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    policy.eval()
    critic_model.policy.eval()
    policy.set_active_arch(arch)
    rollout_buffer.reset()

    num_envs = int(env.num_envs)
    observation = np.asarray(initial_observation)
    episode_starts = np.asarray(initial_episode_starts, dtype=np.bool_)
    rollout_reward_sum = 0.0
    rollout_done_count = 0
    rollout_episode_returns: list[float] = []
    rollout_episode_lengths: list[float] = []
    current_rollout_returns = np.zeros(num_envs, dtype=np.float64)
    current_rollout_lengths = np.zeros(num_envs, dtype=np.int64)
    last_dones = episode_starts

    with torch.no_grad():
        for _ in range(int(n_steps)):
            observation_tensor = torch.as_tensor(observation, device=device)
            action_tensor, log_prob_tensor, _ = policy.act(observation_tensor, deterministic=False)
            value_tensor = predict_critic_values(critic_model, observation_tensor)
            stored_actions, env_actions = prepare_env_actions(env.action_space, action_tensor, num_envs)

            next_observation, raw_rewards, raw_dones, infos = env.step(env_actions)
            info_list = list(infos)
            resolved_next_observation = resolve_terminal_next_observations(next_observation, info_list)
            terminated, _truncated = split_done_flags(raw_dones, info_list)
            dynamics_mask = (~terminated).astype(np.float32)
            adjusted_rewards = bootstrap_time_limit_rewards(
                rewards=np.asarray(raw_rewards, dtype=np.float32),
                dones=np.asarray(raw_dones, dtype=np.bool_),
                infos=info_list,
                critic_model=critic_model,
                device=device,
                gamma=float(gamma),
            )

            rollout_buffer.add_transition(
                obs=observation,
                next_obs=resolved_next_observation,
                action=stored_actions,
                reward=adjusted_rewards,
                episode_start=episode_starts,
                value=value_tensor,
                log_prob=log_prob_tensor,
                dynamics_mask=dynamics_mask,
            )

            rollout_reward_sum += float(np.asarray(raw_rewards, dtype=np.float32).sum())
            reward_array = np.asarray(raw_rewards, dtype=np.float64)
            done_array = np.asarray(raw_dones, dtype=np.bool_)
            current_rollout_returns += reward_array
            current_rollout_lengths += 1
            rollout_done_count += int(done_array.sum())
            for env_index, done in enumerate(done_array):
                if not bool(done):
                    continue
                episode_info = info_list[env_index].get("episode") if isinstance(info_list[env_index], Mapping) else None
                if isinstance(episode_info, Mapping) and "r" in episode_info:
                    episode_return = float(episode_info["r"])
                else:
                    episode_return = float(current_rollout_returns[env_index])
                if isinstance(episode_info, Mapping) and "l" in episode_info:
                    episode_length = float(episode_info["l"])
                else:
                    episode_length = float(current_rollout_lengths[env_index])
                rollout_episode_returns.append(episode_return)
                rollout_episode_lengths.append(episode_length)
                current_rollout_returns[env_index] = 0.0
                current_rollout_lengths[env_index] = 0
            observation = np.asarray(next_observation)
            episode_starts = done_array
            last_dones = episode_starts

        last_observation_tensor = torch.as_tensor(observation, device=device)
        last_values = predict_critic_values(critic_model, last_observation_tensor)

    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=last_dones)
    ep_return = float(np.mean(rollout_episode_returns)) if rollout_episode_returns else 0.0
    ep_length = float(np.mean(rollout_episode_lengths)) if rollout_episode_lengths else 0.0
    metrics = {
        "rollout/reward_per_step": rollout_reward_sum / float(max(1, int(n_steps) * num_envs)),
        "rollout/ep_return": ep_return,
        "rollout/ep_length": ep_length,
        "rollout/done_count": float(rollout_done_count),
        "rollout/advantage_mean": float(rollout_buffer.advantages.mean()),
        "rollout/advantage_std": float(rollout_buffer.advantages.std()),
    }
    return observation, episode_starts, metrics


def fixed_arch_actor_update(
    policy: PolicySupernet,
    actor_optimizer: torch.optim.Optimizer,
    rollout_buffer: DynamicsRolloutBuffer,
    arch: ArchConfig,
    action_space: spaces.Space,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    normalize_advantage: bool,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
) -> dict[str, float]:
    policy.train()
    loss_sum = 0.0
    policy_loss_sum = 0.0
    entropy_loss_sum = 0.0
    approx_kl_sum = 0.0
    clip_fraction_sum = 0.0
    update_count = 0
    continue_training = True

    for _ in range(int(n_epochs)):
        if not continue_training:
            break
        for rollout_data in rollout_buffer.get(int(batch_size)):
            batch_actions = rollout_data.actions
            if isinstance(action_space, spaces.Discrete):
                batch_actions = batch_actions.long().flatten()

            batch_old_log_probs = rollout_data.old_log_prob
            batch_advantages = rollout_data.advantages
            if normalize_advantage and batch_advantages.numel() > 1:
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            actor_optimizer.zero_grad(set_to_none=True)
            policy.set_active_arch(arch)
            new_log_probs, entropy = policy.evaluate_actions(rollout_data.observations, batch_actions)
            log_ratio = new_log_probs - batch_old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped_loss = batch_advantages * ratio
            clipped_loss = batch_advantages * torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
            policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + float(ent_coef) * entropy_loss
            approx_kl = torch.mean(((ratio - 1.0) - log_ratio).detach())
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > float(clip_range)).float().detach())
            loss.backward()

            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in policy.parameters() if parameter.requires_grad],
                    float(max_grad_norm),
                )
            actor_optimizer.step()

            update_count += 1
            loss_sum += float(loss.detach().cpu())
            policy_loss_sum += float(policy_loss.detach().cpu())
            entropy_loss_sum += float(entropy_loss.detach().cpu())
            approx_kl_sum += float(approx_kl.cpu())
            clip_fraction_sum += float(clip_fraction.cpu())

            if target_kl is not None and float(approx_kl.cpu()) > 1.5 * float(target_kl):
                continue_training = False
                break

    denominator = float(max(1, update_count))
    return {
        "actor/loss": loss_sum / denominator,
        "actor/policy_loss": policy_loss_sum / denominator,
        "actor/entropy_loss": entropy_loss_sum / denominator,
        "actor/approx_kl": approx_kl_sum / denominator,
        "actor/clip_fraction": clip_fraction_sum / denominator,
    }


def prefixed_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def critic_warmup(
    policy: PolicySupernet,
    critic_model: PPO,
    env: VecEnv,
    arch: ArchConfig,
    rollout_buffer: DynamicsRolloutBuffer,
    initial_observation: np.ndarray,
    initial_episode_starts: np.ndarray,
    target_timesteps: int,
    ppo_config: Any,
    critic_lr_schedule: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]:
    observation = np.asarray(initial_observation)
    episode_starts = np.asarray(initial_episode_starts, dtype=np.bool_)
    actual_timesteps = 0
    last_metrics: dict[str, float] = {}
    target_timesteps = max(0, int(target_timesteps))

    while actual_timesteps < target_timesteps:
        progress_remaining = 1.0 - float(actual_timesteps) / float(max(1, target_timesteps))
        critic_lr = float(critic_lr_schedule(progress_remaining)) if callable(critic_lr_schedule) else float(critic_lr_schedule)
        update_optimizer_learning_rate(critic_model.policy.optimizer, critic_lr)
        observation, episode_starts, rollout_metrics = collect_candidate_rollout(
            policy=policy,
            critic_model=critic_model,
            env=env,
            arch=arch,
            rollout_buffer=rollout_buffer,
            initial_observation=observation,
            initial_episode_starts=episode_starts,
            n_steps=int(ppo_config.n_steps),
            gamma=float(ppo_config.gamma),
            device=device,
        )
        actual_timesteps += int(ppo_config.n_steps) * int(env.num_envs)
        critic_metrics = critic_update(
            critic_model=critic_model,
            optimizer=critic_model.policy.optimizer,
            rollout_buffer=rollout_buffer,
            n_epochs=int(ppo_config.n_epochs),
            batch_size=int(ppo_config.batch_size),
            max_grad_norm=float(ppo_config.max_grad_norm),
        )
        last_metrics = {
            "progress_remaining": float(progress_remaining),
            "critic_lr": float(critic_lr),
            **prefixed_metrics("critic_warmup_rollout", rollout_metrics),
            **prefixed_metrics("critic_warmup", critic_metrics),
        }

    return observation, episode_starts, actual_timesteps, last_metrics


def set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def finetune_and_evaluate_candidate(
    args: argparse.Namespace,
    ppo_config: Any,
    arch_config: ArchConfig,
    train_seed: int,
    eval_seed: int,
) -> dict[str, Any]:
    set_global_seeds(train_seed)
    device = resolve_device(str(ppo_config.device))
    search_space = SearchSpace()
    checkpoint_payload = load_checkpoint_payload(args.supernet_checkpoint, map_location=device)
    validate_checkpoint_search_space(checkpoint_payload, search_space)

    train_env = make_vec_env_from_ppo_config(ppo_config, seed=train_seed, n_envs=get_train_n_envs(ppo_config))
    eval_env = make_vec_env_from_ppo_config(ppo_config, seed=eval_seed, n_envs=get_eval_n_envs(ppo_config))
    try:
        policy = build_policy_from_checkpoint(
            args=args,
            ppo_config=ppo_config,
            train_env=train_env,
            search_space=search_space,
            checkpoint_payload=checkpoint_payload,
            device=device,
        )
        policy.set_active_arch(arch_config)

        actor_lr_schedule = parse_schedule_value(getattr(ppo_config, "learning_rate"))
        critic_lr_schedule = parse_schedule_value(args.critic_learning_rate or getattr(ppo_config, "learning_rate"))
        clip_range_schedule = parse_schedule_value(getattr(ppo_config, "clip_range"))
        actor_optimizer = configure_actor_optimizer(
            policy=policy,
            actor_lr=actor_lr_schedule,
            backbone_lr=float(args.supernet_backbone_lr),
        )

        critic_ppo_config = build_network_ppo_config(checkpoint_payload, ppo_config, seed=train_seed)
        critic_model = build_sb3_critic_model(
            ppo_config=critic_ppo_config,
            env=train_env,
            learning_rate=critic_lr_schedule,
        )
        loaded_critic = load_critic_from_checkpoint(critic_model, checkpoint_payload)

        rollout_buffer = DynamicsRolloutBuffer(
            buffer_size=int(ppo_config.n_steps),
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            device=device,
            gae_lambda=float(ppo_config.gae_lambda),
            gamma=float(ppo_config.gamma),
            n_envs=int(train_env.num_envs),
        )

        observation = train_env.reset()
        episode_starts = np.ones((train_env.num_envs,), dtype=np.bool_)
        critic_warmup_actual_timesteps = 0
        critic_warmup_metrics: dict[str, float] = {}
        if int(args.critic_warmup_timesteps) > 0:
            observation, episode_starts, critic_warmup_actual_timesteps, critic_warmup_metrics = critic_warmup(
                policy=policy,
                critic_model=critic_model,
                env=train_env,
                arch=arch_config,
                rollout_buffer=rollout_buffer,
                initial_observation=observation,
                initial_episode_starts=episode_starts,
                target_timesteps=int(args.critic_warmup_timesteps),
                ppo_config=ppo_config,
                critic_lr_schedule=critic_lr_schedule,
                device=device,
            )

        total_timesteps = 0
        last_metrics: dict[str, float] = {}
        target_timesteps = max(0, int(args.candidate_timesteps))
        target_kl = parse_optional_float(getattr(ppo_config, "target_kl", None))

        while total_timesteps < target_timesteps:
            progress_remaining = 1.0 - float(total_timesteps) / float(max(1, target_timesteps))
            actor_lr = float(actor_lr_schedule(progress_remaining)) if callable(actor_lr_schedule) else float(actor_lr_schedule)
            critic_lr = float(critic_lr_schedule(progress_remaining)) if callable(critic_lr_schedule) else float(critic_lr_schedule)
            clip_range = float(clip_range_schedule(progress_remaining)) if callable(clip_range_schedule) else float(clip_range_schedule)
            update_actor_optimizer_learning_rate(actor_optimizer, actor_lr, float(args.supernet_backbone_lr))
            update_optimizer_learning_rate(critic_model.policy.optimizer, critic_lr)

            observation, episode_starts, rollout_metrics = collect_candidate_rollout(
                policy=policy,
                critic_model=critic_model,
                env=train_env,
                arch=arch_config,
                rollout_buffer=rollout_buffer,
                initial_observation=observation,
                initial_episode_starts=episode_starts,
                n_steps=int(ppo_config.n_steps),
                gamma=float(ppo_config.gamma),
                device=device,
            )
            total_timesteps += int(ppo_config.n_steps) * int(train_env.num_envs)
            actor_metrics = fixed_arch_actor_update(
                policy=policy,
                actor_optimizer=actor_optimizer,
                rollout_buffer=rollout_buffer,
                arch=arch_config,
                action_space=train_env.action_space,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                clip_range=clip_range,
                normalize_advantage=bool(ppo_config.normalize_advantage),
                ent_coef=float(ppo_config.ent_coef),
                max_grad_norm=float(ppo_config.max_grad_norm),
                target_kl=target_kl,
            )
            critic_metrics = critic_update(
                critic_model=critic_model,
                optimizer=critic_model.policy.optimizer,
                rollout_buffer=rollout_buffer,
                n_epochs=int(ppo_config.n_epochs),
                batch_size=int(ppo_config.batch_size),
                max_grad_norm=float(ppo_config.max_grad_norm),
            )
            last_metrics = {
                "progress_remaining": float(progress_remaining),
                "actor_lr": float(actor_lr),
                "critic_lr": float(critic_lr),
                "clip_range": float(clip_range),
                **rollout_metrics,
                **actor_metrics,
                **critic_metrics,
            }

        eval_episodes = int(getattr(ppo_config, "eval_episodes", 0) or 0)
        if eval_episodes <= 0:
            raise ValueError("ppo.eval_episodes must be positive for search fitness evaluation.")
        eval_metrics = evaluate_actor_subnet(
            policy=policy,
            train_env=train_env,
            eval_env=eval_env,
            arch=arch_config,
            n_eval_episodes=eval_episodes,
            deterministic=bool(getattr(ppo_config, "eval_deterministic", True)),
            device=device,
        )
        active_backbone_params = int(policy.backbone.elastic_num_params)
        actor_head_params = count_parameters(actor_head_parameters(policy))
        trainable_params = int(sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad))

        return {
            "return": float(eval_metrics["ep_return"]),
            "return_std": float(eval_metrics["ep_return_std"]),
            "ep_return": float(eval_metrics["ep_return"]),
            "ep_return_std": float(eval_metrics["ep_return_std"]),
            "ep_length": float(eval_metrics["ep_length"]),
            "ep_length_std": float(eval_metrics["ep_length_std"]),
            "params": active_backbone_params,
            "actor_head_params": actor_head_params,
            "policy_params": int(sum(parameter.numel() for parameter in policy.parameters())),
            "trainable_policy_params": trainable_params,
            "actual_timesteps": int(total_timesteps),
            "critic_warmup_configured_timesteps": int(args.critic_warmup_timesteps),
            "critic_warmup_actual_timesteps": int(critic_warmup_actual_timesteps),
            "total_env_timesteps": int(critic_warmup_actual_timesteps + total_timesteps),
            "loaded_critic": bool(loaded_critic),
            "critic_warmup_metrics": critic_warmup_metrics,
            "finetune_metrics": last_metrics,
        }
    finally:
        train_env.close()
        eval_env.close()


@dataclass(frozen=True)
class NewPolicySubnetEvalConfig:
    args: dict[str, Any]
    ppo_config: dict[str, Any]
    arch_config: dict[str, Any]
    gene: list[int]
    train_seed: int
    eval_seed: int


def _evaluate_new_policy_subnet_worker(config: NewPolicySubnetEvalConfig) -> dict[str, Any]:
    worker_threads = int(config.args.get("worker_torch_threads", 1))
    if worker_threads > 0:
        torch.set_num_threads(worker_threads)
    args = argparse.Namespace(**config.args)
    ppo_config = OmegaConf.create(config.ppo_config)
    arch_config = ArchConfig.from_dict(config.arch_config)
    result = finetune_and_evaluate_candidate(
        args=args,
        ppo_config=ppo_config,
        arch_config=arch_config,
        train_seed=config.train_seed,
        eval_seed=config.eval_seed,
    )
    return {
        "gene": config.gene,
        "arch_config": config.arch_config,
        "train_seed": int(config.train_seed),
        "eval_seed": int(config.eval_seed),
        "pid": os.getpid(),
        **result,
    }


class NewPolicySubnetProblem(Problem):
    """EvoX problem that evaluates policy-supernet subnet genes by PPO fine-tuning."""

    def __init__(
        self,
        args: argparse.Namespace,
        ppo_config: DictConfig,
        codec: GeneCodec,
    ) -> None:
        super().__init__()
        self.args_dict = vars(args).copy()
        self.ppo_config_dict = ppo_config_to_dict(ppo_config)
        self.codec = codec
        self.eval_workers = max(1, int(args.eval_workers))
        self.mp_start_method = args.mp_start_method
        self.eval_call_index = 0
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.last_records: list[dict[str, Any]] = []
        self.last_cache_hits = 0
        self._pool = None

    def _make_eval_config(self, gene: list[int], candidate_index: int) -> NewPolicySubnetEvalConfig:
        train_seed = (
            int(self.ppo_config_dict["seed"])
            + self.eval_call_index * int(self.args_dict["eval_call_seed_stride"])
            + candidate_index * int(self.args_dict["candidate_seed_stride"])
        )
        eval_seed = train_seed + int(self.args_dict["eval_seed_offset"])
        return NewPolicySubnetEvalConfig(
            args=self.args_dict,
            ppo_config=self.ppo_config_dict,
            arch_config=self.codec.gene_to_arch(gene).to_dict(),
            gene=gene,
            train_seed=train_seed,
            eval_seed=eval_seed,
        )

    def _ensure_pool(self):
        if self.eval_workers <= 1:
            return None
        if self._pool is None:
            context = mp.get_context(self.mp_start_method)
            self._pool = context.Pool(processes=self.eval_workers)
        return self._pool

    @torch.compiler.disable
    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        genes = [[int(round(value)) for value in row.tolist()] for row in pop.detach().cpu()]
        records: list[dict[str, Any] | None] = [None] * len(genes)
        pending: list[NewPolicySubnetEvalConfig] = []
        pending_indices: list[int] = []
        self.last_cache_hits = 0

        for index, gene in enumerate(genes):
            self.codec.validate_gene(gene)
            key = tuple(gene)
            if key in self.cache:
                records[index] = copy.deepcopy(self.cache[key])
                self.last_cache_hits += 1
            else:
                pending.append(self._make_eval_config(gene, index))
                pending_indices.append(index)

        if pending:
            if self.eval_workers <= 1:
                evaluated = [_evaluate_new_policy_subnet_worker(config) for config in pending]
            else:
                pool = self._ensure_pool()
                evaluated = pool.map(_evaluate_new_policy_subnet_worker, pending)
            for index, record in zip(pending_indices, evaluated):
                key = tuple(record["gene"])
                self.cache[key] = copy.deepcopy(record)
                records[index] = record

        self.last_records = [record for record in records if record is not None]
        self.eval_call_index += 1
        objectives = [
            [-float(record["return"]), float(record["params"])]
            for record in self.last_records
        ]
        return torch.tensor(objectives, dtype=torch.float32, device=pop.device)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run("new_stage2_ea_search", run_config, output_dir)
    search_space = SearchSpace()
    codec = GeneCodec(search_space)
    (output_dir / "search_space.json").write_text(json.dumps(search_space.to_dict(), indent=2))

    lower_bounds, upper_bounds = codec.gene_bounds()
    initial_population = build_initial_population(args, codec)
    algorithm = DiscreteNSGA2(
        pop_size=args.population_size,
        n_objs=2,
        lb=torch.tensor(lower_bounds, dtype=torch.float32),
        ub=torch.tensor(upper_bounds, dtype=torch.float32),
        device=torch.device("cpu"),
        initial_population=torch.tensor(initial_population, dtype=torch.float32),
    )
    problem = NewPolicySubnetProblem(args=args, ppo_config=ppo_config, codec=codec)
    monitor = EvalMonitor(
        multi_obj=True,
        full_fit_history=args.save_full_history,
        full_sol_history=args.save_full_history,
        full_pop_history=args.save_full_history,
        device=torch.device("cpu"),
        history_device=torch.device("cpu"),
    )
    workflow = StdWorkflow(algorithm, problem, monitor=monitor, device=torch.device("cpu"))

    records_path = output_dir / "nsga2_records.jsonl"
    log_path = output_dir / "search.log"
    for path in (records_path, log_path):
        if path.exists():
            path.unlink()

    all_records: list[dict[str, Any]] = []
    try:
        workflow.init_step()
        latest_pop = monitor.get_latest_solution()
        latest_fit = monitor.get_latest_fitness()
        records = build_generation_records(0, latest_pop, latest_fit, codec, problem)
        write_generation(records_path, records)
        log_generation(log_path, 0, records, problem.last_cache_hits, wandb_run)
        all_records.extend(records)

        for generation in range(1, args.generations):
            workflow.step()
            latest_pop = monitor.get_latest_solution()
            latest_fit = monitor.get_latest_fitness()
            records = build_generation_records(generation, latest_pop, latest_fit, codec, problem)
            write_generation(records_path, records)
            log_generation(log_path, generation, records, problem.last_cache_hits, wandb_run)
            all_records.extend(records)
    finally:
        problem.close()

    final_pop = monitor.get_latest_solution().detach().cpu()
    final_fit = monitor.get_latest_fitness().detach().cpu()
    final_records = build_generation_records(max(0, args.generations - 1), final_pop, final_fit, codec, problem)
    pareto_records = [record for record in final_records if record["is_pareto_front"]]
    manifest = {
        "stage": "new_stage2_ea_search",
        "records": str(records_path),
        "log": str(log_path),
        "search_space": str(output_dir / "search_space.json"),
        "supernet_checkpoint": str(args.supernet_checkpoint),
        "objectives": ["negative_return", "params"],
        "fitness_procedure": "load_policy_supernet_then_ppo_finetune_then_eval_return",
        "pareto_front": pareto_records,
        "final_population": final_records,
        "num_logged_records": len(all_records),
        "cache_size": len(problem.cache),
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log_wandb(
        wandb_run,
        {
            "num_logged_records": len(all_records),
            "cache_size": len(problem.cache),
            "final_pareto_count": len(pareto_records),
        },
        step=max(0, args.generations - 1),
    )
    log_wandb_artifact(
        wandb_run,
        name=f"new-stage2-{output_dir.name}",
        artifact_type="new-stage2-output",
        paths=[records_path, log_path, output_dir / "search_space.json", manifest_path],
    )
    finish_wandb_run(wandb_run)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
