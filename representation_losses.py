from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentDynamicsPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        start_latent: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        if action_features.dim() == 2:
            return self.net(torch.cat([start_latent, action_features], dim=-1))
        if action_features.dim() != 3:
            raise ValueError("action_features must have shape [batch, action_dim] or [batch, horizon, action_dim].")

        latent = start_latent
        predictions = []
        for step_index in range(action_features.size(1)):
            latent = self.net(torch.cat([latent, action_features[:, step_index]], dim=-1))
            predictions.append(latent)
        return torch.stack(predictions, dim=1)


def latent_dynamics_loss(
    predictions: torch.Tensor,
    teacher_targets: torch.Tensor,
    beta_weights: torch.Tensor | Sequence[float] | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    predictions = F.normalize(predictions, dim=-1)
    teacher_targets = F.normalize(teacher_targets.detach(), dim=-1)
    distance = 2.0 - 2.0 * F.cosine_similarity(predictions, teacher_targets, dim=-1)
    if distance.dim() == 1:
        if sample_weights is None:
            return distance.mean()
        weights = sample_weights.to(dtype=distance.dtype, device=distance.device).view_as(distance)
        return (distance * weights).sum() / weights.sum().clamp_min(1.0)
    if distance.dim() != 2:
        raise ValueError("latent dynamics distance must have shape [batch] or [batch, horizon].")

    weights = torch.ones(distance.size(1), dtype=distance.dtype, device=distance.device)
    if beta_weights is not None:
        weights = torch.as_tensor(beta_weights, dtype=distance.dtype, device=distance.device)
        if weights.numel() != distance.size(1):
            raise ValueError("beta_weights length must match the dynamics horizon.")
    weighted_distance = distance * weights.view(1, -1)
    if sample_weights is not None:
        weights = sample_weights.to(dtype=distance.dtype, device=distance.device)
        if weights.shape != distance.shape:
            raise ValueError("sample_weights shape must match the latent dynamics distance shape.")
        weighted_distance = weighted_distance * weights
    return weighted_distance.sum(dim=1).mean()


def cosine_kd_loss(
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    student_latent = F.normalize(student_latent, dim=-1)
    teacher_latent = F.normalize(teacher_latent.detach(), dim=-1)
    distance = 2.0 - 2.0 * F.cosine_similarity(student_latent, teacher_latent, dim=-1)
    if sample_weights is None:
        return distance.mean()
    weights = sample_weights.to(dtype=distance.dtype, device=distance.device).view_as(distance)
    return (distance * weights).sum() / weights.sum().clamp_min(1.0)


def get_action_dim(action_space: gym.Space) -> int:
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action_space.n)
    if isinstance(action_space, gym.spaces.Box):
        return int(torch.tensor(action_space.shape).prod().item())
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return int(sum(int(value) for value in action_space.nvec))
    if isinstance(action_space, gym.spaces.MultiBinary):
        return int(torch.tensor(action_space.shape).prod().item())
    raise TypeError(f"Unsupported action space: {type(action_space).__name__}")


def encode_action_batch(actions: torch.Tensor, action_space: gym.Space) -> torch.Tensor:
    if isinstance(action_space, gym.spaces.Discrete):
        return F.one_hot(actions.long().view(-1), num_classes=int(action_space.n)).float()
    if isinstance(action_space, gym.spaces.Box):
        return actions.float().view(actions.size(0), -1)
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        parts = []
        for index, classes in enumerate(action_space.nvec):
            parts.append(F.one_hot(actions[:, index].long(), num_classes=int(classes)).float())
        return torch.cat(parts, dim=-1)
    if isinstance(action_space, gym.spaces.MultiBinary):
        return actions.float().view(actions.size(0), -1)
    raise TypeError(f"Unsupported action space: {type(action_space).__name__}")
