from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from primitive_blocks import (
    ElasticConv2d,
    ElasticGroupNorm2d,
    ElasticLinear,
    GroupNorm2d,
)


@dataclass(frozen=True)
class LayerConfig:
    kernel_size: int
    expand_ratio: int


@dataclass(frozen=True)
class ArchConfig:
    stage_depths: tuple[int, ...]
    layer_configs: tuple[tuple[LayerConfig, ...], ...]

    def active_layers(self, stage_index: int) -> tuple[LayerConfig, ...]:
        return self.layer_configs[stage_index][: self.stage_depths[stage_index]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_depths": list(self.stage_depths),
            "layer_configs": [
                [asdict(layer) for layer in stage_layers]
                for stage_layers in self.layer_configs
            ],
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ArchConfig:
        return cls(
            stage_depths=tuple(int(value) for value in config_dict["stage_depths"]),
            layer_configs=tuple(
                tuple(
                    LayerConfig(
                        kernel_size=int(layer["kernel_size"]),
                        expand_ratio=int(layer["expand_ratio"]),
                    )
                    for layer in stage_layers
                )
                for stage_layers in config_dict["layer_configs"]
            ),
        )


@dataclass(frozen=True)
class SearchSpace:
    stage_widths: tuple[int, ...] = (16, 32, 64)
    stage_depth_candidates: tuple[tuple[int, ...], ...] = ((1, 2), (1, 2), (1, 2))
    kernel_size_candidates: tuple[int, ...] = (3, 5)
    expand_ratio_candidates: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self) -> None:
        if len(self.stage_widths) != len(self.stage_depth_candidates):
            raise ValueError(
                "stage_widths and stage_depth_candidates must have the same length."
            )
        if not self.kernel_size_candidates:
            raise ValueError("kernel_size_candidates must not be empty.")
        if not self.expand_ratio_candidates:
            raise ValueError("expand_ratio_candidates must not be empty.")
        for candidates in self.stage_depth_candidates:
            if not candidates:
                raise ValueError("Each stage must have at least one depth candidate.")
            if tuple(sorted(candidates)) != tuple(candidates):
                raise ValueError("Depth candidates must be sorted in ascending order.")

    @property
    def num_stages(self) -> int:
        return len(self.stage_widths)

    @property
    def max_stage_depths(self) -> tuple[int, ...]:
        return tuple(max(candidates) for candidates in self.stage_depth_candidates)

    @property
    def max_expand_ratio(self) -> int:
        return max(self.expand_ratio_candidates)

    def min_arch(self) -> ArchConfig:
        return self._build_arch(
            stage_depth_indices=[0] * self.num_stages,
            kernel_indices=[0] * sum(self.max_stage_depths),
            expand_indices=[0] * sum(self.max_stage_depths),
        )

    def max_arch(self) -> ArchConfig:
        return self._build_arch(
            stage_depth_indices=[
                len(candidates) - 1 for candidates in self.stage_depth_candidates
            ],
            kernel_indices=[len(self.kernel_size_candidates) - 1]
            * sum(self.max_stage_depths),
            expand_indices=[len(self.expand_ratio_candidates) - 1]
            * sum(self.max_stage_depths),
        )

    def sample_arch(self) -> ArchConfig:
        total_layers = sum(self.max_stage_depths)
        return self._build_arch(
            stage_depth_indices=[
                random.randrange(len(candidates))
                for candidates in self.stage_depth_candidates
            ],
            kernel_indices=[
                random.randrange(len(self.kernel_size_candidates))
                for _ in range(total_layers)
            ],
            expand_indices=[
                random.randrange(len(self.expand_ratio_candidates))
                for _ in range(total_layers)
            ],
        )

    def _build_arch(
        self,
        stage_depth_indices: list[int],
        kernel_indices: list[int],
        expand_indices: list[int],
    ) -> ArchConfig:
        stage_depths = tuple(
            candidates[index]
            for candidates, index in zip(
                self.stage_depth_candidates, stage_depth_indices, strict=True
            )
        )
        layer_configs = []
        offset = 0
        for max_depth in self.max_stage_depths:
            stage_layers = []
            for _ in range(max_depth):
                stage_layers.append(
                    LayerConfig(
                        kernel_size=self.kernel_size_candidates[kernel_indices[offset]],
                        expand_ratio=self.expand_ratio_candidates[
                            expand_indices[offset]
                        ],
                    )
                )
                offset += 1
            layer_configs.append(tuple(stage_layers))
        return ArchConfig(stage_depths=stage_depths, layer_configs=tuple(layer_configs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_widths": list(self.stage_widths),
            "stage_depth_candidates": [
                list(values) for values in self.stage_depth_candidates
            ],
            "kernel_size_candidates": list(self.kernel_size_candidates),
            "expand_ratio_candidates": list(self.expand_ratio_candidates),
        }


class ConvLNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            GroupNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FixedMBConvBlock(nn.Module):
    def __init__(
        self,
        expand: nn.Module,
        expand_norm: nn.Module,
        depthwise: nn.Module,
        depthwise_norm: nn.Module,
        project: nn.Module,
        project_norm: nn.Module,
    ):
        super().__init__()
        self.expand = expand
        self.expand_norm = expand_norm
        self.depthwise = depthwise
        self.depthwise_norm = depthwise_norm
        self.project = project
        self.project_norm = project_norm
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.expand_norm(self.expand(x)))
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        x = self.project_norm(self.project(x))
        return self.activation(x + residual)


class FixedCNNBackbone(nn.Module):
    def __init__(
        self,
        stem: nn.Module,
        transitions: list[nn.Module],
        stages: list[list[nn.Module]],
        pool: nn.Module,
        project: nn.Module,
        output_activation: nn.Module,
    ):
        super().__init__()
        self.stem = stem
        self.transitions = nn.ModuleList(transitions)
        self.stages = nn.ModuleList(
            [nn.ModuleList(stage_blocks) for stage_blocks in stages]
        )
        self.pool = pool
        self.project = project
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        x = self.stem(x)
        for stage_index, blocks in enumerate(self.stages):
            x = self.transitions[stage_index](x)
            for block in blocks:
                x = block(x)
        x = self.pool(x).flatten(1)
        x = self.project(x)
        return self.output_activation(x)


class ElasticMBConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        max_expand_ratio: int,
        kernel_size_candidates: tuple[int, ...],
    ):
        super().__init__()
        self.channels = channels
        self.max_expand_ratio = max_expand_ratio
        self.kernel_size_candidates = kernel_size_candidates
        self.max_mid_channels = channels * max_expand_ratio
        max_kernel = max(kernel_size_candidates)

        self.expand = ElasticConv2d(
            super_in_channels=channels,
            super_out_channels=self.max_mid_channels,
            kernel_size=1,
            padding=0,
            bias=False,
            candidate_kernel_sizes=(1,),
        )
        self.expand_norm = ElasticGroupNorm2d(super_num_channels=self.max_mid_channels)
        self.depthwise = ElasticConv2d(
            super_in_channels=self.max_mid_channels,
            super_out_channels=self.max_mid_channels,
            kernel_size=max_kernel,
            padding=max_kernel // 2,
            groups=self.max_mid_channels,
            bias=False,
            candidate_kernel_sizes=kernel_size_candidates,
        )
        self.depthwise_norm = ElasticGroupNorm2d(
            super_num_channels=self.max_mid_channels
        )
        self.project = ElasticConv2d(
            super_in_channels=self.max_mid_channels,
            super_out_channels=channels,
            kernel_size=1,
            padding=0,
            bias=False,
            candidate_kernel_sizes=(1,),
        )
        self.project_norm = ElasticGroupNorm2d(super_num_channels=channels)
        self.activation = nn.SiLU(inplace=True)
        self.active_config = LayerConfig(
            kernel_size=max_kernel,
            expand_ratio=max_expand_ratio,
        )
        self.set_sample_config(
            sample_kernel_size=self.active_config.kernel_size,
            sample_expand_ratio=self.active_config.expand_ratio,
        )

    def set_sample_config(
        self,
        *,
        sample_kernel_size: int,
        sample_expand_ratio: int,
    ) -> None:
        if sample_expand_ratio > self.max_expand_ratio:
            raise ValueError("sample_expand_ratio exceeds the block maximum.")
        if sample_kernel_size not in self.kernel_size_candidates:
            raise ValueError("sample_kernel_size is not in the candidate set.")
        mid_channels = self.channels * sample_expand_ratio
        self.active_config = LayerConfig(
            kernel_size=sample_kernel_size,
            expand_ratio=sample_expand_ratio,
        )
        self.expand.set_sample_config(
            sample_in_channels=self.channels,
            sample_out_channels=mid_channels,
            sample_kernel_size=1,
        )
        self.expand_norm.set_sample_config(sample_num_channels=mid_channels)
        self.depthwise.set_sample_config(
            sample_in_channels=mid_channels,
            sample_out_channels=mid_channels,
            sample_groups=mid_channels,
            sample_kernel_size=sample_kernel_size,
        )
        self.depthwise_norm.set_sample_config(sample_num_channels=mid_channels)
        self.project.set_sample_config(
            sample_in_channels=mid_channels,
            sample_out_channels=self.channels,
            sample_kernel_size=1,
        )
        self.project_norm.set_sample_config(sample_num_channels=self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.expand_norm(self.expand(x)))
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        x = self.project_norm(self.project(x))
        return self.activation(x + residual)

    def get_active_subnet(self) -> nn.Module:
        return FixedMBConvBlock(
            expand=self.expand.get_active_subnet(),
            expand_norm=self.expand_norm.get_active_subnet(),
            depthwise=self.depthwise.get_active_subnet(),
            depthwise_norm=self.depthwise_norm.get_active_subnet(),
            project=self.project.get_active_subnet(),
            project_norm=self.project_norm.get_active_subnet(),
        )

    @property
    def elastic_num_params(self) -> int:
        return int(
            self.expand.elastic_num_params
            + self.expand_norm.elastic_num_params
            + self.depthwise.elastic_num_params
            + self.depthwise_norm.elastic_num_params
            + self.project.elastic_num_params
            + self.project_norm.elastic_num_params
        )


class SupernetCNNBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        search_space: SearchSpace | None = None,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.search_space = search_space or SearchSpace()
        self.feature_dim = feature_dim
        widths = self.search_space.stage_widths

        self.stem = ConvLNAct(input_channels, widths[0], stride=2)
        transitions: list[nn.Module] = [nn.Identity()]
        for in_channels, out_channels in pairwise(widths):
            transitions.append(ConvLNAct(in_channels, out_channels, stride=2))
        self.transitions = nn.ModuleList(transitions)

        stages: list[nn.ModuleList] = []
        for width, max_depth in zip(
            widths, self.search_space.max_stage_depths, strict=True
        ):
            blocks = nn.ModuleList(
                [
                    ElasticMBConvBlock(
                        channels=width,
                        max_expand_ratio=self.search_space.max_expand_ratio,
                        kernel_size_candidates=self.search_space.kernel_size_candidates,
                    )
                    for _ in range(max_depth)
                ]
            )
            stages.append(blocks)
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.project = ElasticLinear(
            super_in_dim=widths[-1], super_out_dim=feature_dim, bias=True
        )
        self.project.set_sample_config(
            sample_in_dim=widths[-1], sample_out_dim=feature_dim
        )
        self.output_activation = nn.SiLU(inplace=True)
        self.active_arch = self.search_space.max_arch()
        self.set_sample_config(self.active_arch)

    def set_sample_config(self, arch_config: ArchConfig) -> None:
        if len(arch_config.stage_depths) != len(self.stages):
            raise ValueError("Architecture stage count does not match the backbone.")
        self.active_arch = arch_config
        for blocks, stage_layers in zip(
            self.stages, arch_config.layer_configs, strict=True
        ):
            if len(stage_layers) != len(blocks):
                raise ValueError(
                    "Architecture layer count does not match the backbone."
                )
            for block, layer_config in zip(blocks, stage_layers, strict=True):
                block.set_sample_config(
                    sample_kernel_size=layer_config.kernel_size,
                    sample_expand_ratio=layer_config.expand_ratio,
                )

    def set_active_arch(self, arch: ArchConfig) -> None:
        self.set_sample_config(arch)

    def set_max_arch(self) -> None:
        self.set_sample_config(self.search_space.max_arch())

    def set_min_arch(self) -> None:
        self.set_sample_config(self.search_space.min_arch())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        x = self.stem(x)
        for stage_index, blocks in enumerate(self.stages):
            x = self.transitions[stage_index](x)
            active_depth = self.active_arch.stage_depths[stage_index]
            for block in blocks[:active_depth]:
                x = block(x)
        x = self.pool(x).flatten(1)
        x = self.project(x)
        return self.output_activation(x)

    def get_active_subnet(self) -> nn.Module:
        stages: list[list[nn.Module]] = []
        for stage_index, blocks in enumerate(self.stages):
            active_depth = self.active_arch.stage_depths[stage_index]
            stages.append(
                [block.get_active_subnet() for block in blocks[:active_depth]]
            )
        return FixedCNNBackbone(
            stem=copy.deepcopy(self.stem),
            transitions=[copy.deepcopy(transition) for transition in self.transitions],
            stages=stages,
            pool=copy.deepcopy(self.pool),
            project=self.project.get_active_subnet(),
            output_activation=copy.deepcopy(self.output_activation),
        )

    @property
    def elastic_num_params(self) -> int:
        total = sum(parameter.numel() for parameter in self.stem.parameters())
        for transition in self.transitions:
            total += sum(parameter.numel() for parameter in transition.parameters())
        for stage_index, blocks in enumerate(self.stages):
            active_depth = self.active_arch.stage_depths[stage_index]
            for block in blocks[:active_depth]:
                total += block.elastic_num_params
        total += self.project.elastic_num_params
        return int(total)


def infer_input_channels(observation_shape: tuple[int, ...]) -> int:
    if len(observation_shape) != 3:
        raise ValueError("Only image observations with three dimensions are supported.")
    if observation_shape[0] in (1, 3, 4):
        return int(observation_shape[0])
    return int(observation_shape[-1])


POLICY_BACKBONE_PREFIX = "features_extractor.backbone."


def extract_backbone_state_dict_from_policy_state_dict(
    policy_state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    backbone_state_dict = {
        key.removeprefix(POLICY_BACKBONE_PREFIX): value
        for key, value in policy_state_dict.items()
        if key.startswith(POLICY_BACKBONE_PREFIX)
    }
    if not backbone_state_dict:
        raise KeyError(
            f"No keys found with prefix {POLICY_BACKBONE_PREFIX!r} in policy_state_dict."
        )
    return backbone_state_dict


def load_backbone_from_policy_checkpoint(
    backbone: SupernetCNNBackbone,
    checkpoint_path: str | Path | None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None

    from checkpoint_utils import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, map_location)
    policy_state_dict = checkpoint.get("policy_state_dict")
    if not isinstance(policy_state_dict, Mapping):
        raise KeyError("Stage1 PPO checkpoint must contain policy_state_dict.")
    state_dict = extract_backbone_state_dict_from_policy_state_dict(policy_state_dict)
    backbone.load_state_dict(state_dict, strict=False)
    return dict(checkpoint)


def load_backbone_from_backbone_checkpoint(
    backbone: SupernetCNNBackbone,
    checkpoint_path: str | Path | None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None

    from checkpoint_utils import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, map_location)
    backbone_state_dict = checkpoint.get("backbone_state_dict")
    if not isinstance(backbone_state_dict, Mapping):
        raise KeyError("Stage2 checkpoint must contain backbone_state_dict.")
    backbone.load_state_dict(backbone_state_dict, strict=False)
    return dict(checkpoint)
