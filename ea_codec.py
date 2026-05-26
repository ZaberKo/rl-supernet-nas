from __future__ import annotations

import random
from collections.abc import Sequence

from supernet_backbone import ArchConfig, LayerConfig, SearchSpace


class GeneCodec:
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    @property
    def gene_length(self) -> int:
        return self.search_space.num_stages + 2 * sum(
            self.search_space.max_stage_depths
        )

    def gene_bounds(self) -> tuple[list[int], list[int]]:
        lower = [0] * self.gene_length
        upper: list[int] = []
        for candidates in self.search_space.stage_depth_candidates:
            upper.append(len(candidates) - 1)
        for depth in self.search_space.max_stage_depths:
            for _ in range(depth):
                upper.append(len(self.search_space.kernel_size_candidates) - 1)
                upper.append(len(self.search_space.expand_ratio_candidates) - 1)
        return lower, upper

    def sample_gene(self) -> list[int]:
        gene: list[int] = []
        for candidates in self.search_space.stage_depth_candidates:
            gene.append(random.randrange(len(candidates)))
        for depth in self.search_space.max_stage_depths:
            for _ in range(depth):
                gene.append(
                    random.randrange(len(self.search_space.kernel_size_candidates))
                )
                gene.append(
                    random.randrange(len(self.search_space.expand_ratio_candidates))
                )
        return gene

    def max_gene(self) -> list[int]:
        gene = [
            len(candidates) - 1
            for candidates in self.search_space.stage_depth_candidates
        ]
        for depth in self.search_space.max_stage_depths:
            for _ in range(depth):
                gene.append(len(self.search_space.kernel_size_candidates) - 1)
                gene.append(len(self.search_space.expand_ratio_candidates) - 1)
        return gene

    def validate_gene(self, gene: Sequence[int]) -> None:
        if len(gene) != self.gene_length:
            raise ValueError(
                f"Expected gene length {self.gene_length}, got {len(gene)}."
            )
        offset = 0
        for candidates in self.search_space.stage_depth_candidates:
            value = int(gene[offset])
            if value < 0 or value >= len(candidates):
                raise ValueError(f"Depth gene index {value} is out of range.")
            offset += 1
        for _ in range(sum(self.search_space.max_stage_depths)):
            kernel_index = int(gene[offset])
            expand_index = int(gene[offset + 1])
            if kernel_index < 0 or kernel_index >= len(
                self.search_space.kernel_size_candidates
            ):
                raise ValueError(f"Kernel gene index {kernel_index} is out of range.")
            if expand_index < 0 or expand_index >= len(
                self.search_space.expand_ratio_candidates
            ):
                raise ValueError(f"Expand gene index {expand_index} is out of range.")
            offset += 2

    def gene_to_arch(self, gene: Sequence[int]) -> ArchConfig:
        self.validate_gene(gene)
        offset = 0
        stage_depths = []
        for candidates in self.search_space.stage_depth_candidates:
            stage_depths.append(candidates[int(gene[offset])])
            offset += 1

        layer_configs = []
        for max_depth in self.search_space.max_stage_depths:
            stage_layers = []
            for _ in range(max_depth):
                kernel_size = self.search_space.kernel_size_candidates[
                    int(gene[offset])
                ]
                expand_ratio = self.search_space.expand_ratio_candidates[
                    int(gene[offset + 1])
                ]
                stage_layers.append(
                    LayerConfig(kernel_size=kernel_size, expand_ratio=expand_ratio)
                )
                offset += 2
            layer_configs.append(tuple(stage_layers))

        return ArchConfig(
            stage_depths=tuple(stage_depths),
            layer_configs=tuple(layer_configs),
        )
