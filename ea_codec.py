from __future__ import annotations

import random
from typing import Sequence

from supernet_backbone import ArchConfig, SearchSpace, LayerConfig


class GeneCodec:
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    @property
    def gene_length(self) -> int:
        return self.search_space.num_stages + 2 * sum(self.search_space.max_stage_depths)


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

    def get_gene_space(self) -> dict[str, list[int]]:
        lower, upper = self.gene_bounds()
        return {"lower_bounds": lower, "upper_bounds": upper}

    def sample_gene(self) -> list[int]:
        gene: list[int] = []
        for candidates in self.search_space.stage_depth_candidates:
            gene.append(random.randrange(len(candidates)))
        for depth in self.search_space.max_stage_depths:
            for _ in range(depth):
                gene.append(random.randrange(len(self.search_space.kernel_size_candidates)))
                gene.append(random.randrange(len(self.search_space.expand_ratio_candidates)))
        return gene

    def min_gene(self) -> list[int]:
        return [0] * self.gene_length

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
            raise ValueError(f"Expected gene length {self.gene_length}, got {len(gene)}.")
        offset = 0
        for candidates in self.search_space.stage_depth_candidates:
            value = int(gene[offset])
            if value < 0 or value >= len(candidates):
                raise ValueError(f"Depth gene index {value} is out of range.")
            offset += 1
        for _ in range(sum(self.search_space.max_stage_depths)):
            kernel_index = int(gene[offset])
            expand_index = int(gene[offset + 1])
            if kernel_index < 0 or kernel_index >= len(self.search_space.kernel_size_candidates):
                raise ValueError(f"Kernel gene index {kernel_index} is out of range.")
            if expand_index < 0 or expand_index >= len(self.search_space.expand_ratio_candidates):
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
                kernel_size = self.search_space.kernel_size_candidates[int(gene[offset])]
                expand_ratio = self.search_space.expand_ratio_candidates[int(gene[offset + 1])]
                stage_layers.append(
                    LayerConfig(kernel_size=kernel_size, expand_ratio=expand_ratio)
                )
                offset += 2
            layer_configs.append(tuple(stage_layers))

        return ArchConfig(
            stage_depths=tuple(stage_depths),
            layer_configs=tuple(layer_configs),
        )

    def arch_to_gene(self, arch: ArchConfig) -> list[int]:
        if len(arch.stage_depths) != self.search_space.num_stages:
            raise ValueError("Architecture has the wrong number of stage depths.")
        if len(arch.layer_configs) != self.search_space.num_stages:
            raise ValueError("Architecture has the wrong number of stage layer configs.")

        gene: list[int] = []
        for depth, candidates in zip(arch.stage_depths, self.search_space.stage_depth_candidates):
            gene.append(candidates.index(depth))

        for stage_layers, max_depth in zip(arch.layer_configs, self.search_space.max_stage_depths):
            if len(stage_layers) != max_depth:
                raise ValueError("Architecture stores the wrong number of layer configs.")
            for layer in stage_layers:
                gene.append(self.search_space.kernel_size_candidates.index(layer.kernel_size))
                gene.append(self.search_space.expand_ratio_candidates.index(layer.expand_ratio))

        self.validate_gene(gene)
        return gene

    def mutate_gene(
        self,
        gene: Sequence[int],
        mutation_prob: float,
    ) -> list[int]:
        self.validate_gene(gene)
        mutated = [int(value) for value in gene]
        offset = 0
        for candidates in self.search_space.stage_depth_candidates:
            if random.random() < mutation_prob:
                mutated[offset] = random.randrange(len(candidates))
            offset += 1
        for _ in range(sum(self.search_space.max_stage_depths)):
            if random.random() < mutation_prob:
                mutated[offset] = random.randrange(len(self.search_space.kernel_size_candidates))
            if random.random() < mutation_prob:
                mutated[offset + 1] = random.randrange(len(self.search_space.expand_ratio_candidates))
            offset += 2
        return mutated

    def crossover_gene(
        self,
        parent_a: Sequence[int],
        parent_b: Sequence[int],
    ) -> list[int]:
        self.validate_gene(parent_a)
        self.validate_gene(parent_b)
        return [
            int(value_a) if random.random() < 0.5 else int(value_b)
            for value_a, value_b in zip(parent_a, parent_b)
        ]
