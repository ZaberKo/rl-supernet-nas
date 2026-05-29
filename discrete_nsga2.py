"""Discrete NSGA-II algorithm for integer-gene NAS search.

Extracted from the former ``nsga2_search.py`` to decouple the algorithm
implementation from the evaluation logic.
"""

from __future__ import annotations

import torch
from evox.algorithms.mo import NSGA2
from evox.core import Mutable


def random_integer_population(
    *, pop_size: int, lb: torch.Tensor, ub: torch.Tensor
) -> torch.Tensor:
    span = (ub - lb + 1).clamp_min(1)
    random_values = torch.rand(pop_size, lb.numel(), device=lb.device)
    return torch.floor(random_values * span + lb)


def uniform_integer_crossover(x: torch.Tensor, pro_c: float = 1.0) -> torch.Tensor:
    offspring = x.round().clone()
    pair_count = offspring.shape[0] // 2
    if pair_count == 0:
        return offspring

    parent1 = offspring[:pair_count]
    parent2 = offspring[pair_count : pair_count * 2]
    pair_mask = torch.rand(pair_count, 1, device=x.device) < pro_c
    gene_mask = torch.rand(pair_count, x.shape[1], device=x.device) < 0.5
    swap_mask = pair_mask & gene_mask

    parent1_values = parent1.clone()
    parent2_values = parent2.clone()
    parent1[swap_mask] = parent2_values[swap_mask]
    parent2[swap_mask] = parent1_values[swap_mask]
    return offspring


def random_reset_integer_mutation(
    x: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    mutation_prob: float | None = None,
) -> torch.Tensor:
    offspring = x.round().clone()
    if mutation_prob is None:
        mutation_prob = 1.0 / max(1, offspring.shape[1])

    mutation_mask = torch.rand_like(offspring, dtype=torch.float32) < mutation_prob
    sampled = random_integer_population(
        pop_size=offspring.shape[0],
        lb=lb.to(offspring.device),
        ub=ub.to(offspring.device),
    )
    offspring[mutation_mask] = sampled[mutation_mask]
    return offspring


class DiscreteNSGA2(NSGA2):
    """NSGA-II configured for fixed-length integer NAS genes."""

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        device: torch.device | None = None,
        crossover_prob: float = 1.0,
        mutation_prob: float | None = None,
        initial_population: torch.Tensor | None = None,
    ) -> None:
        lb = lb.to(dtype=torch.float32)
        ub = ub.to(dtype=torch.float32)
        super().__init__(
            pop_size=pop_size,
            n_objs=n_objs,
            lb=lb,
            ub=ub,
            crossover_op=lambda x: uniform_integer_crossover(x, pro_c=crossover_prob),
            mutation_op=lambda x, lower, upper: random_reset_integer_mutation(
                x, lower, upper, mutation_prob=mutation_prob
            ),
            device=device,
        )
        population = random_integer_population(
            pop_size=pop_size, lb=self.lb, ub=self.ub
        )
        if initial_population is not None and initial_population.numel() > 0:
            initial_population = initial_population.to(
                device=self.lb.device, dtype=torch.float32
            ).round()
            count = min(pop_size, initial_population.shape[0])
            population[:count] = initial_population[:count]
        self.pop = Mutable(population)
