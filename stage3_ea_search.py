from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from evox.operators.selection import non_dominate_rank
from evox.workflows import EvalMonitor, StdWorkflow

from config_utils import add_ppo_config_args, load_ppo_config
from ea_codec import GeneCodec
from nsga2_search import DiscreteNSGA2, RLSubnetProblem
from supernet_backbone import SearchSpace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 NSGA-II subnet search with EvoX.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--output_dir", default="runs/stage3", help="Directory for NSGA-II records, search space, and manifest.")
    parser.add_argument("--supernet_checkpoint", default="runs/stage2/supernet_backbone_stage2.pt", help="Stage2 supernet checkpoint used to initialize subnet backbones.")
    parser.add_argument("--population_size", type=int, default=6, help="NSGA-II population size.")
    parser.add_argument("--generations", type=int, default=3, help="Number of NSGA-II generations to evaluate.")
    parser.add_argument("--mutation_prob", type=float, default=0.2, help="Per-gene random-reset mutation probability.")
    parser.add_argument("--crossover_prob", type=float, default=1.0, help="Pair-level uniform crossover probability.")
    parser.add_argument("--candidate_timesteps", type=int, default=1024, help="PPO finetune timesteps for each subnet candidate.")
    parser.add_argument("--eval_episodes", type=int, default=3, help="Evaluation episodes used to estimate candidate return.")
    parser.add_argument("--eval_workers", type=int, default=1, help="Torch multiprocessing workers for parallel subnet evaluation.")
    parser.set_defaults(include_max_initial=True, include_min_initial=True)
    parser.add_argument("--include_max_initial", dest="include_max_initial", action="store_true", help="Seed the initial population with the max architecture gene.")
    parser.add_argument("--no_include_max_initial", dest="include_max_initial", action="store_false", help="Do not force the max architecture gene into the initial population.")
    parser.add_argument("--include_min_initial", dest="include_min_initial", action="store_true", help="Seed the initial population with the min architecture gene.")
    parser.add_argument("--no_include_min_initial", dest="include_min_initial", action="store_false", help="Do not force the min architecture gene into the initial population.")
    parser.add_argument("--initial_genes_json", default="", help="Optional JSON file containing an initial list of integer genes.")
    parser.add_argument("--supernet_backbone_lr", type=float, default=0.0, help="Backbone learning rate during candidate PPO finetune; <=0 freezes the backbone.")
    parser.add_argument("--save_full_history", action="store_true", help="Store full EvoX monitor history in memory for debugging.")
    args = parser.parse_args()
    load_ppo_config(args)
    args.eval_call_seed_stride = 10_000
    args.candidate_seed_stride = 100
    args.eval_seed_offset = 50
    args.mp_start_method = "spawn"
    args.worker_torch_threads = 1
    return args

def load_initial_genes(path: str) -> list[list[int]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("initial genes JSON must contain a list.")
    return [[int(value) for value in gene] for gene in payload]


def build_initial_population(args: argparse.Namespace, codec: GeneCodec, rng: random.Random) -> list[list[int]]:
    if args.population_size <= 0:
        raise ValueError("population_size must be positive.")
    population: list[list[int]] = []
    for gene in load_initial_genes(args.initial_genes_json):
        codec.validate_gene(gene)
        population.append(gene)
    if args.include_max_initial:
        population.append(codec.max_gene())
    if args.include_min_initial:
        population.append(codec.min_gene())
    while len(population) < args.population_size:
        population.append(codec.sample_gene(rng))
    return population[: args.population_size]


def tensor_to_genes(pop: torch.Tensor) -> list[list[int]]:
    return [[int(round(value)) for value in row.tolist()] for row in pop.detach().cpu()]


def build_generation_records(
    generation: int,
    pop: torch.Tensor,
    fit: torch.Tensor,
    codec: GeneCodec,
    problem: RLSubnetProblem,
) -> list[dict[str, Any]]:
    genes = tensor_to_genes(pop)
    fit_cpu = fit.detach().cpu()
    rank = non_dominate_rank(fit_cpu)
    records = []
    worker_records = {tuple(record["gene"]): record for record in problem.last_records}
    for index, gene in enumerate(genes):
        worker_record = worker_records.get(tuple(gene), {})
        objectives = [float(value) for value in fit_cpu[index].tolist()]
        records.append(
            {
                "generation": generation,
                "candidate_index": index,
                "gene": gene,
                "arch": codec.gene_to_arch(gene).to_dict(),
                "objectives": {
                    "negative_return": objectives[0],
                    "params": objectives[1],
                },
                "return": -objectives[0],
                "params": objectives[1],
                "pareto_rank": int(rank[index].item()),
                "is_pareto_front": bool(rank[index].item() == 0),
                "worker_record": worker_record,
            }
        )
    return records


def write_generation(records_path: Path, records: list[dict[str, Any]]) -> None:
    with records_path.open("a") as records_file:
        for record in records:
            records_file.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_space = SearchSpace()
    codec = GeneCodec(search_space)
    (output_dir / "search_space.json").write_text(json.dumps(search_space.to_dict(), indent=2))

    lower_bounds, upper_bounds = codec.gene_bounds()
    initial_population = build_initial_population(args, codec, rng)
    algorithm = DiscreteNSGA2(
        pop_size=args.population_size,
        n_objs=2,
        lb=torch.tensor(lower_bounds, dtype=torch.float32),
        ub=torch.tensor(upper_bounds, dtype=torch.float32),
        device=torch.device("cpu"),
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        initial_population=torch.tensor(initial_population, dtype=torch.float32),
    )
    problem = RLSubnetProblem(args=args, codec=codec)
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
    if records_path.exists():
        records_path.unlink()

    all_records: list[dict[str, Any]] = []
    try:
        workflow.init_step()
        latest_pop = monitor.get_latest_solution()
        latest_fit = monitor.get_latest_fitness()
        records = build_generation_records(0, latest_pop, latest_fit, codec, problem)
        write_generation(records_path, records)
        all_records.extend(records)

        for generation in range(1, args.generations):
            workflow.step()
            latest_pop = monitor.get_latest_solution()
            latest_fit = monitor.get_latest_fitness()
            records = build_generation_records(generation, latest_pop, latest_fit, codec, problem)
            write_generation(records_path, records)
            all_records.extend(records)
    finally:
        problem.close()

    final_pop = monitor.get_latest_solution().detach().cpu()
    final_fit = monitor.get_latest_fitness().detach().cpu()
    final_records = build_generation_records(max(0, args.generations - 1), final_pop, final_fit, codec, problem)
    pareto_records = [record for record in final_records if record["is_pareto_front"]]
    manifest = {
        "records": str(records_path),
        "search_space": str(output_dir / "search_space.json"),
        "objectives": ["negative_return", "params"],
        "pareto_front": pareto_records,
        "final_population": final_records,
        "num_logged_records": len(all_records),
        "cache_size": len(problem.cache),
        "args": vars(args),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
