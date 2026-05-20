from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from evox.operators.selection import non_dominate_rank
from evox.workflows import EvalMonitor, StdWorkflow

from config_utils import add_ppo_config_args, build_run_config, load_ppo_config, ppo_config_to_dict
from ea_codec import GeneCodec
from nsga2_search import DiscreteNSGA2, RLSubnetProblem
from supernet_backbone import SearchSpace
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


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
    parser.add_argument("--candidate_timesteps", type=int, default=1024, help="PPO finetune timesteps for each subnet candidate.")
    parser.add_argument("--eval_workers", type=int, default=1, help="Torch multiprocessing workers for parallel subnet evaluation.")
    parser.add_argument("--supernet_backbone_lr", type=float, default=0.0, help="Backbone learning rate during candidate PPO finetune; <=0 freezes the backbone.")
    parser.add_argument("--save_full_history", action="store_true", help="Store full EvoX monitor history in memory for debugging.")
    args = parser.parse_args()
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
    log_wandb(
        wandb_run,
        {f"stage3/{key}": value for key, value in summary.items()},
        step=generation,
    )


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run("stage3_ea_search", run_config, output_dir)
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
    problem = RLSubnetProblem(args=args, ppo_config=ppo_config, codec=codec)
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
        "records": str(records_path),
        "log": str(log_path),
        "search_space": str(output_dir / "search_space.json"),
        "objectives": ["negative_return", "params"],
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
            "stage3/num_logged_records": len(all_records),
            "stage3/cache_size": len(problem.cache),
            "stage3/final_pareto_count": len(pareto_records),
        },
        step=max(0, args.generations - 1),
    )
    log_wandb_artifact(
        wandb_run,
        name=f"stage3-{output_dir.name}",
        artifact_type="stage3-output",
        paths=[records_path, log_path, output_dir / "search_space.json", manifest_path],
    )
    finish_wandb_run(wandb_run)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
