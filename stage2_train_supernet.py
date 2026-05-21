from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from config_utils import add_ppo_config_args, build_run_config, load_ppo_config, ppo_config_to_dict
from ppo_utils import get_vision_spaces_from_ppo_config, parse_optional_float, resolve_device
from representation_losses import (
    LatentDynamicsPredictor,
    ProjectionHead,
    cosine_kd_loss,
    encode_action_batch,
    get_action_dim,
    latent_dynamics_loss,
)
from supernet_backbone import SearchSpace, SupernetCNNBackbone, infer_input_channels, load_backbone_from_policy_checkpoint
from trajectory_data import TransitionDataset
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 supernet representation learning.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--trajectory_data", default="runs/stage1_mix/representation_data.arrow", help="Stage1 supervised transition dataset used for representation learning.")
    parser.add_argument("--output_dir", default="runs/stage2", help="Directory for stage2 checkpoints, metrics, and manifest.")
    parser.add_argument("--stage1_backbone", default="runs/stage1_ppo_max_arch/ppo_supernet_stage1.pt", help="Stage1 PPO policy checkpoint; use an empty string to train from scratch.")
    parser.add_argument("--train_steps", type=int, default=1000, help="Number of optimizer updates. When <= 0, the budget is epochs multiplied by DataLoader length.")
    parser.add_argument("--epochs", type=int, default=0, help="Fallback epoch budget used only when train_steps <= 0.")
    parser.add_argument("--batch_size", type=int, default=64, help="Transition sequence batch size for stage2 DataLoader.")
    parser.add_argument("--random_subnets", type=int, default=2, help="Number of random student subnets sampled per batch, in addition to the min arch.")
    parser.add_argument("--projection_dim", type=int, default=128, help="Latent projection dimension used by KD and dynamics losses.")
    parser.add_argument("--predictor_hidden_dim", type=int, default=512, help="Hidden dimension for the action-conditioned latent dynamics predictor.")
    parser.add_argument("--dynamics_horizon", type=int, default=0, help="Optional expected horizon check; 0 infers horizon from the dataset.")
    parser.add_argument("--dynamics_betas", default="", help="Comma-separated beta weights for each dynamics horizon step; empty uses all ones.")
    parser.add_argument("--backbone_learning_rate", type=float, default=3e-5, help="AdamW learning rate for inherited backbone parameters.")
    parser.add_argument("--head_learning_rate", type=float, default=1e-4, help="AdamW learning rate for the projection head and dynamics predictor.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay for stage2 representation learning.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Fraction of optimizer steps used for linear warmup before cosine decay.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Final learning-rate multiplier at the end of cosine decay.")
    parser.add_argument("--gradient_clip_norm", type=parse_optional_float, default=None, help="Optional gradient clipping norm; use none/null/off to disable.")
    parser.add_argument("--dyn_weight", type=float, default=1.0, help="Weight for latent dynamics prediction loss.")
    parser.add_argument("--kd_weight", type=float, default=1.0, help="Weight for cosine latent distillation loss.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count for shuffled transition sequences.")
    parser.add_argument("--save_every_steps", type=int, default=0, help="Save an extra checkpoint every N optimizer steps; 0 disables periodic checkpoints.")
    return parser.parse_args()


def build_optimizer(
    args: argparse.Namespace,
    backbone: SupernetCNNBackbone,
    projection: ProjectionHead,
    predictor: LatentDynamicsPredictor,
) -> torch.optim.Optimizer:
    head_parameters = list(projection.parameters()) + list(predictor.parameters())
    return torch.optim.AdamW(
        [
            {"params": list(backbone.parameters()), "lr": args.backbone_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )


def build_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1].")
    warmup_steps = int(round(total_steps * args.warmup_ratio))

    def lr_lambda(step_index: int) -> float:
        if total_steps <= 1:
            return 1.0
        current_step = min(step_index + 1, total_steps)
        if warmup_steps > 0 and current_step <= warmup_steps:
            return current_step / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (current_step - warmup_steps) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_total_steps(args: argparse.Namespace, steps_per_epoch: int) -> int:
    if args.train_steps > 0:
        return int(args.train_steps)
    if args.epochs <= 0:
        raise ValueError("Either train_steps must be positive or epochs must be positive.")
    return int(args.epochs * steps_per_epoch)


def resolve_dynamics_betas(text: str, horizon: int, device: torch.device) -> torch.Tensor:
    if horizon <= 0:
        raise ValueError("dynamics_horizon must be positive.")
    if not text:
        return torch.ones(horizon, dtype=torch.float32, device=device)
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) != horizon:
        raise ValueError("dynamics_betas must contain exactly dynamics_horizon values.")
    return torch.tensor(values, dtype=torch.float32, device=device)


def valid_steps_from_done_signals(dones: torch.Tensor, terminateds: torch.Tensor) -> torch.Tensor:
    if dones.dim() != 2 or terminateds.dim() != 2:
        raise ValueError("dones and terminateds must have shape [batch, horizon].")
    if dones.shape != terminateds.shape:
        raise ValueError("dones and terminateds must have the same shape.")
    done_values = dones.to(torch.int64)
    previous_done_count = torch.cumsum(done_values, dim=1) - done_values
    return ((previous_done_count == 0) & ~terminateds).to(torch.float32)


def iterate_batches(loader: DataLoader) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    while True:
        for batch_index, batch in enumerate(loader):
            yield batch_index, batch


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    ppo_config: DictConfig,
    backbone: SupernetCNNBackbone,
    projection: ProjectionHead,
    predictor: LatentDynamicsPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    search_space,
    global_step: int,
    total_steps: int,
) -> None:
    torch.save(
        {
            "stage": "stage2",
            "global_step": global_step,
            "total_steps": total_steps,
            "backbone_state_dict": backbone.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "search_space": search_space.to_dict(),
            "features_dim": ppo_config.features_dim,
            "projection_dim": args.projection_dim,
            "args": vars(args),
            "ppo_config": ppo_config_to_dict(ppo_config),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    ppo_config = load_ppo_config(args)
    torch.manual_seed(ppo_config.seed)
    device = resolve_device(ppo_config.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args, ppo_config)
    wandb_run = init_wandb_run("stage2_train_supernet", run_config, output_dir)

    search_space = SearchSpace()
    (output_dir / "search_space.json").write_text(json.dumps(search_space.to_dict(), indent=2))

    observation_space, action_space = get_vision_spaces_from_ppo_config(ppo_config, seed=ppo_config.seed)
    input_channels = infer_input_channels(tuple(observation_space.shape))
    backbone = SupernetCNNBackbone(
        input_channels=input_channels,
        search_space=search_space,
        feature_dim=ppo_config.features_dim,
    ).to(device)
    if args.stage1_backbone:
        stage1_backbone_path = Path(args.stage1_backbone)
        if not stage1_backbone_path.exists():
            raise FileNotFoundError(f"Stage1 checkpoint does not exist: {stage1_backbone_path}")
        load_backbone_from_policy_checkpoint(backbone, stage1_backbone_path, map_location=device)

    projection = ProjectionHead(ppo_config.features_dim, args.projection_dim).to(device)
    predictor = LatentDynamicsPredictor(
        latent_dim=args.projection_dim,
        action_dim=get_action_dim(action_space),
        hidden_dim=args.predictor_hidden_dim,
    ).to(device)
    optimizer = build_optimizer(args, backbone, projection, predictor)

    dataset = TransitionDataset([args.trajectory_data])
    if args.dynamics_horizon > 0 and args.dynamics_horizon != dataset.horizon:
        raise ValueError("dynamics_horizon does not match the dataset horizon.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    steps_per_epoch = len(loader)
    total_steps = resolve_total_steps(args, steps_per_epoch)
    scheduler = build_scheduler(args, optimizer, total_steps)
    beta_weights = resolve_dynamics_betas(args.dynamics_betas, dataset.horizon, device)
    log_wandb(
        wandb_run,
        {
            "stage2/num_samples": len(dataset),
            "stage2/dataset_horizon": dataset.horizon,
            "stage2/total_steps": total_steps,
        },
        step=0,
    )

    train_batches = iterate_batches(loader)
    metrics_path = output_dir / "metrics.jsonl"
    trainable_parameters = list(backbone.parameters()) + list(projection.parameters()) + list(predictor.parameters())
    with metrics_path.open("w") as metrics_file:
        for step_index in range(total_steps):
            batch_index, batch = next(train_batches)
            train_step = step_index + 1
            epoch = step_index // steps_per_epoch

            observation = batch["observation"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["targets"].to(device)
            dones = batch["dones"].to(device)
            terminateds = batch["terminateds"].to(device)
            batch_size, horizon = actions.shape[:2]
            flat_actions = actions.reshape(batch_size * horizon, *actions.shape[2:])
            action_features = encode_action_batch(flat_actions, action_space).to(device)
            action_features = action_features.view(batch_size, horizon, -1)
            flat_targets = targets.reshape(batch_size * horizon, *targets.shape[2:])
            valid_steps = valid_steps_from_done_signals(dones, terminateds).to(device)
            flat_valid_steps = valid_steps.reshape(batch_size * horizon)

            backbone.set_max_arch()
            with torch.no_grad():
                teacher_start = F.normalize(projection(backbone(observation)), dim=-1)
                teacher_targets = F.normalize(projection(backbone(flat_targets)), dim=-1)
                teacher_targets = teacher_targets.view(batch_size, horizon, -1)

            sampled_arches = [search_space.min_arch()]
            for _ in range(args.random_subnets):
                sampled_arches.append(search_space.sample_arch())

            total_loss = torch.zeros((), device=device)
            dyn_value = torch.zeros((), device=device)
            kd_value = torch.zeros((), device=device)
            for arch in sampled_arches:
                backbone.set_sample_config(arch)
                student_start = F.normalize(projection(backbone(observation)), dim=-1)
                predictions = predictor(student_start, action_features)
                dyn_loss = latent_dynamics_loss(predictions, teacher_targets, beta_weights, sample_weights=valid_steps)

                student_targets = F.normalize(projection(backbone(flat_targets)), dim=-1)
                student_targets = student_targets.view(batch_size, horizon, -1)
                kd_loss = cosine_kd_loss(student_start, teacher_start)
                kd_loss = kd_loss + cosine_kd_loss(
                    student_targets.flatten(0, 1),
                    teacher_targets.flatten(0, 1),
                    sample_weights=flat_valid_steps,
                )
                total_loss = total_loss + args.dyn_weight * dyn_loss + args.kd_weight * kd_loss
                dyn_value = dyn_value + dyn_loss.detach()
                kd_value = kd_value + kd_loss.detach()

            total_loss = total_loss / len(sampled_arches)
            dyn_value = dyn_value / len(sampled_arches)
            kd_value = kd_value / len(sampled_arches)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.gradient_clip_norm is not None and args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            backbone_lr = float(optimizer.param_groups[0]["lr"])
            head_lr = float(optimizer.param_groups[1]["lr"])
            record = {
                "step": train_step,
                "epoch": epoch,
                "batch_index": batch_index,
                "loss": float(total_loss.detach().cpu()),
                "dynamics_loss": float(dyn_value.cpu()),
                "kd_loss": float(kd_value.cpu()),
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "num_student_subnets": len(sampled_arches),
                "dynamics_horizon": int(dataset.horizon),
                "dynamics_betas": [float(value) for value in beta_weights.detach().cpu().tolist()],
                "valid_horizon_fraction": float(valid_steps.detach().mean().cpu()),
                "student_arches": [arch.to_dict() for arch in sampled_arches],
            }
            metrics_file.write(json.dumps(record) + "\n")
            log_wandb(
                wandb_run,
                {
                    "stage2/loss": record["loss"],
                    "stage2/dynamics_loss": record["dynamics_loss"],
                    "stage2/kd_loss": record["kd_loss"],
                    "stage2/epoch": epoch,
                    "stage2/batch_index": batch_index,
                    "stage2/backbone_lr": backbone_lr,
                    "stage2/head_lr": head_lr,
                    "stage2/num_student_subnets": len(sampled_arches),
                    "stage2/dynamics_horizon": int(dataset.horizon),
                    "stage2/valid_horizon_fraction": record["valid_horizon_fraction"],
                },
                step=train_step,
            )

            if args.save_every_steps > 0 and train_step % args.save_every_steps == 0:
                save_checkpoint(
                    output_dir / f"supernet_backbone_stage2_step{train_step:06d}.pt",
                    args,
                    ppo_config,
                    backbone,
                    projection,
                    predictor,
                    optimizer,
                    scheduler,
                    search_space,
                    train_step,
                    total_steps,
                )

    checkpoint_path = output_dir / "supernet_backbone_stage2.pt"
    save_checkpoint(
        checkpoint_path,
        args,
        ppo_config,
        backbone,
        projection,
        predictor,
        optimizer,
        scheduler,
        search_space,
        total_steps,
        total_steps,
    )
    manifest = {
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "search_space": str(output_dir / "search_space.json"),
        "trajectory_data": args.trajectory_data,
        "num_samples": len(dataset),
        "dataset_horizon": dataset.horizon,
        "total_steps": total_steps,
        "args": vars(args),
        "ppo_config": ppo_config_to_dict(ppo_config),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log_wandb(
        wandb_run,
        {
            "stage2/num_logged_steps": total_steps,
            "stage2/num_samples": len(dataset),
        },
        step=total_steps,
    )
    log_wandb_artifact(
        wandb_run,
        name=f"stage2-{output_dir.name}",
        artifact_type="stage2-output",
        paths=[checkpoint_path, metrics_path, output_dir / "search_space.json", manifest_path],
    )
    finish_wandb_run(wandb_run)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
