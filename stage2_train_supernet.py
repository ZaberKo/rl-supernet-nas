from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config_utils import add_ppo_config_args, load_ppo_config
from env_utils import get_vision_spaces
from ppo_utils import parse_optional_float, resolve_device, use_render_observation
from representation_losses import (
    LatentDynamicsPredictor,
    ProjectionHead,
    cosine_kd_loss,
    encode_action_batch,
    get_action_dim,
    latent_dynamics_loss,
)
from supernet_backbone import SearchSpace, SupernetCNNBackbone, infer_input_channels, load_backbone_checkpoint
from trajectory_data import TransitionDataset
from wandb_utils import finish_wandb_run, init_wandb_run, log_wandb, log_wandb_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 supernet representation learning.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument("--trajectory_data", default="runs/stage1_mix/mixed_trajectories.arrow", help="Stage1 mixed PPO+random Arrow dataset used for one-step representation learning.")
    parser.add_argument("--output_dir", default="runs/stage2", help="Directory for stage2 checkpoints, metrics, and manifest.")
    parser.add_argument("--stage1_backbone", default="runs/stage1_ppo_max_arch/supernet_backbone_stage1.pt", help="Backbone checkpoint inherited from stage1 PPO training; use an empty string to train from scratch.")
    parser.add_argument("--train_steps", type=int, default=1000, help="Number of optimizer updates. When <= 0, the budget is epochs multiplied by DataLoader length.")
    parser.add_argument("--epochs", type=int, default=0, help="Fallback epoch budget used only when train_steps <= 0.")
    parser.add_argument("--batch_size", type=int, default=64, help="One-step transition batch size for stage2 DataLoader.")
    parser.add_argument("--random_subnets", type=int, default=2, help="Number of random student subnets sampled per batch, in addition to the min arch.")
    parser.add_argument("--projection_dim", type=int, default=128, help="Latent projection dimension used by KD and dynamics losses.")
    parser.add_argument("--predictor_hidden_dim", type=int, default=512, help="Hidden dimension for the action-conditioned one-step latent dynamics predictor.")
    parser.add_argument("--backbone_learning_rate", type=float, default=3e-5, help="AdamW learning rate for inherited backbone parameters.")
    parser.add_argument("--head_learning_rate", type=float, default=1e-4, help="AdamW learning rate for the projection head and dynamics predictor.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay for stage2 representation learning.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Fraction of optimizer steps used for linear warmup before cosine decay.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Final learning-rate multiplier at the end of cosine decay.")
    parser.add_argument("--gradient_clip_norm", type=parse_optional_float, default=None, help="Optional gradient clipping norm; use none/null/off to disable.")
    parser.add_argument("--dyn_weight", type=float, default=1.0, help="Weight for one-step latent dynamics prediction loss.")
    parser.add_argument("--kd_weight", type=float, default=1.0, help="Weight for cosine latent distillation loss.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count for shuffled one-step transitions.")
    parser.add_argument("--save_every_steps", type=int, default=0, help="Save an extra checkpoint every N optimizer steps; 0 disables periodic checkpoints.")
    args = parser.parse_args()
    load_ppo_config(args)
    return args


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


def iterate_batches(loader: DataLoader) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    while True:
        for batch_index, batch in enumerate(loader):
            yield batch_index, batch


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
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
            "features_dim": args.features_dim,
            "projection_dim": args.projection_dim,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb_run("stage2_train_supernet", args, output_dir)

    search_space = SearchSpace()
    (output_dir / "search_space.json").write_text(json.dumps(search_space.to_dict(), indent=2))

    observation_space, action_space = get_vision_spaces(
        env_id=args.env_id,
        seed=args.seed,
        image_size=args.image_size,
        use_render_observation=use_render_observation(args),
        vector_env_type=getattr(args, "vector_env_type", "dummy"),
    )
    input_channels = infer_input_channels(tuple(observation_space.shape))
    backbone = SupernetCNNBackbone(
        input_channels=input_channels,
        search_space=search_space,
        feature_dim=args.features_dim,
    ).to(device)
    if args.stage1_backbone:
        stage1_backbone_path = Path(args.stage1_backbone)
        if not stage1_backbone_path.exists():
            raise FileNotFoundError(f"Stage1 backbone checkpoint does not exist: {stage1_backbone_path}")
        load_backbone_checkpoint(backbone, stage1_backbone_path, map_location=device)

    projection = ProjectionHead(args.features_dim, args.projection_dim).to(device)
    predictor = LatentDynamicsPredictor(
        latent_dim=args.projection_dim,
        action_dim=get_action_dim(action_space),
        hidden_dim=args.predictor_hidden_dim,
    ).to(device)
    optimizer = build_optimizer(args, backbone, projection, predictor)

    dataset = TransitionDataset([args.trajectory_data])
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
    log_wandb(
        wandb_run,
        {
            "stage2/num_transitions": len(dataset),
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
            actions = batch["action"].to(device)
            target = batch["target"].to(device)
            action_features = encode_action_batch(actions, action_space).to(device)

            backbone.set_max_arch()
            with torch.no_grad():
                teacher_start = F.normalize(projection(backbone(observation)), dim=-1)
                teacher_target = F.normalize(projection(backbone(target)), dim=-1)

            sampled_arches = [search_space.min_arch()]
            for _ in range(args.random_subnets):
                sampled_arches.append(search_space.sample_arch())

            total_loss = torch.zeros((), device=device)
            dyn_value = torch.zeros((), device=device)
            kd_value = torch.zeros((), device=device)
            for arch in sampled_arches:
                backbone.set_active_arch(arch)
                student_start = F.normalize(projection(backbone(observation)), dim=-1)
                predictions = predictor(student_start, action_features)
                dyn_loss = latent_dynamics_loss(predictions, teacher_target)

                student_target = F.normalize(projection(backbone(target)), dim=-1)
                kd_loss = cosine_kd_loss(student_start, teacher_start)
                kd_loss = kd_loss + cosine_kd_loss(student_target, teacher_target)
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
                },
                step=train_step,
            )

            if args.save_every_steps > 0 and train_step % args.save_every_steps == 0:
                save_checkpoint(
                    output_dir / f"supernet_backbone_stage2_step{train_step:06d}.pt",
                    args,
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
        "num_transitions": len(dataset),
        "total_steps": total_steps,
        "args": vars(args),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log_wandb(
        wandb_run,
        {
            "stage2/num_logged_steps": total_steps,
            "stage2/num_transitions": len(dataset),
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
