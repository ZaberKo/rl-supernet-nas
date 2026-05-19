from __future__ import annotations


import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config_utils import add_ppo_config_args, load_ppo_config
from env_utils import get_vision_spaces
from ppo_utils import parse_optional_float, resolve_device, use_render_observation
from representation_losses import (
    CosineKDLoss,
    LatentDynamicsLoss,
    LatentDynamicsPredictor,
    ProjectionHead,
    encode_action_sequence,
    get_action_dim,
)
from supernet_backbone import SearchSpace, SupernetCNNBackbone, infer_input_channels, load_backbone_checkpoint
from trajectory_data import TrajectoryDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 supernet representation learning.",
        allow_abbrev=False,
    )
    add_ppo_config_args(parser)
    parser.add_argument(
        "--trajectory_files",
        nargs="+",
        default=[
            "runs/stage1_ppo_max_arch/ppo_train_trajectories.arrow",
            "runs/stage1_mix/random_trajectories.arrow",
        ],
        help="Trajectory dataset paths used for representation learning when no manifest is provided.",
    )
    parser.add_argument("--trajectory_manifest", default="", help="Stage1 mixed-data manifest; overrides trajectory_files when provided.")
    parser.add_argument("--output_dir", default="runs/stage2", help="Directory for stage2 checkpoints, metrics, and manifest.")
    parser.add_argument("--stage1_backbone", default="runs/stage1_ppo_max_arch/supernet_backbone_stage1.pt", help="Backbone checkpoint inherited from stage1 PPO training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of supervised representation-learning epochs.")
    parser.add_argument("--max_batches_per_epoch", type=int, default=0, help="Debug limit per epoch; 0 means use all batches.")
    parser.add_argument("--batch_size", type=int, default=32, help="Trajectory-window batch size for stage2 DataLoader.")
    parser.add_argument("--horizon", type=int, default=3, help="Number of future transitions predicted by the latent dynamics loss.")
    parser.add_argument("--random_subnets", type=int, default=2, help="Number of random student subnets sampled per batch, in addition to the min arch.")
    parser.add_argument("--projection_dim", type=int, default=128, help="Latent projection dimension used by KD and dynamics losses.")
    parser.add_argument("--predictor_hidden_dim", type=int, default=512, help="Hidden dimension for the action-conditioned latent dynamics predictor.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="AdamW learning rate for backbone, projection head, and dynamics predictor.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay for stage2 representation learning.")
    parser.add_argument("--gradient_clip_norm", type=parse_optional_float, default=None, help="Optional gradient clipping norm; use none/null/off to disable.")
    parser.add_argument("--dyn_weight", type=float, default=1.0, help="Weight for latent dynamics prediction loss.")
    parser.add_argument("--kd_weight", type=float, default=1.0, help="Weight for cosine latent distillation loss.")
    parser.add_argument("--beta", default="", help="Comma-separated horizon weights for dynamics loss; empty means all ones.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count for trajectory windows.")
    parser.add_argument("--save_every_epochs", type=int, default=0, help="Save an extra checkpoint every N epochs; 0 disables periodic checkpoints.")
    args = parser.parse_args()
    load_ppo_config(args)
    return args

def parse_beta(beta_value, horizon: int) -> list[float]:
    if not beta_value:
        return [1.0] * horizon
    if isinstance(beta_value, (list, tuple)):
        values = [float(value) for value in beta_value]
    else:
        values = [float(value.strip()) for value in str(beta_value).split(",") if value.strip()]
    if len(values) == 1:
        return values * horizon
    if len(values) != horizon:
        raise ValueError("--beta must contain one value or exactly horizon values.")
    return values


def build_optimizer(args: argparse.Namespace, parameters) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    backbone: SupernetCNNBackbone,
    projection: ProjectionHead,
    predictor: LatentDynamicsPredictor,
    optimizer: torch.optim.Optimizer,
    search_space,
    epoch: int,
) -> None:
    torch.save(
        {
            "stage": "stage2",
            "epoch": epoch,
            "backbone_state_dict": backbone.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "search_space": search_space.to_dict(),
            "features_dim": args.features_dim,
            "projection_dim": args.projection_dim,
            "horizon": args.horizon,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.trajectory_manifest:
        manifest = json.loads(Path(args.trajectory_manifest).read_text())
        args.trajectory_files = manifest["trajectory_files"]

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
    if args.stage1_backbone and Path(args.stage1_backbone).exists():
        load_backbone_checkpoint(backbone, args.stage1_backbone, map_location=device)

    projection = ProjectionHead(args.features_dim, args.projection_dim).to(device)
    predictor = LatentDynamicsPredictor(
        latent_dim=args.projection_dim,
        action_dim=get_action_dim(action_space),
        horizon=args.horizon,
        hidden_dim=args.predictor_hidden_dim,
    ).to(device)
    dynamics_loss_fn = LatentDynamicsLoss(beta=parse_beta(args.beta, args.horizon)).to(device)
    kd_loss_fn = CosineKDLoss().to(device)
    optimizer = build_optimizer(
        args,
        list(backbone.parameters()) + list(projection.parameters()) + list(predictor.parameters()),
    )

    dataset = TrajectoryDataset(args.trajectory_files, horizon=args.horizon)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w") as metrics_file:
        global_step = 0
        for epoch in range(args.epochs):
            for batch_index, batch in enumerate(loader):
                if args.max_batches_per_epoch > 0 and batch_index >= args.max_batches_per_epoch:
                    break

                observation = batch["observation"].to(device)
                actions = batch["actions"].to(device)
                targets = batch["targets"].to(device)
                batch_size, horizon = targets.shape[:2]
                action_features = encode_action_sequence(actions, action_space).to(device)
                flat_targets = targets.reshape(batch_size * horizon, *targets.shape[2:])

                backbone.set_max_arch()
                with torch.no_grad():
                    teacher_start = F.normalize(projection(backbone(observation)), dim=-1)
                    teacher_targets = F.normalize(
                        projection(backbone(flat_targets)).reshape(batch_size, horizon, -1),
                        dim=-1,
                    )

                sampled_arches = [search_space.min_arch()]
                for _ in range(args.random_subnets):
                    sampled_arches.append(search_space.sample_arch(rng))

                total_loss = torch.zeros((), device=device)
                dyn_value = torch.zeros((), device=device)
                kd_value = torch.zeros((), device=device)
                for arch in sampled_arches:
                    backbone.set_active_arch(arch)
                    student_start = F.normalize(projection(backbone(observation)), dim=-1)
                    predictions = predictor(student_start, action_features)
                    dyn_loss = dynamics_loss_fn(predictions, teacher_targets)

                    student_targets = F.normalize(
                        projection(backbone(flat_targets)).reshape(batch_size, horizon, -1),
                        dim=-1,
                    )
                    kd_loss = kd_loss_fn(student_start, teacher_start)
                    kd_loss = kd_loss + kd_loss_fn(
                        student_targets.reshape(batch_size * horizon, -1),
                        teacher_targets.reshape(batch_size * horizon, -1),
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
                    torch.nn.utils.clip_grad_norm_(
                        list(backbone.parameters())
                        + list(projection.parameters())
                        + list(predictor.parameters()),
                        args.gradient_clip_norm,
                    )
                optimizer.step()

                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "batch_index": batch_index,
                    "loss": float(total_loss.detach().cpu()),
                    "dynamics_loss": float(dyn_value.cpu()),
                    "kd_loss": float(kd_value.cpu()),
                    "num_student_subnets": len(sampled_arches),
                    "student_arches": [arch.to_dict() for arch in sampled_arches],
                }
                metrics_file.write(json.dumps(record) + "\n")
                global_step += 1

            if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                save_checkpoint(
                    output_dir / f"supernet_backbone_stage2_epoch{epoch + 1:03d}.pt",
                    args,
                    backbone,
                    projection,
                    predictor,
                    optimizer,
                    search_space,
                    epoch,
                )

    checkpoint_path = output_dir / "supernet_backbone_stage2.pt"
    save_checkpoint(
        checkpoint_path,
        args,
        backbone,
        projection,
        predictor,
        optimizer,
        search_space,
        args.epochs,
    )
    manifest = {
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "search_space": str(output_dir / "search_space.json"),
        "num_windows": len(dataset),
        "args": vars(args),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
