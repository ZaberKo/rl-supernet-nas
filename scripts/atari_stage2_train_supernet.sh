#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage2_train_supernet.py \
  --trajectory_data runs/atari_space_invaders/stage1_mix/representation_data.arrow \
  --stage1_backbone runs/atari_space_invaders/stage1_ppo_max/supernet_backbone_stage1.pt \
  --output_dir runs/atari_space_invaders/stage2 \
  --train_steps 5000 \
  --batch_size 64 \
  --random_subnets 2 \
  --dynamics_horizon 3 \
  --dynamics_betas 1.0,0.5,0.25 \
  --projection_dim 128 \
  --predictor_hidden_dim 512 \
  --backbone_learning_rate 0.00003 \
  --head_learning_rate 0.0001
