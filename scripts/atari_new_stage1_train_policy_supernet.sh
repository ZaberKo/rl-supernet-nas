#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python new_stage1_train_policy_supernet.py \
  --ppo_config config.yaml \
  --output_dir runs/atari_space_invaders/new_stage1_policy_supernet \
  --random_subnets 2 \
  --beta_dyn 0.01 \
  --ema_tau 0.99 \
  --projection_dim 128 \
  --predictor_hidden_dim 512
