#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python new_stage1_train_policy_supernet.py \
  --ppo_config config.yaml \
  --ppo_config_override ppo.n_epochs=4 \
  --ppo_config_override ppo.total_timesteps=30000000 \
  --ppo_config_override ppo.z_dyn_coef=0.01 \
  --ppo_config_override ppo.ema_tau=0.99 \
  --ppo_config_override ppo.projection_dim=128 \
  --ppo_config_override ppo.predictor_hidden_dim=512 \
  --output_dir runs/atari_space_invaders/new_stage1_policy_supernet \
  --random_subnets 2 \
  --distill_temperature 2.0
