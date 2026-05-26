#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python new_stage1_train_policy_supernet.py \
  --ppo_config config_box2d.yaml \
  --ppo_config_override ppo.n_epochs=10 \
  --ppo_config_override ppo.total_timesteps=12000000 \
  --ppo_config_override ppo.z_dyn_coef=0.1 \
  --ppo_config_override ppo.ema_tau=0.99 \
  --ppo_config_override ppo.projection_dim=128 \
  --ppo_config_override ppo.predictor_hidden_dim=512 \
  --output_dir runs/box2d_carracing/new_stage1_policy_supernet \
  --random_subnets 2 \
  --distill_temperature 1.0
