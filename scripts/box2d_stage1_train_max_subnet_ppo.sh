#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_train_max_subnet_ppo.py \
  --ppo_config config_box2d.yaml \
  --ppo_config_override ppo.n_epochs=4 \
  --ppo_config_override ppo.total_timesteps=4000000 \
  --ppo_config_override ppo.z_dyn_coef=0.1 \
  --ppo_config_override ppo.ema_tau=0.99 \
  --ppo_config_override ppo.projection_dim=128 \
  --ppo_config_override ppo.predictor_hidden_dim=512 \
  --ppo_config_override ppo.policy_backbone_lr=lin_1e-4 \
  --ppo_config_override ppo.policy_head_lr=lin_5e-5 \
  --ppo_config_override ppo.critic_lr=lin_1e-4 \
  --ppo_config_override ppo.clip_range=lin_0.1 \
  --output_dir runs/box2d_carracing/stage1_max_subnet_ppo
