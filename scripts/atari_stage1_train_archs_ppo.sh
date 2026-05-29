#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_train_archs_ppo.py \
  --ppo_config config.yaml \
  --supernet_checkpoint runs/atari_space_invaders/stage1_policy_supernet/policy_supernet_best.pt \
  --arch_configs arch_configs/random_archs.json \
  --output_dir runs/atari_space_invaders/stage1_train_archs_ppo \
  --workers 2 \
  --critic_warmup_timesteps 0 \
  --ppo_config_override ppo.n_epochs=4 \
  --ppo_config_override ppo.total_timesteps=1000000 \
  --ppo_config_override ppo.z_dyn_coef=0.1 \
  --ppo_config_override ppo.ema_tau=0.99 \
  --ppo_config_override ppo.projection_dim=128 \
  --ppo_config_override ppo.predictor_hidden_dim=512 \
  --ppo_config_override ppo.policy_backbone_lr=0 \
  --ppo_config_override ppo.policy_head_lr=lin_1e-4 \
  --ppo_config_override ppo.critic_lr=lin_2.5e-4 \
  --ppo_config_override ppo.clip_range=lin_0.1
