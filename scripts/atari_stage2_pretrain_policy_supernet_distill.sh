#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source .venv/bin/activate

python stage2_pretrain_policy_supernet_distill.py \
  --ppo_config config.yaml \
  --ppo_config_override ppo.n_epochs=4 \
  --ppo_config_override ppo.total_timesteps=30000000 \
  --ppo_config_override ppo.policy_backbone_lr=lin_2.5e-4 \
  --ppo_config_override ppo.policy_head_lr=lin_1e-4 \
  --ppo_config_override ppo.z_dyn_coef=0 \
  --ppo_config_override ppo.ema_tau=0.99 \
  --teacher_checkpoint runs/atari_space_invaders/stage1_max_subnet_ppo/max_subnet_ppo_best.pt \
  --output_dir runs/atari_space_invaders/stage2_policy_supernet_distill_pretrain \
  --random_subnets 2 \
  --distill_temperature 2.0
