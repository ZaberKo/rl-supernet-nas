#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_train_arch_ppo.py \
  --ppo_config config_box2d.yaml \
  --supernet_checkpoint runs/box2d_carracing/stage1_policy_supernet/policy_supernet_best.pt \
  --arch_config arch_configs/max_arch.json \
  --output_dir runs/box2d_carracing/stage1_arch_ppo_max_arch \
  --critic_warmup_timesteps 0 \
  --ppo_config_override ppo.eval_episodes=3 ppo.total_timesteps=10000 ppo.policy_backbone_lr=0.0
