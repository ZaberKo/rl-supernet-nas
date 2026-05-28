#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage2_ea_search.py \
  --supernet_checkpoint runs/box2d_carracing/stage1_policy_supernet/policy_supernet_best.pt \
  --output_dir runs/box2d_carracing/stage2_ea_search \
  --population_size 6 \
  --generations 3 \
  --critic_warmup_timesteps 0 \
  --eval_workers 1 \
  --ppo_config_override ppo.total_timesteps=10000 ppo.policy_backbone_lr=0.0
