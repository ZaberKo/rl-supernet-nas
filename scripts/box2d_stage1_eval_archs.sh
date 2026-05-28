#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_eval_archs.py \
  --ppo_config config_box2d.yaml \
  --supernet_checkpoint runs/box2d_carracing/stage1_policy_supernet/policy_supernet_best.pt \
  --arch_configs arch_configs/random_archs.json \
  --output_dir runs/box2d_carracing/stage1_eval_archs \
  --ppo_config_override ppo.eval_episodes=3
