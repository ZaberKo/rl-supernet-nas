#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_eval_archs.py \
  --ppo_config config.yaml \
  --supernet_checkpoint runs/atari_space_invaders/stage1_policy_supernet/policy_supernet_best.pt \
  --arch_configs arch_configs/random_archs.json \
  --output_dir runs/atari_space_invaders/stage1_eval_archs2 \
  --workers 4
