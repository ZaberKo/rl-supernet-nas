#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage3_ea_search.py \
  --supernet_checkpoint runs/atari_pong/stage2/supernet_backbone_stage2.pt \
  --output_dir runs/atari_pong/stage3 \
  --population_size 6 \
  --generations 3 \
  --candidate_timesteps 10000 \
  --eval_episodes 3 \
  --eval_workers 1 \
  --supernet_backbone_lr 0.0
