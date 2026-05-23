#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_mix_random_data.py \
  --ppo_data_file runs/atari_space_invaders/stage1_ppo_max/ppo_representation_samples.h5 \
  --output_dir runs/atari_space_invaders/stage1_mix \
  --random_samples 10000 \
  --seed 114514 \
  --horizon 1
