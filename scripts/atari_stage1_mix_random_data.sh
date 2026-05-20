#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_mix_random_data.py \
  --ppo_trajectory_file runs/atari_pong/stage1_ppo_max/ppo_train_trajectories.arrow \
  --output_dir runs/atari_pong/stage1_mix \
  --random_transitions 50176 \
  --representation_horizon 3
