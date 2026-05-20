#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_mix_random_data.py \
  --ppo_trajectory_file runs/atari_space_invaders/stage1_ppo_max/ppo_train_trajectories.arrow \
  --output_dir runs/atari_space_invaders/stage1_mix \
  --random_transitions 10000384 \
  --representation_horizon 3
