#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_train_max_ppo.py \
  --output_dir runs/atari_space_invaders/stage1_ppo_max \
  --horizon 1 \
  --sample_ratio 0.01 \
  --ppo_config_override ppo.total_timesteps=10000000 \
  --ppo_config_override ppo.train_n_envs=8 \
  --ppo_config_override ppo.eval_n_envs=8 \
  --ppo_config_override ppo.eval_episodes=8 \
  --ppo_config_override ppo.eval_freq=102400 \
  --ppo_config_override ppo.n_steps=128 \
  --ppo_config_override ppo.batch_size=256 \
  --ppo_config_override ppo.n_epochs=4
