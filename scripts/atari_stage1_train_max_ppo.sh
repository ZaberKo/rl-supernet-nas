#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python stage1_train_max_ppo.py \
  --output_dir runs/atari_pong/stage1_ppo_max \
  --save_ppo_model runs/atari_pong/stage1_ppo_max/ppo_max_supernet_model.zip \
  --ppo_config_override ppo.total_timesteps=50000 \
  --ppo_config_override ppo.train_n_envs=4 \
  --ppo_config_override ppo.eval_n_envs=1 \
  --ppo_config_override ppo.n_steps=128 \
  --ppo_config_override ppo.batch_size=256 \
  --ppo_config_override ppo.n_epochs=4
