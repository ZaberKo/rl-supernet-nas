#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_ID="${ENV_ID:-ALE/Pong-v5}"
RUN_ROOT="${RUN_ROOT:-runs/atari_pong}"
PPO_TRAJECTORY_FILE="${PPO_TRAJECTORY_FILE:-${RUN_ROOT}/stage1_ppo_max/ppo_train_trajectories.arrow}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/stage1_mix}"
SEED="${SEED:-0}"
TRAIN_N_ENVS="${TRAIN_N_ENVS:-4}"
VECTOR_ENV_TYPE="${VECTOR_ENV_TYPE:-subproc}"
RANDOM_TO_PPO_RATIO="${RANDOM_TO_PPO_RATIO:-1.0}"
REPRESENTATION_HORIZON="${REPRESENTATION_HORIZON:-3}"

"${PYTHON_BIN}" stage1_mix_random_data.py \
  --ppo_trajectory_file "${PPO_TRAJECTORY_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --random_to_ppo_ratio "${RANDOM_TO_PPO_RATIO}" \
  --representation_horizon "${REPRESENTATION_HORIZON}" \
  --ppo_config_override env.env_id="${ENV_ID}" \
  --ppo_config_override env.seed="${SEED}" \
  --ppo_config_override env.native_image_env=true \
  --ppo_config_override env.image_size=84 \
  --ppo_config_override env.vector_env_type="${VECTOR_ENV_TYPE}" \
  --ppo_config_override env.frame_stack=4 \
  --ppo_config_override env.atari_wrapper=sb3 \
  --ppo_config_override ppo.train_n_envs="${TRAIN_N_ENVS}"
