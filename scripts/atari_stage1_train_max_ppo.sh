#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_ID="${ENV_ID:-ALE/Pong-v5}"
RUN_ROOT="${RUN_ROOT:-runs/atari_pong}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/stage1_ppo_max}"
SEED="${SEED:-0}"
TRAIN_N_ENVS="${TRAIN_N_ENVS:-4}"
VECTOR_ENV_TYPE="${VECTOR_ENV_TYPE:-subproc}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
N_STEPS="${N_STEPS:-128}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_EPOCHS="${N_EPOCHS:-4}"
FEATURES_DIM="${FEATURES_DIM:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.00025}"
HEAD_LR="${HEAD_LR:-0.00025}"
ENT_COEF="${ENT_COEF:-0.01}"
DEVICE="${DEVICE:-auto}"

"${PYTHON_BIN}" stage1_train_max_ppo.py \
  --output_dir "${OUTPUT_DIR}" \
  --save_ppo_model \
  --ppo_config_override env.env_id="${ENV_ID}" \
  --ppo_config_override env.seed="${SEED}" \
  --ppo_config_override env.native_image_env=true \
  --ppo_config_override env.image_size=84 \
  --ppo_config_override env.vector_env_type="${VECTOR_ENV_TYPE}" \
  --ppo_config_override env.frame_stack=4 \
  --ppo_config_override env.atari_wrapper=sb3 \
  --ppo_config_override ppo.train_n_envs="${TRAIN_N_ENVS}" \
  --ppo_config_override ppo.eval_n_envs=1 \
  --ppo_config_override ppo.total_timesteps="${TOTAL_TIMESTEPS}" \
  --ppo_config_override ppo.features_dim="${FEATURES_DIM}" \
  --ppo_config_override ppo.learning_rate="${LEARNING_RATE}" \
  --ppo_config_override ppo.head_lr="${HEAD_LR}" \
  --ppo_config_override ppo.n_steps="${N_STEPS}" \
  --ppo_config_override ppo.batch_size="${BATCH_SIZE}" \
  --ppo_config_override ppo.n_epochs="${N_EPOCHS}" \
  --ppo_config_override ppo.ent_coef="${ENT_COEF}" \
  --ppo_config_override ppo.device="${DEVICE}"
