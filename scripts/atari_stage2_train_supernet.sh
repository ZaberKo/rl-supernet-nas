#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_ID="${ENV_ID:-ALE/Pong-v5}"
RUN_ROOT="${RUN_ROOT:-runs/atari_pong}"
TRAJECTORY_DATA="${TRAJECTORY_DATA:-${RUN_ROOT}/stage1_mix/representation_data.arrow}"
STAGE1_BACKBONE="${STAGE1_BACKBONE:-${RUN_ROOT}/stage1_ppo_max/supernet_backbone_stage1.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/stage2}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RANDOM_SUBNETS="${RANDOM_SUBNETS:-2}"
DYNAMICS_HORIZON="${DYNAMICS_HORIZON:-3}"
DYNAMICS_BETAS="${DYNAMICS_BETAS:-1.0,0.5,0.25}"
FEATURES_DIM="${FEATURES_DIM:-256}"
PROJECTION_DIM="${PROJECTION_DIM:-128}"
PREDICTOR_HIDDEN_DIM="${PREDICTOR_HIDDEN_DIM:-512}"
BACKBONE_LR="${BACKBONE_LR:-0.00003}"
HEAD_LR="${HEAD_LR:-0.0001}"
DEVICE="${DEVICE:-auto}"

"${PYTHON_BIN}" stage2_train_supernet.py \
  --trajectory_data "${TRAJECTORY_DATA}" \
  --stage1_backbone "${STAGE1_BACKBONE}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_steps "${TRAIN_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --random_subnets "${RANDOM_SUBNETS}" \
  --dynamics_horizon "${DYNAMICS_HORIZON}" \
  --dynamics_betas "${DYNAMICS_BETAS}" \
  --projection_dim "${PROJECTION_DIM}" \
  --predictor_hidden_dim "${PREDICTOR_HIDDEN_DIM}" \
  --backbone_learning_rate "${BACKBONE_LR}" \
  --head_learning_rate "${HEAD_LR}" \
  --ppo_config_override env.env_id="${ENV_ID}" \
  --ppo_config_override env.seed="${SEED}" \
  --ppo_config_override env.native_image_env=true \
  --ppo_config_override env.image_size=84 \
  --ppo_config_override env.vector_env_type=dummy \
  --ppo_config_override env.frame_stack=4 \
  --ppo_config_override env.atari_wrapper=sb3 \
  --ppo_config_override ppo.features_dim="${FEATURES_DIM}" \
  --ppo_config_override ppo.device="${DEVICE}"
