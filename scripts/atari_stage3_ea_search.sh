#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_ID="${ENV_ID:-ALE/Pong-v5}"
RUN_ROOT="${RUN_ROOT:-runs/atari_pong}"
SUPERNET_CHECKPOINT="${SUPERNET_CHECKPOINT:-${RUN_ROOT}/stage2/supernet_backbone_stage2.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/stage3}"
SEED="${SEED:-0}"
TRAIN_N_ENVS="${TRAIN_N_ENVS:-2}"
EVAL_N_ENVS="${EVAL_N_ENVS:-1}"
POPULATION_SIZE="${POPULATION_SIZE:-6}"
GENERATIONS="${GENERATIONS:-3}"
CANDIDATE_TIMESTEPS="${CANDIDATE_TIMESTEPS:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
SUPERNET_BACKBONE_LR="${SUPERNET_BACKBONE_LR:-0.0}"
FEATURES_DIM="${FEATURES_DIM:-256}"
HEAD_LR="${HEAD_LR:-0.00025}"
N_STEPS="${N_STEPS:-128}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_EPOCHS="${N_EPOCHS:-4}"
ENT_COEF="${ENT_COEF:-0.01}"
DEVICE="${DEVICE:-auto}"

"${PYTHON_BIN}" stage3_ea_search.py \
  --supernet_checkpoint "${SUPERNET_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --population_size "${POPULATION_SIZE}" \
  --generations "${GENERATIONS}" \
  --candidate_timesteps "${CANDIDATE_TIMESTEPS}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --eval_workers "${EVAL_WORKERS}" \
  --supernet_backbone_lr "${SUPERNET_BACKBONE_LR}" \
  --ppo_config_override env.env_id="${ENV_ID}" \
  --ppo_config_override env.seed="${SEED}" \
  --ppo_config_override env.native_image_env=true \
  --ppo_config_override env.image_size=84 \
  --ppo_config_override env.vector_env_type=dummy \
  --ppo_config_override env.frame_stack=4 \
  --ppo_config_override env.atari_wrapper=sb3 \
  --ppo_config_override ppo.train_n_envs="${TRAIN_N_ENVS}" \
  --ppo_config_override ppo.eval_n_envs="${EVAL_N_ENVS}" \
  --ppo_config_override ppo.features_dim="${FEATURES_DIM}" \
  --ppo_config_override ppo.head_lr="${HEAD_LR}" \
  --ppo_config_override ppo.learning_rate="${HEAD_LR}" \
  --ppo_config_override ppo.n_steps="${N_STEPS}" \
  --ppo_config_override ppo.batch_size="${BATCH_SIZE}" \
  --ppo_config_override ppo.n_epochs="${N_EPOCHS}" \
  --ppo_config_override ppo.ent_coef="${ENT_COEF}" \
  --ppo_config_override ppo.device="${DEVICE}"
