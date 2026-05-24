#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python new_stage2_ea_search.py \
  --supernet_checkpoint runs/atari_space_invaders/new_stage1_policy_supernet/policy_supernet_best.pt \
  --output_dir runs/atari_space_invaders/new_stage2_ea_search \
  --population_size 6 \
  --generations 3 \
  --candidate_timesteps 10000 \
  --critic_warmup_timesteps 0 \
  --eval_workers 1 \
  --supernet_backbone_lr 0.0 \
  --checkpoint_policy_key policy_state_dict \
  --ppo_config_override ppo.eval_episodes=3
