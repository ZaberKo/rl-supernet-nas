# PLAN

## Goal

Build a runnable prototype for testing supernet NAS on visual Gymnasium RL tasks. The default RL algorithm is PPO. The supernet is a single-path weight-slicing CNN backbone, actor/critic heads are task-specific modules attached by Stable-Baselines3 policies, and shared environment/PPO settings are loaded from an OmegaConf YAML file.

## Implementation Checklist

All listed implementation items are complete and smoke-tested.

0. Shared configuration
   - Store only reusable environment and PPO settings in `config.yaml`.
   - Load configuration through OmegaConf in each stage script.
   - Keep stage-specific settings in each stage script argparse, not in YAML.

1. CNN search space and EA codec
   - Put the runnable CNN supernet search-space schema directly in `supernet_backbone.py`.
   - Keep integer gene encoding, decoding, sampling, mutation, and crossover in `ea_codec.py`.
   - Do not encode fixed widths, fixed stems, fixed transitions, fixed pooling, or fixed heads as gene entries.

2. Supernet backbone
   - Implement an OFA-style CNN backbone with a maximum network and single-path slices.
   - Keep the supernet limited to feature extraction.
   - Expose APIs to set active subnet configs and load inherited weights.

3. Stage 1: PPO sampling and random-data mixing
   - Use `stage1_train_max_ppo.py` to train PPO using the maximum supernet backbone with shared actor/critic feature extraction and YAML-loaded env/PPO settings.
   - Record all trajectories generated during PPO training, without a separate trajectory cap, save supernet backbone weights, and log metrics/artifacts to W&B.
   - Store trajectories through HuggingFace `datasets` as PyArrow-backed datasets, with `terminateds`, `truncateds`, and merged `dones` kept independently.
   - Use `stage1_mix_random_data.py` to collect or subset random-policy trajectories independently.
   - Control the random/PPO mixture ratio without rerunning the PPO training script.
   - Emit one mixed Arrow dataset that stage 2 can consume directly and log counts/artifacts to W&B.

4. Stage 2: representation learning
   - Load the supervised representation Arrow dataset produced by stage 1B.
   - Stage 1B preserves terminal flags and done boundaries, then pre-packs one-step or k-step supervised sliding-window samples without episode-boundary padding.
   - Train on shuffled samples directly; stage2 does not build k-step windows itself and masks offsets after the first done signal.
   - Initialize the supernet from stage 1 weights when provided.
   - Train sampled subnets with sandwich sampling: max teacher, min subnet, and random subnets.
   - Use direct loss functions for action-conditioned k-step latent dynamics prediction and cosine latent KD.
   - Use AdamW with separate backbone/head learning rates and warmup + cosine scheduling.
   - Save the trained supernet checkpoint and record stage metrics/artifacts to W&B.

5. Stage 3: EvoX NSGA-II subnet search
   - Wrap the RL subnet evaluation as an EvoX `Problem`.
   - Use a discrete EvoX NSGA-II `Algorithm` over integer genes with default crossover/mutation settings, and decode each gene to `ArchConfig` before PPO evaluation.
   - Optimize two objectives: maximize return via `negative_return` minimization and minimize active backbone parameter count.
   - Print and write one generation-level search log line per iteration, write one JSONL row per individual with `gen`, `arch`, and `is_pareto` fields, and record metrics/artifacts to W&B.
   - Use torch multiprocessing inside `Problem.evaluate` when `--eval_workers > 1`.
   - For each subnet, inherit compatible supernet weights and initialize new actor/critic heads.
   - Support `--supernet_backbone_lr`: freeze the backbone when `<= 0`, otherwise train it with a separate learning rate.

6. Smoke tests
   - Validate import and search-space conversion.
   - Run a short stage 1 PPO training job.
   - Run a short stage 1 random mixing job.
   - Run a short stage 2 representation-training job.
   - Run a short stage 3 EA job.
   - Each run should be stopped or bounded after proving the code path is executable.

## Default Smoke-Test Environment

Use `CartPole-v1` with `rgb_array` rendering converted into image observations. This avoids requiring Atari or Box2D packages while still testing the visual-observation CNN path. CLI/config overrides also support ALE Atari environments with SB3 Atari preprocessing and frame stacking.

## Expected Artifacts

- `supernet_backbone.py`
- `primitive_blocks.py`
- `ea_codec.py`
- `nsga2_search.py`
- `env_utils.py`
- `sb3_nas_policy.py`
- `ppo_utils.py`
- `trajectory_data.py`
- `representation_losses.py`
- `stage1_train_max_ppo.py`
- `stage1_mix_random_data.py`
- `stage2_train_supernet.py`
- `stage3_ea_search.py`
- `PLAN.md`
