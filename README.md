# RL Supernet NAS Prototype

这个仓库用于测试在视觉 Gymnasium RL 任务中，用 single-path weight-slicing supernet 做 NAS 搜索的可行性。默认 RL 算法是 Stable-Baselines3 PPO，supernet 只作为共享 backbone，actor 与 critic head 由 PPO policy 负责。

## 配置边界

`config.yaml` 只放跨阶段复用的运行配置：

- `env`: 环境名、seed、图像尺寸、是否使用原生图像观测、SB3 vector env 类型、Atari wrapper 和 frame stack。
- `ppo`: PPO 训练/finetune/eval 的共用参数，包括 `train_n_envs`、`eval_n_envs`、`total_timesteps`、`features_dim`、`n_steps`、`batch_size`、`head_lr` 等。

各 stage 自己的参数仍然定义在对应脚本 argparse 中，例如输出目录、random 数据比例、stage2 epoch、NSGA-II population size 等。加载 `ppo_config` 时会先合并代码内置的 env/PPO 默认值，再合并 YAML 和 `--ppo_config_override`；最后只补充当前 args 中不存在的字段。如果名字冲突，stage argparse 的值优先。

临时覆盖 env/PPO 配置使用 `--ppo_config_override key=value`：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py --ppo_config_override ppo.total_timesteps=10000 --ppo_config_override ppo.train_n_envs=4
```

当前 search space 不从 CLI 传入。默认搜索空间 hardcode 在 `supernet_backbone.py` 的 `SearchSpace` 中；如果要改候选宽度、深度、kernel 或 expand ratio，直接改这个类的默认值。

## W&B 记录

所有 stage 都会初始化 W&B run，project 固定为 `rl-supernet-nas`。默认 `WANDB_MODE=offline`，记录会落在各 stage 输出目录下的 `wandb/`；需要在线同步时先登录 W&B 并设置：

```bash
export WANDB_MODE=online
```

每个 stage 会记录 args/config、关键指标，并把主要输出文件或 Arrow dataset 作为 artifact：stage1A 记录 PPO 轨迹和 backbone，stage1B 记录 random/mixed dataset，stage2 记录 loss 曲线和 checkpoint，stage3 记录每代搜索日志、JSONL 个体记录和最终 manifest。

## Vector Env

当前 PPO 代码不直接使用 Gymnasium `AsyncVectorEnv`。Stable-Baselines3 PPO 使用 SB3 自己的 `VecEnv` API，所以这里提供：

- `env.vector_env_type=dummy`: 默认，使用 SB3 `DummyVecEnv`。
- `env.vector_env_type=subproc`: 使用 SB3 `SubprocVecEnv`，适合 `ppo.train_n_envs > 1` 时并行采样。

## Stage 1A: 训练最大 subnet PPO

运行：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py
```

这个阶段会：

- 构造 hardcoded `SearchSpace`，用最大 `ArchConfig` 激活最大 subnet。
- 用最大 subnet backbone + PPO actor/critic head 训练。
- 记录 PPO 训练期间产生的全部轨迹到 `ppo_train_trajectories.arrow`。
- 轨迹会通过 HuggingFace `datasets` 写成 PyArrow-backed dataset 目录，并独立保存 `terminateds`、`truncateds` 和合并后的 `dones`；SB3 VecEnv 的 `TimeLimit.truncated` 会被还原为 Gymnasium 的 truncated 标记。
- 保存训练后的 backbone 到 `supernet_backbone_stage1.pt`。
- 写出 `search_space.json` 和 `manifest.json`。

没有单独的 `ppo_trajectory_transitions` 参数。PPO 训练轨迹数量由 `ppo.total_timesteps`、`ppo.train_n_envs` 和 SB3 rollout 设置共同决定；callback 不再做额外截断，训练期间实际产生的环境 transition 都会记录下来。

常用参数：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py --output_dir runs/stage1_ppo_cartpole --ppo_config_override ppo.total_timesteps=20000
```

如需保存完整 PPO 模型 zip：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py --save_ppo_model
```

## Stage 1B: 采样或混合 random 数据

运行：

```bash
source .venv/bin/activate
python stage1_mix_random_data.py
```

这个阶段会读取 stage1A 的 PPO 轨迹，按脚本参数采样 random-policy 轨迹，并写出 raw mixed 轨迹 `mixed_trajectories.arrow`，同时生成给 stage2 直接训练用的 supervised dataset `representation_data.arrow`。

随机数据规模可用三种方式控制，优先级从高到低：

1. `--random_transitions`: 直接指定 random transition 数。
2. `--random_steps`: 指定 random env step 数，实际 transition 数为 `random_steps * ppo.train_n_envs`。
3. `--random_fraction` 或 `--random_to_ppo_ratio`: 按 PPO 数据量计算 random 数据比例。

示例：

```bash
source .venv/bin/activate
python stage1_mix_random_data.py --random_to_ppo_ratio 0.5
```

如果已经有 random 轨迹文件，可以只截取前缀并重新生成 mixed dataset：

```bash
source .venv/bin/activate
python stage1_mix_random_data.py --existing_random_trajectory_file runs/stage1_mix/random_trajectories.arrow --random_fraction 0.25
```

## Stage 2: supernet 表征学习

运行：

```bash
source .venv/bin/activate
python stage2_train_supernet.py --trajectory_data runs/stage1_mix/representation_data.arrow
```

这个阶段会：

- 从 stage1A backbone checkpoint 继承初始参数。
- 从 stage1B 生成的 `representation_data.arrow` 读取 PPO + random supervised samples。
- stage1 会保留 `terminated`/`truncated` transition，并在 `representation_data.arrow` 中预先打包 one-step 或 k-step 滑窗样本。
- k-step 样本不在 episode 边界 padding；窗口会直接跨到后续 episode，`done`/`terminated`/`truncated` 序列随样本保存，stage2 用 `done` mask 掉 episode 边界之后的 offset。
- 每个 batch 使用 sandwich 采样：最大网络作为 teacher，最小 subnet 与若干随机 subnet 作为 student。
- 使用函数式 `latent_dynamics_loss` 和 `cosine_kd_loss` 训练 supernet backbone；`--dynamics_betas` 可设置每个 horizon step 的 beta 权重，mask 会排除 episode 边界之后的无效未来步。
- 使用 backbone/head 两组 AdamW learning rate，并使用 warmup + cosine scheduler。
- 保存 `supernet_backbone_stage2.pt`、`metrics.jsonl`、`manifest.json`。

stage2 的 AdamW 使用 PyTorch 默认 beta/eps；DataLoader 的 shuffle/pin_memory/drop_last 也 hardcode 为常规训练默认值。

常用参数：

```bash
source .venv/bin/activate
python stage1_mix_random_data.py --representation_horizon 3
python stage2_train_supernet.py --trajectory_data runs/stage1_mix/representation_data.arrow --train_steps 5000 --random_subnets 4 --projection_dim 128 --dynamics_horizon 3 --dynamics_betas 1.0,0.5,0.25
```

## Stage 3: NSGA-II subnet 搜索

运行：

```bash
source .venv/bin/activate
python stage3_ea_search.py
```

这个阶段会：

- 从 stage2 supernet checkpoint 继承 backbone 参数。
- 用 EvoX NSGA-II 在整数 gene 空间中搜索 subnet。
- EA 层保留 gene；进入 PPO finetune 前先解码为 `ArchConfig`。
- 每个 subnet 初始化新的 actor/critic head，然后做短程 PPO finetune。
- 用两个目标评估：`negative_return` 和 active backbone `params`。
- 每代都会 print 一行搜索日志并追加到 `search.log`。
- 每代每个个体都会向 `nsga2_records.jsonl` 写一行，包含 `gen`、`arch`、`objectives`、`return`、`params`、`pareto_rank`、`is_pareto` 等字段。
- 写出 `manifest.json`，其中包含最终 Pareto front。mutation/crossover 使用代码默认值，初始种群固定包含 max architecture。

`--supernet_backbone_lr <= 0` 时冻结 backbone；大于 0 时，backbone 使用这个单独 learning rate，actor/critic head 使用 `ppo.head_lr`。

并发评估示例：

```bash
source .venv/bin/activate
python stage3_ea_search.py --eval_workers 2 --population_size 8 --generations 5
```

## Atari 示例脚本

仓库提供了基于 `ALE/Pong-v5` 的分阶段脚本，默认启用 SB3 Atari wrapper，并直接使用 wrapper 默认参数：no-op reset、frame skip + max-pooling、episodic life、fire reset、84x84 grayscale、reward clipping，以及 `VecFrameStack(4)`。启用 wrapper 时，底层 ALE 会固定为 no-frame-skip/no-sticky-action，避免和 wrapper 默认预处理叠加；这些 Atari wrapper 细项不再作为脚本或 config 参数暴露。

```bash
source .venv/bin/activate
scripts/atari_stage1_train_max_ppo.sh
scripts/atari_stage1_mix_random_data.sh
scripts/atari_stage2_train_supernet.sh
scripts/atari_stage3_ea_search.sh
```

常用环境变量覆盖：

```bash
ENV_ID=ALE/Breakout-v5 RUN_ROOT=runs/atari_breakout TOTAL_TIMESTEPS=200000 scripts/atari_stage1_train_max_ppo.sh
RUN_ROOT=runs/atari_breakout TRAIN_STEPS=20000 DYNAMICS_HORIZON=3 scripts/atari_stage2_train_supernet.sh
RUN_ROOT=runs/atari_breakout POPULATION_SIZE=8 GENERATIONS=5 CANDIDATE_TIMESTEPS=50000 scripts/atari_stage3_ea_search.sh
```

## Smoke Test

下面命令用于快速验证每个阶段代码路径可运行，不代表有效训练配置：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py --output_dir runs/smoke_stage1 --ppo_config_override ppo.total_timesteps=8 --ppo_config_override ppo.n_steps=8 --ppo_config_override ppo.batch_size=8 --ppo_config_override ppo.n_epochs=1 --ppo_config_override ppo.features_dim=32 --ppo_config_override ppo.quiet=true
```

```bash
source .venv/bin/activate
python stage1_mix_random_data.py --ppo_trajectory_file runs/smoke_stage1/ppo_train_trajectories.arrow --output_dir runs/smoke_mix --random_to_ppo_ratio 0.5
```

```bash
source .venv/bin/activate
python stage2_train_supernet.py --trajectory_data runs/smoke_mix/representation_data.arrow --stage1_backbone runs/smoke_stage1/supernet_backbone_stage1.pt --output_dir runs/smoke_stage2 --train_steps 1 --batch_size 2 --random_subnets 1 --projection_dim 16 --predictor_hidden_dim 32 --ppo_config_override ppo.features_dim=32
```

```bash
source .venv/bin/activate
python stage3_ea_search.py --supernet_checkpoint runs/smoke_stage2/supernet_backbone_stage2.pt --output_dir runs/smoke_stage3 --population_size 2 --generations 1 --candidate_timesteps 0 --eval_episodes 1 --eval_workers 2 --ppo_config_override ppo.n_steps=8 --ppo_config_override ppo.batch_size=8 --ppo_config_override ppo.n_epochs=1 --ppo_config_override ppo.features_dim=32 --ppo_config_override ppo.quiet=true
```
