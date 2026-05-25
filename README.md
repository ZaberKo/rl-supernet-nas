# RL Supernet NAS Prototype

这个仓库用于测试在视觉 Gymnasium RL 任务中，用 single-path weight-slicing supernet 做 NAS 搜索的可行性。默认 RL 算法是 Stable-Baselines3 PPO，supernet 只作为共享 backbone，actor 与 critic head 由 PPO policy 负责。

## 配置边界

`config.yaml` 只放跨阶段复用的运行配置：

- `env`: 环境名、seed、统一图像尺寸、SB3 vector env 类型，以及互斥的 `atari_wrapper` 或 `box2d_wrapper` 参数块。
- `ppo`: PPO 训练/finetune/eval 的共用参数，包括 `train_n_envs`、`eval_n_envs`、`eval_episodes`、`eval_freq`、`total_timesteps`、`features_dim`、`n_steps`、`batch_size`、`head_lr`、`policy_net_arch`、`value_net_arch` 等。

`ppo.learning_rate`、`ppo.clip_range` 和 `ppo.clip_range_vf` 支持常数，也支持 RL-Zoo 风格的 `lin_<value>` 线性退火写法，例如 `lin_2.5e-4`。

各 stage 自己的参数仍然定义在对应脚本 argparse 中，例如输出目录、random 数据量、stage2 epoch、NSGA-II population size 等。加载 `ppo_config` 时会先读取 YAML，再合并 `--ppo_config_override`，并返回独立的 OmegaConf config object；它不会把 env/PPO 字段写回 argparse 的 `args`。

临时覆盖 env/PPO 配置使用 `--ppo_config_override key=value`：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py --ppo_config_override ppo.total_timesteps=10000 --ppo_config_override ppo.train_n_envs=4
```

当前 search space 不从 CLI 传入。默认搜索空间 hardcode 在 `supernet_backbone.py` 的 `SearchSpace` 中；如果要改候选宽度、深度、kernel 或 expand ratio，直接改这个类的默认值。

`policy_net_arch` 和 `value_net_arch` 会传给 SB3 `MlpExtractor` 的 `net_arch`，只控制 supernet backbone 输出之后、最终 action/value 线性层之前的 MLP。当前 ALE 默认值为 `[256]`，给 actor/critic head 一个适度容量，避免线性 head 成为 PPO 收敛瓶颈；需要退回 Atari `NatureCNN` 风格的线性 head 时可用：

```bash
python stage1_train_max_ppo.py --ppo_config_override 'ppo.policy_net_arch=[]' --ppo_config_override 'ppo.value_net_arch=[]'
```

## W&B 记录

所有 stage 都会初始化 W&B run，project 固定为 `rl-supernet-nas`。默认 `WANDB_MODE=online`，需要提前登录 W&B；如果只想在本地记录，可以运行前设置：

```bash
export WANDB_MODE=offline
```

每个 stage 会记录 args/config 和关键指标。W&B artifact 只上传轻量输出，例如 manifest、模型 checkpoint、search space 和日志；trajectory / representation 这类大体量数据集只保留在本地路径，不上传到 W&B。

## Vector Env

当前 PPO 代码不直接使用 Gymnasium `AsyncVectorEnv`。Stable-Baselines3 PPO 使用 SB3 自己的 `VecEnv` API，所以这里提供：

- `env.vector_env_type=dummy`: 默认，使用 SB3 `DummyVecEnv`。
- `env.vector_env_type=subproc`: 使用 SB3 `SubprocVecEnv`，适合 `ppo.train_n_envs > 1` 时并行采样。

`env.atari_wrapper` 和 `env.box2d_wrapper` 只能配置其中一个。wrapper 参数块里的 `max_episode_steps` 会控制传给 `gym.make(..., max_episode_steps=...)` 的值：正整数启用 Gymnasium `TimeLimit`，`null` 表示使用环境注册时的默认值，`<=0` 会传 `-1` 来禁用 Gymnasium 自动 `TimeLimit`。ALE 默认使用 `108000` raw frames，这是 Atari 常见的 30 分钟上限；在 SB3 `AtariWrapper` 默认 `MaxAndSkipEnv(skip=4)` 下大约对应 `27000` 个 agent decision。

## Box2D 视觉环境

默认配置仍然面向 Atari；Box2D 可以直接使用 `config_box2d.yaml`。当前 env pipeline 假设环境本身提供图像 observation；`CarRacing-v3` 本身就是 `96x96x3` RGB 图像观测，适合这条路径。`LunarLander-v3` 和 `BipedalWalker-v3` 默认是状态向量，不再通过 `render_mode="rgb_array"` 自动转换成 observation。

Box2D 不需要 Atari 的 no-op、episodic-life、fire-reset 这类 wrapper。对 `CarRacing-v3`，更常见的是轻量视觉预处理：`env.box2d_wrapper.frame_skip=2`、`env.image_size=64`、`env.box2d_wrapper.grayscale_observation=true`、`env.box2d_wrapper.frame_stack=2`。RL-Zoo 的 PPO CarRacing 配置还使用 reward normalization、`n_steps=512`、`batch_size=128`、`n_epochs=10`、`learning_rate=lin_1e-4`、`clip_range=0.2` 和连续动作的 gSDE。

可以直接使用 `config_box2d.yaml`：

```bash
python stage1_train_max_ppo.py \
  --ppo_config config_box2d.yaml \
  --output_dir runs/box2d_car_racing/stage1_ppo_max
```

## Stage 1A: 训练最大 subnet PPO

运行：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py
```

这个阶段会：

- 构造 hardcoded `SearchSpace`，用最大 `ArchConfig` 激活最大 subnet。
- 用最大 subnet backbone + PPO actor/critic head 训练。
- 按 `--sample_ratio` 抽样保存 PPO rollout 生成的 supervised samples 到 `ppo_representation_samples.h5`。
- 按 `ppo.eval_freq` 个 training timestep 定期评估 `ppo.eval_episodes` 个 episode，并和 PPO training log 一起写入 `metrics.jsonl`；每行用 `type` 和 `total_timesteps` 区分 train/eval 记录。
- HDF5 样本会保存 `observation`、`actions`、`targets`、`terminateds`、`truncateds` 和合并后的 `dones`；SB3 VecEnv 的 `TimeLimit.truncated` 会被还原为 Gymnasium 的 truncated 标记。
- 保存训练后的 PPO supernet checkpoint 到 `ppo_supernet_stage1.pt`；每次 eval 和训练结束都会覆盖保存 `ppo_supernet_stage1_last.pt`，并按 `eval/mean_reward` 保存 `ppo_supernet_stage1_best.pt`。
- 写出 `search_space.json` 和 `manifest.json`。

PPO 训练产生的候选样本数量由 `ppo.total_timesteps`、`ppo.train_n_envs`、SB3 rollout 设置和 `--horizon` 共同决定；实际保存数量由 `--sample_ratio` 和可选 `--max_samples` 控制。

常用参数：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py --output_dir runs/stage1_ppo_cartpole --ppo_config_override ppo.total_timesteps=20000
```

## Stage 1B: 采样或混合 random 数据

运行：

```bash
source .venv/bin/activate
python stage1_mix_random_data.py
```

这个阶段会读取 stage1A 的 HDF5 PPO samples，按 `--random_samples` 可选采样 random-policy horizon samples，并生成给 stage2 直接训练用的 datasets Arrow supervised dataset `representation_data.arrow`。random 数据不再写 raw Arrow 中间文件，而是按 `ppo.n_steps` 分块 rollout、切片，并流式写入 `random_representation_samples.h5` 后参与混合。

`--horizon` 必须和 stage1A 写入 HDF5 PPO samples 时使用的 horizon 一致；PPO samples 和 random samples 会先分别切好，再混合保存。

示例：

```bash
source .venv/bin/activate
python stage1_mix_random_data.py --random_samples 50000
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
python stage1_train_max_ppo.py --horizon 3
python stage1_mix_random_data.py --horizon 3
python stage2_train_supernet.py --trajectory_data runs/stage1_mix/representation_data.arrow --train_steps 5000 --random_subnets 4 --projection_dim 128 --dynamics_horizon 3 --dynamics_betas 1.0,0.5,0.25
```

## New Stage 1B: 单架构 PPO finetune 诊断

运行：

```bash
source .venv/bin/activate
python new_stage1_train_arch_ppo.py
```

这个阶段用于在进入 `new_stage2_ea_search.py` 前，先拿一个 JSON 中指定的单个 `ArchConfig` 做 PPO finetune 诊断。默认 `--arch_config arch_configs/max_arch.json` 表示当前 search space 的最大架构；脚本会从 `--supernet_checkpoint runs/new_stage1_policy_supernet/policy_supernet_best.pt` 读取 `new_stage1_train_policy_supernet.py` 保存的 actor supernet 和 critic 参数，然后按 `--candidate_timesteps` 对指定子网做一次 PPO finetune。

输出包括：

- `metrics.jsonl`：critic warmup、rollout、actor/critic update、初始/周期性/最终 eval return；每行用 `type`、`total_timesteps` 和 `total_env_timesteps` 区分记录。
- `policy_supernet_arch_ppo.pt`：可选保存最终 actor supernet、critic、optimizer 和当前激活架构。
- `manifest.json`：记录 arch、checkpoint、参数量、seed、train/eval metrics 路径。

`--supernet_backbone_lr <= 0` 时冻结继承的 supernet backbone，只训练 actor head；大于 0 时会给 backbone 单独设置 learning rate，actor head 使用 `ppo.learning_rate`。`--critic_warmup_timesteps` 可在 actor PPO finetune 前只更新 critic。

示例：

```bash
source .venv/bin/activate
python new_stage1_train_arch_ppo.py \
  --supernet_checkpoint runs/new_stage1_policy_supernet/policy_supernet_best.pt \
  --arch_config arch_configs/max_arch.json \
  --output_dir runs/new_stage1_arch_ppo_max \
  --candidate_timesteps 10000 \
  --critic_warmup_timesteps 0 \
  --ppo_config_override ppo.eval_episodes=3
```

Atari 便捷脚本默认评估最大网络：

```bash
scripts/atari_new_stage1_train_arch_ppo.sh
```

切换到最小网络或固定随机采样网络：

```bash
python new_stage1_train_arch_ppo.py \
  --ppo_config config.yaml \
  --supernet_checkpoint runs/atari_space_invaders/new_stage1_policy_supernet/policy_supernet_best.pt \
  --arch_config arch_configs/min_arch.json \
  --output_dir runs/atari_space_invaders/new_stage1_arch_ppo_min_arch \
  --candidate_timesteps 10000 \
  --critic_warmup_timesteps 0 \
  --supernet_backbone_lr 0.0 \
  --ppo_config_override ppo.eval_episodes=3

python new_stage1_train_arch_ppo.py \
  --ppo_config config.yaml \
  --supernet_checkpoint runs/atari_space_invaders/new_stage1_policy_supernet/policy_supernet_best.pt \
  --arch_config arch_configs/random_arch_seed2026.json \
  --output_dir runs/atari_space_invaders/new_stage1_arch_ppo_random_arch_seed2026 \
  --candidate_timesteps 10000 \
  --critic_warmup_timesteps 0 \
  --supernet_backbone_lr 0.0 \
  --ppo_config_override ppo.eval_episodes=3
```

只看 supernet checkpoint 的初始化质量，不做 PPO finetune：

```bash
python new_stage1_train_arch_ppo.py \
  --ppo_config config.yaml \
  --supernet_checkpoint runs/atari_space_invaders/new_stage1_policy_supernet/policy_supernet_best.pt \
  --arch_config arch_configs/max_arch.json \
  --output_dir runs/atari_space_invaders/new_stage1_arch_ppo_max_arch_zero_step \
  --candidate_timesteps 0 \
  --critic_warmup_timesteps 0 \
  --supernet_backbone_lr 0.0 \
  --ppo_config_override ppo.eval_episodes=3
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
- 用两个目标评估：`negative_return` 和 active backbone `params`；return 基于 `ppo.eval_episodes` 个评估 episode 计算。
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

仓库提供了基于 `ALE/SpaceInvaders-v5` 的分阶段脚本，默认启用 SB3 Atari wrapper，并直接使用 wrapper 默认参数：no-op reset、frame skip + max-pooling、episodic life、fire reset、84x84 grayscale、reward clipping，以及 `VecFrameStack(4)`。启用 wrapper 时，底层 ALE 会固定为 no-frame-skip/no-sticky-action，避免和 wrapper 默认预处理叠加；这些 Atari wrapper 细项不再作为脚本或 config 参数暴露。

```bash
source .venv/bin/activate
scripts/atari_stage1_train_max_ppo.sh
scripts/atari_stage1_mix_random_data.sh
scripts/atari_stage2_train_supernet.sh
scripts/atari_stage3_ea_search.sh
```

## Smoke Test

下面命令用于快速验证每个阶段代码路径可运行，不代表有效训练配置：

```bash
source .venv/bin/activate
python stage1_train_max_ppo.py --output_dir runs/smoke_stage1 --ppo_config_override ppo.total_timesteps=8 --ppo_config_override ppo.n_steps=8 --ppo_config_override ppo.batch_size=8 --ppo_config_override ppo.n_epochs=1 --ppo_config_override ppo.features_dim=32 --ppo_config_override ppo.quiet=true
```

```bash
source .venv/bin/activate
python stage1_mix_random_data.py --ppo_data_file runs/smoke_stage1/ppo_representation_samples.h5 --output_dir runs/smoke_mix --random_samples 4
```

```bash
source .venv/bin/activate
python stage2_train_supernet.py --trajectory_data runs/smoke_mix/representation_data.arrow --stage1_backbone runs/smoke_stage1/ppo_supernet_stage1.pt --output_dir runs/smoke_stage2 --train_steps 1 --batch_size 2 --random_subnets 1 --projection_dim 16 --predictor_hidden_dim 32 --ppo_config_override ppo.features_dim=32
```

```bash
source .venv/bin/activate
python stage3_ea_search.py --supernet_checkpoint runs/smoke_stage2/supernet_backbone_stage2.pt --output_dir runs/smoke_stage3 --population_size 2 --generations 1 --candidate_timesteps 0 --eval_workers 2 --ppo_config_override ppo.eval_episodes=1 --ppo_config_override ppo.n_steps=8 --ppo_config_override ppo.batch_size=8 --ppo_config_override ppo.n_epochs=1 --ppo_config_override ppo.features_dim=32 --ppo_config_override ppo.quiet=true
```
