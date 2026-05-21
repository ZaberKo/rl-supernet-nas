# Data Storage Refactor Plan

## Goal

Keep Atari stage1 data under a practical local storage budget, with a target budget of about 10 GB for a 10M-timestep run, while preserving enough samples for stage2 supernet representation training.

The main direction is to stop saving the full PPO trajectory. Stage1 should directly emit sampled supervised representation records, using the same logical sample shape as the current stage1 mix output:

- `observation`: the starting observation for the sample.
- `actions`: the next `representation_horizon` actions.
- `targets`: the next `representation_horizon` target observations.
- `dones`, `terminateds`, `truncateds`: horizon-length masks.

## Current Size Problem

Current Atari config:

- `total_timesteps`: `10_000_000`
- `train_n_envs`: `8`
- `n_steps`: `128`
- actual PPO transition count: `ceil(10_000_000 / 1024) * 1024 = 10_000_384`
- observation shape after wrappers: `(4, 84, 84)` uint8

Current raw trajectory storage saves both `observation` and `next_observation`:

```text
bytes_per_observation = 4 * 84 * 84 = 28,224
bytes_per_transition_images = 2 * 28,224 = 56,448
total_image_bytes = 10,000,384 * 56,448 = 564.5 GB
```

This does not include Arrow nested-list overhead, Python buffering, metadata, actions, rewards, or temporary copies. The current approach is not viable.

For `representation_horizon=3`, if stage1 directly saves supervised samples using stacked observations and targets:

```text
bytes_per_sample_images = (1 + horizon) * 4 * 84 * 84
                       = 4 * 28,224
                       = 112,896 bytes
```

Approximate uncompressed image budget:

```text
80,000 samples  ~=  9.0 GB raw image bytes
100,000 samples ~= 11.3 GB raw image bytes
```

Compression should help because Atari frames contain large flat regions, but the implementation should not depend on optimistic compression to avoid runaway disk usage.

## Storage Library Decision

Use Zarr with chunked arrays and Blosc/Zstd compression for the new stage1 representation dataset.

Rationale:

- The data is mostly fixed-shape numeric arrays, not heterogeneous records.
- Zarr supports chunked N-dimensional arrays and append-style writes.
- Compressed chunks are suitable for image-like data.
- A PyTorch map-style dataset can randomly index Zarr arrays without loading the full dataset.
- It avoids the current HuggingFace `datasets` pattern of building Python lists and then writing one huge Arrow dataset.

Preferred dependency:

```text
zarr
numcodecs
```

Preferred compressor:

```python
numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
```

If adding Zarr is not acceptable, the fallback is PyArrow Parquet shards with Zstd compression and an IterableDataset-style stage2 loader. That fallback is less convenient for random access.

## Dataset Layout

Create a new local dataset directory, for example:

```text
ppo_representation_samples.zarr/
  observation
  actions
  targets
  dones
  terminateds
  truncateds
  env_index
  source_step
  metadata.json
```

Array shapes:

```text
observation:  [num_samples, C, H, W] uint8
actions:      [num_samples, horizon, *action_shape]
targets:      [num_samples, horizon, C, H, W] uint8
dones:        [num_samples, horizon] bool
terminateds:  [num_samples, horizon] bool
truncateds:   [num_samples, horizon] bool
env_index:    [num_samples] int16 or int32
source_step:  [num_samples] int64
```

For Discrete Atari actions, store actions as `int64` or `int16`. Prefer `int64` initially for compatibility with existing PyTorch action handling, then optimize later if needed.

Chunk shape proposal:

```text
observation: [1024, C, H, W]
targets:     [1024, horizon, C, H, W]
actions:     [4096, horizon, *action_shape]
masks:       [4096, horizon]
```

The chunk size can be tuned after a smoke benchmark. Start conservative so flushes are frequent enough and memory remains bounded.

## Sampling CLI

Add stage1 CLI arguments:

```text
--representation_output_name ppo_representation_samples.zarr
--representation_horizon 3
--representation_sample_ratio 0.005
--representation_max_samples 80000
--representation_sample_seed 0
--representation_storage zarr
--representation_compression zstd
--representation_chunk_size 1024
```

Behavior:

- `representation_horizon` moves from stage1 mix into stage1 train.
- Stage1 train writes representation samples directly during PPO training.
- `representation_sample_ratio` controls how many eligible horizon windows are retained.
- `representation_max_samples` is a hard cap.
- If both ratio and max samples are set, save only samples selected by the ratio until the hard cap is reached.
- Print and save an estimated uncompressed byte budget before training starts.
- Refuse to run or warn loudly when the estimated uncompressed output exceeds a configurable budget.

Suggested additional argument:

```text
--representation_budget_gb 10
```

This should compute a conservative `max_samples` when `--representation_max_samples` is not explicitly set:

```text
max_samples = floor(budget_bytes / bytes_per_sample_images)
```

For Atari with `horizon=3`, `budget_gb=10` gives about 88k samples before non-image overhead. Use a lower default such as 80k for safety.

## Online Horizon Builder

Replace the full `TrajectoryRecorderCallback` path for stage1 with a sampled representation writer callback.

Per vector env index, keep a small rolling deque of recent transition rows:

```text
observation
action
next_observation
terminated
truncated
done
source_step
```

When the deque has `representation_horizon` transitions:

1. Build one supervised sample from the first `horizon` transitions.
2. Apply the sampling decision.
3. If selected, append the sample to the Zarr writer buffer.
4. Flush the writer buffer when it reaches `representation_chunk_size`.
5. If a done is encountered, reset that env deque after emitting valid non-crossing samples.

Important behavior decision:

- Do not let a sample cross an episode boundary.
- If a transition has `done=True`, the sample may include that transition, but later targets should not come from the next episode.
- This is cleaner than the current offline stage1 mix behavior, which can synthesize truncation at trajectory boundaries after sorting rows.

## Sampling Strategy

Use deterministic thinning rather than pure Bernoulli by default.

Option A: ratio-based hash sampling

```text
keep = hash(global_eligible_sample_index, representation_sample_seed) < ratio
```

Pros:

- Deterministic.
- Does not depend on Python RNG ordering.
- Uniform over the whole run.

Cons:

- Final count is approximate.

Option B: stride-based sampling when `representation_max_samples` is set and expected eligible count is known

```text
stride = ceil(expected_eligible_samples / max_samples)
keep = (global_eligible_sample_index + offset) % stride == 0
```

Pros:

- Predictable upper bound.
- Even coverage over the full run.

Cons:

- Less random.

Recommended initial implementation:

- Use stride sampling when `representation_max_samples` or `representation_budget_gb` is provided.
- Use hash sampling only when the user explicitly provides `representation_sample_ratio` without a max sample cap.

## Stage Changes

Stage1 train:

- Stop saving full `ppo_train_trajectories.arrow` by default.
- Save `ppo_representation_samples.zarr` instead.
- Keep a CLI escape hatch for short debugging runs:

```text
--save_full_trajectory false
```

The default should be `false`.

Stage1 mix:

- No longer required for the default Atari flow.
- Keep it as an optional legacy/offline data mixing script for small datasets.
- If random samples are still desired, add a random-policy representation writer that writes the same Zarr schema directly, instead of first writing raw trajectories.

Stage2 train:

- Add a Zarr-backed lazy dataset.
- Do not call `load_supervised_transition_rows`.
- Keep the existing Arrow dataset loader only for backward compatibility with small old runs.
- DataLoader can remain map-style with `shuffle=True` if Zarr random indexing is fast enough.
- If random indexing is slow, switch to a shard-aware IterableDataset with buffer shuffle.

W&B:

- Do not upload trajectory, mixed, random, or representation datasets as artifacts.
- Only upload manifest, checkpoints, metrics, search space, and compact logs.

## Implementation Steps

1. Add Zarr dependency to the project environment.
2. Add a `ZarrRepresentationWriter` in `trajectory_data.py`.
3. Add a `RepresentationRecorderCallback` in `trajectory_data.py`.
4. Add stage1 CLI arguments for horizon, sample ratio, sample cap, budget, chunk size, and storage path.
5. Replace the default stage1 callback with the representation recorder.
6. Update manifest fields:
   - `representation_data`
   - `representation_horizon`
   - `representation_sample_ratio`
   - `representation_max_samples`
   - `representation_num_samples`
   - `representation_estimated_uncompressed_bytes`
   - `representation_actual_bytes`
7. Add `ZarrTransitionDataset` for stage2.
8. Update stage2 to select loader by dataset metadata/path suffix.
9. Keep existing Arrow support for old smoke tests and legacy artifacts.
10. Add smoke tests:
    - tiny Atari run, horizon 3, max samples 16
    - verify no sample crosses done boundary
    - verify stage2 can train one step from Zarr data
    - verify W&B artifact paths exclude large dataset directories

## Open Questions

- Default sample budget: use `80_000` samples for Atari horizon 3, or derive from `--representation_budget_gb=10`.
- Whether to store stacked observations initially, or immediately switch to single-frame storage and reconstruct frame stacks in the dataset.
- Whether random-policy data is still needed for stage2; if yes, implement it directly in the same representation Zarr schema.
- Whether to make `stage1_mix_random_data.py` legacy-only or rewrite it around Zarr.

## Recommended First Pass

Implement the conservative version first:

- Store stacked observations and stacked targets, matching the current stage2 input contract.
- Use Zarr + Zstd compression.
- Default `representation_horizon=3`.
- Default `representation_budget_gb=10`.
- Derive `representation_max_samples` from the raw image byte estimate.
- Do not save full PPO trajectories unless explicitly requested.

This should keep the implementation small while fixing the immediate storage explosion.
