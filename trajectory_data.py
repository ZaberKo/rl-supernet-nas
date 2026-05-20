from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import gymnasium as gym
import numpy as np
import torch
from datasets import Dataset as ArrowDataset
from datasets import Features, Sequence as ArrowSequence, Value
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.data import Dataset as TorchDataset


TRAJECTORY_METADATA_FILE = "metadata.json"
TRAJECTORY_STORAGE_FORMAT = "datasets_arrow"
SUPERVISED_DATASET_TYPE = "supervised_transition_samples"


def split_done_flags(
    dones: np.ndarray,
    infos: Sequence[dict[str, Any]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    done_array = np.asarray(dones, dtype=np.bool_)
    info_items = list(infos or [{} for _ in range(done_array.shape[0])])
    truncated = np.asarray(
        [bool(info.get("TimeLimit.truncated", False)) for info in info_items],
        dtype=np.bool_,
    )
    terminated = done_array & ~truncated
    return terminated, truncated, done_array


def resolve_terminal_next_observations(
    next_observation: np.ndarray,
    infos: Sequence[dict[str, Any]] | None,
) -> np.ndarray:
    resolved = np.asarray(next_observation).copy()
    for env_index, info in enumerate(infos or []):
        terminal_observation = info.get("terminal_observation")
        if terminal_observation is not None:
            resolved[env_index] = np.asarray(terminal_observation)
    return resolved


def is_arrow_trajectory(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "state.json").exists()


def value_dtype_name(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return "bool"
    if np.issubdtype(dtype, np.unsignedinteger) and dtype.itemsize <= 1:
        return "uint8"
    if np.issubdtype(dtype, np.integer):
        return "int64"
    if np.issubdtype(dtype, np.floating):
        return "float32"
    raise TypeError(f"Unsupported trajectory dtype: {dtype}")


def fixed_shape_feature(shape: Sequence[int], dtype: np.dtype | str):
    feature = Value(value_dtype_name(np.dtype(dtype)))
    for size in reversed(tuple(int(value) for value in shape)):
        feature = ArrowSequence(feature, length=size)
    return feature


def build_trajectory_features(
    observation_shape: Sequence[int],
    observation_dtype: np.dtype,
    action_shape: Sequence[int],
    action_dtype: np.dtype,
) -> Features:
    return Features(
        {
            "trajectory_id": Value("int64"),
            "step_index": Value("int64"),
            "env_index": Value("int64"),
            "observation": fixed_shape_feature(observation_shape, observation_dtype),
            "action": fixed_shape_feature(action_shape, action_dtype),
            "reward": Value("float32"),
            "terminated": Value("bool"),
            "truncated": Value("bool"),
            "done": Value("bool"),
            "next_observation": fixed_shape_feature(observation_shape, observation_dtype),
        }
    )


def scalar_or_array_value(value: np.ndarray | np.generic | Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path, TRAJECTORY_METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def read_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if is_arrow_trajectory(path):
        metadata_path = path / TRAJECTORY_METADATA_FILE
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
        return {}

    loaded = np.load(path, allow_pickle=False)
    if "metadata" not in loaded:
        return {}
    return json.loads(str(loaded["metadata"]))


def normalize_trajectory_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    observations = np.asarray(arrays["observations"])
    actions = np.asarray(arrays["actions"])
    rewards = np.asarray(arrays["rewards"], dtype=np.float32)
    terminateds = np.asarray(arrays["terminateds"], dtype=np.bool_)
    truncateds = np.asarray(arrays["truncateds"], dtype=np.bool_)
    next_observations = np.asarray(arrays["next_observations"])

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [num_steps, num_envs].")
    if observations.shape[:2] != rewards.shape:
        raise ValueError("observations must start with [num_steps, num_envs].")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations must match observations shape.")
    if actions.shape[:2] != rewards.shape:
        raise ValueError("actions must start with [num_steps, num_envs].")
    if terminateds.shape != rewards.shape or truncateds.shape != rewards.shape:
        raise ValueError("done flags must have shape [num_steps, num_envs].")

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminateds": terminateds,
        "truncateds": truncateds,
        "dones": terminateds | truncateds,
        "next_observations": next_observations,
    }


def load_done_arrays(loaded: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "terminateds" in loaded and "truncateds" in loaded:
        terminateds = np.asarray(loaded["terminateds"], dtype=np.bool_)
        truncateds = np.asarray(loaded["truncateds"], dtype=np.bool_)
        dones = np.asarray(loaded["dones"], dtype=np.bool_)
        return terminateds, truncateds, dones

    dones = np.asarray(loaded["dones"], dtype=np.bool_)
    terminateds = dones.copy()
    truncateds = np.zeros_like(dones, dtype=np.bool_)
    return terminateds, truncateds, dones


def load_npz_trajectory_arrays(path: str | Path) -> dict[str, np.ndarray]:
    loaded = np.load(Path(path), allow_pickle=False)
    terminateds, truncateds, dones = load_done_arrays(loaded)
    return {
        "observations": loaded["observations"],
        "actions": loaded["actions"],
        "rewards": loaded["rewards"],
        "terminateds": terminateds,
        "truncateds": truncateds,
        "dones": dones,
        "next_observations": loaded["next_observations"],
    }


def iter_rows_from_arrays(
    arrays: dict[str, np.ndarray],
    trajectory_offset: int = 0,
    skip_terminated: bool = False,
) -> Iterable[dict[str, Any]]:
    data = normalize_trajectory_arrays(arrays)
    num_steps, num_envs = data["rewards"].shape
    next_trajectory_id = int(trajectory_offset)
    for env_index in range(num_envs):
        trajectory_id = next_trajectory_id
        next_trajectory_id += 1
        saved_step_index = 0
        for source_step_index in range(num_steps):
            terminated = bool(data["terminateds"][source_step_index, env_index])
            truncated = bool(data["truncateds"][source_step_index, env_index])
            done = bool(terminated or truncated)
            if skip_terminated and terminated:
                trajectory_id = next_trajectory_id
                next_trajectory_id += 1
                saved_step_index = 0
                continue
            yield {
                "trajectory_id": int(trajectory_id),
                "step_index": int(saved_step_index),
                "env_index": int(env_index),
                "observation": data["observations"][source_step_index, env_index],
                "action": scalar_or_array_value(data["actions"][source_step_index, env_index]),
                "reward": float(data["rewards"][source_step_index, env_index]),
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "next_observation": data["next_observations"][source_step_index, env_index],
            }
            saved_step_index += 1
            if done:
                trajectory_id = next_trajectory_id
                next_trajectory_id += 1
                saved_step_index = 0


def load_arrow_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    metadata = read_metadata(path)
    dataset = ArrowDataset.load_from_disk(str(path)).with_format("numpy")
    if len(dataset) == 0:
        raise ValueError(f"Trajectory dataset is empty: {path}")

    first = dataset[0]
    observation_dtype = np.dtype(metadata.get("observation_dtype", np.asarray(first["observation"]).dtype))
    action_dtype = np.dtype(metadata.get("action_dtype", np.asarray(first["action"]).dtype))
    has_trajectory_id = "trajectory_id" in dataset.column_names
    rows = []
    for row in dataset:
        trajectory_id = int(row["trajectory_id"]) if has_trajectory_id else int(row["env_index"])
        terminated = bool(row.get("terminated", row.get("done", False)))
        truncated = bool(row.get("truncated", False))
        done = bool(row.get("done", terminated or truncated))
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "step_index": int(row["step_index"]),
                "env_index": int(row["env_index"]),
                "observation": np.asarray(row["observation"], dtype=observation_dtype),
                "action": scalar_or_array_value(np.asarray(row["action"], dtype=action_dtype)),
                "reward": float(row["reward"]),
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "next_observation": np.asarray(row["next_observation"], dtype=observation_dtype),
            }
        )
    return sorted(rows, key=lambda item: (int(item["trajectory_id"]), int(item["step_index"])))


def load_trajectory_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if is_arrow_trajectory(path):
        return load_arrow_rows(path)
    return list(iter_rows_from_arrays(load_npz_trajectory_arrays(path), skip_terminated=False))


def trajectory_metadata_from_rows(rows: Sequence[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    first = rows[0]
    observation_shape = tuple(int(value) for value in np.asarray(first["observation"]).shape)
    action_shape = tuple(int(value) for value in np.asarray(first["action"]).shape)
    observation_dtype = np.asarray(first["observation"]).dtype
    action_dtype = np.asarray(first["action"]).dtype

    trajectory_lengths: dict[int, int] = {}
    for row in rows:
        if tuple(np.asarray(row["observation"]).shape) != observation_shape:
            raise ValueError("All trajectory rows must share the same observation shape.")
        if tuple(np.asarray(row["next_observation"]).shape) != observation_shape:
            raise ValueError("All next observations must share the same observation shape.")
        if tuple(np.asarray(row["action"]).shape) != action_shape:
            raise ValueError("All trajectory rows must share the same action shape.")
        trajectory_id = int(row["trajectory_id"])
        trajectory_lengths[trajectory_id] = trajectory_lengths.get(trajectory_id, 0) + 1

    lengths = [trajectory_lengths[key] for key in sorted(trajectory_lengths)]
    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "trajectory_schema_version": 5,
            "trajectory_storage_format": TRAJECTORY_STORAGE_FORMAT,
            "num_trajectories": int(len(lengths)),
            "num_steps": int(max(lengths)),
            "num_envs": int(len(lengths)),
            "num_transitions": int(len(rows)),
            "trajectory_lengths": [int(value) for value in lengths],
            "observation_shape": list(observation_shape),
            "observation_dtype": str(observation_dtype),
            "action_shape": list(action_shape),
            "action_dtype": str(action_dtype),
            "reward_dtype": "float32",
        }
    )
    return payload_metadata


def save_trajectory_rows(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("At least one trajectory row is required.")

    payload_metadata = trajectory_metadata_from_rows(rows, metadata=metadata)
    features = build_trajectory_features(
        observation_shape=payload_metadata["observation_shape"],
        observation_dtype=np.dtype(payload_metadata["observation_dtype"]),
        action_shape=payload_metadata["action_shape"],
        action_dtype=np.dtype(payload_metadata["action_dtype"]),
    )

    def row_generator():
        for row in rows:
            terminated = bool(row["terminated"])
            truncated = bool(row["truncated"])
            yield {
                "trajectory_id": int(row["trajectory_id"]),
                "step_index": int(row["step_index"]),
                "env_index": int(row["env_index"]),
                "observation": np.asarray(row["observation"]),
                "action": scalar_or_array_value(row["action"]),
                "reward": float(row["reward"]),
                "terminated": terminated,
                "truncated": truncated,
                "done": bool(row.get("done", terminated or truncated)),
                "next_observation": np.asarray(row["next_observation"]),
            }

    dataset = ArrowDataset.from_generator(row_generator, features=features)
    dataset.save_to_disk(str(path))
    write_metadata(path, payload_metadata)


def save_trajectory_dataset(
    path: str | Path,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    next_observations: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    arrays = {
        "observations": np.asarray(observations),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminateds": np.asarray(terminateds, dtype=np.bool_),
        "truncateds": np.asarray(truncateds, dtype=np.bool_),
        "next_observations": np.asarray(next_observations),
    }
    save_trajectory_rows(path, list(iter_rows_from_arrays(arrays, skip_terminated=False)), metadata=metadata)


def load_arrow_trajectory_arrays(path: str | Path) -> dict[str, np.ndarray]:
    rows = load_arrow_rows(path)
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["trajectory_id"]), []).append(row)

    sorted_ids = sorted(groups)
    lengths = [len(groups[trajectory_id]) for trajectory_id in sorted_ids]
    if len(set(lengths)) != 1:
        raise ValueError(f"Trajectory rows do not form a rectangular array: {path}")

    num_steps = lengths[0]
    first = groups[sorted_ids[0]][0]
    observation_shape = tuple(np.asarray(first["observation"]).shape)
    action_shape = tuple(np.asarray(first["action"]).shape)
    observation_dtype = np.asarray(first["observation"]).dtype
    action_dtype = np.asarray(first["action"]).dtype
    num_trajectories = len(sorted_ids)

    observations = np.empty((num_steps, num_trajectories, *observation_shape), dtype=observation_dtype)
    next_observations = np.empty_like(observations)
    actions = np.empty((num_steps, num_trajectories, *action_shape), dtype=action_dtype)
    rewards = np.empty((num_steps, num_trajectories), dtype=np.float32)
    terminateds = np.empty((num_steps, num_trajectories), dtype=np.bool_)
    truncateds = np.empty((num_steps, num_trajectories), dtype=np.bool_)
    dones = np.empty((num_steps, num_trajectories), dtype=np.bool_)

    for trajectory_index, trajectory_id in enumerate(sorted_ids):
        trajectory_rows = sorted(groups[trajectory_id], key=lambda item: int(item["step_index"]))
        expected_steps = list(range(num_steps))
        actual_steps = [int(row["step_index"]) for row in trajectory_rows]
        if actual_steps != expected_steps:
            raise ValueError(f"Trajectory {trajectory_id} has non-contiguous step indices.")
        for step_index, row in enumerate(trajectory_rows):
            observations[step_index, trajectory_index] = np.asarray(row["observation"], dtype=observation_dtype)
            next_observations[step_index, trajectory_index] = np.asarray(row["next_observation"], dtype=observation_dtype)
            actions[step_index, trajectory_index] = np.asarray(row["action"], dtype=action_dtype)
            rewards[step_index, trajectory_index] = np.float32(row["reward"])
            terminateds[step_index, trajectory_index] = bool(row["terminated"])
            truncateds[step_index, trajectory_index] = bool(row["truncated"])
            dones[step_index, trajectory_index] = bool(row["done"])

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminateds": terminateds,
        "truncateds": truncateds,
        "dones": dones,
        "next_observations": next_observations,
    }


def load_trajectory_arrays(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if is_arrow_trajectory(path):
        return load_arrow_trajectory_arrays(path)
    return load_npz_trajectory_arrays(path)


def remap_trajectory_ids(rows: Sequence[dict[str, Any]], trajectory_offset: int) -> tuple[list[dict[str, Any]], int]:
    id_map: dict[int, int] = {}
    next_step_index: dict[int, int] = {}
    remapped = []
    for row in sorted(rows, key=lambda item: (int(item["trajectory_id"]), int(item["step_index"]))):
        old_id = int(row["trajectory_id"])
        if old_id not in id_map:
            id_map[old_id] = trajectory_offset + len(id_map)
        new_id = int(id_map[old_id])
        new_row = dict(row)
        new_row["trajectory_id"] = new_id
        new_row["step_index"] = int(next_step_index.get(new_id, 0))
        next_step_index[new_id] = int(new_row["step_index"]) + 1
        remapped.append(new_row)
    return remapped, trajectory_offset + len(id_map)


def write_mixed_trajectory_dataset(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    trajectory_offset = 0
    for input_path in input_paths:
        input_rows, trajectory_offset = remap_trajectory_ids(load_trajectory_rows(input_path), trajectory_offset)
        rows.extend(input_rows)
    save_trajectory_rows(output_path, rows, metadata=metadata)


def is_supervised_transition_dataset(path: str | Path) -> bool:
    path = Path(path)
    if not is_arrow_trajectory(path):
        return False
    return read_metadata(path).get("dataset_type") == SUPERVISED_DATASET_TYPE


def order_rows_for_supervised_windows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: (int(item["trajectory_id"]), int(item["step_index"])))


def row_has_synthetic_truncation_boundary(
    ordered_rows: Sequence[dict[str, Any]],
    index: int,
) -> bool:
    row = ordered_rows[index]
    if bool(row["done"]):
        return False
    if index + 1 >= len(ordered_rows):
        return True
    next_row = ordered_rows[index + 1]
    return int(next_row["trajectory_id"]) != int(row["trajectory_id"])


def build_supervised_transition_features(
    observation_shape: Sequence[int],
    observation_dtype: np.dtype,
    action_shape: Sequence[int],
    action_dtype: np.dtype,
    horizon: int,
) -> Features:
    return Features(
        {
            "sample_id": Value("int64"),
            "source_trajectory_id": Value("int64"),
            "step_index": Value("int64"),
            "env_index": Value("int64"),
            "observation": fixed_shape_feature(observation_shape, observation_dtype),
            "actions": fixed_shape_feature((horizon, *tuple(action_shape)), action_dtype),
            "targets": fixed_shape_feature((horizon, *tuple(observation_shape)), observation_dtype),
            "dones": fixed_shape_feature((horizon,), np.dtype(np.bool_)),
            "terminateds": fixed_shape_feature((horizon,), np.dtype(np.bool_)),
            "truncateds": fixed_shape_feature((horizon,), np.dtype(np.bool_)),
        }
    )


def build_supervised_transition_rows(
    rows: Sequence[dict[str, Any]],
    horizon: int,
) -> list[dict[str, Any]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    ordered_rows = order_rows_for_supervised_windows(rows)
    window_count = len(ordered_rows) - horizon + 1
    if window_count <= 0:
        return []

    samples: list[dict[str, Any]] = []
    for sample_id, start_index in enumerate(range(window_count)):
        window = ordered_rows[start_index : start_index + horizon]
        first_row = window[0]
        actions = []
        targets = []
        dones = []
        terminateds = []
        truncateds = []
        for offset, row in enumerate(window):
            row_index = start_index + offset
            synthetic_truncated = row_has_synthetic_truncation_boundary(ordered_rows, row_index)
            actions.append(scalar_or_array_value(row["action"]))
            targets.append(np.asarray(row["next_observation"]))
            dones.append(bool(row["done"]) or synthetic_truncated)
            terminateds.append(bool(row["terminated"]))
            truncateds.append(bool(row["truncated"]) or synthetic_truncated)
        samples.append(
            {
                "sample_id": int(sample_id),
                "source_trajectory_id": int(first_row["trajectory_id"]),
                "step_index": int(first_row["step_index"]),
                "env_index": int(first_row["env_index"]),
                "observation": np.asarray(first_row["observation"]),
                "actions": np.asarray(actions),
                "targets": np.asarray(targets),
                "dones": np.asarray(dones, dtype=np.bool_),
                "terminateds": np.asarray(terminateds, dtype=np.bool_),
                "truncateds": np.asarray(truncateds, dtype=np.bool_),
            }
        )
    return samples


def write_supervised_transition_dataset(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    horizon: int = 1,
    metadata: dict[str, Any] | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    trajectory_offset = 0
    for input_path in input_paths:
        input_rows, trajectory_offset = remap_trajectory_ids(load_trajectory_rows(input_path), trajectory_offset)
        rows.extend(input_rows)
    if not rows:
        raise ValueError("At least one raw transition row is required.")

    samples = build_supervised_transition_rows(rows, horizon=horizon)
    if not samples:
        raise ValueError("At least one supervised transition sample is required.")

    first = samples[0]
    observation_shape = tuple(int(value) for value in np.asarray(first["observation"]).shape)
    action_shape = tuple(int(value) for value in np.asarray(first["actions"]).shape[1:])
    observation_dtype = np.asarray(first["observation"]).dtype
    action_dtype = np.asarray(first["actions"]).dtype
    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "dataset_type": SUPERVISED_DATASET_TYPE,
            "supervised_schema_version": 1,
            "horizon": int(horizon),
            "num_samples": int(len(samples)),
            "num_raw_transitions": int(len(rows)),
            "observation_shape": list(observation_shape),
            "observation_dtype": str(observation_dtype),
            "action_shape": list(action_shape),
            "action_dtype": str(action_dtype),
        }
    )
    features = build_supervised_transition_features(
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        horizon=horizon,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def row_generator():
        for sample in samples:
            yield {
                "sample_id": int(sample["sample_id"]),
                "source_trajectory_id": int(sample["source_trajectory_id"]),
                "step_index": int(sample["step_index"]),
                "env_index": int(sample["env_index"]),
                "observation": np.asarray(sample["observation"]),
                "actions": np.asarray(sample["actions"]),
                "targets": np.asarray(sample["targets"]),
                "dones": np.asarray(sample["dones"], dtype=np.bool_),
                "terminateds": np.asarray(sample["terminateds"], dtype=np.bool_),
                "truncateds": np.asarray(sample["truncateds"], dtype=np.bool_),
            }

    dataset = ArrowDataset.from_generator(row_generator, features=features)
    dataset.save_to_disk(str(output_path))
    write_metadata(output_path, payload_metadata)


def load_supervised_transition_rows(path: str | Path) -> list[dict[str, Any]]:
    dataset = ArrowDataset.load_from_disk(str(path)).with_format("numpy")
    if len(dataset) == 0:
        raise ValueError(f"Supervised transition dataset is empty: {path}")
    rows = []
    for row in dataset:
        rows.append(
            {
                "observation": np.asarray(row["observation"]),
                "actions": np.asarray(row["actions"]),
                "targets": np.asarray(row["targets"]),
                "dones": np.asarray(row["dones"], dtype=np.bool_),
                "terminateds": np.asarray(row["terminateds"], dtype=np.bool_),
                "truncateds": np.asarray(row["truncateds"], dtype=np.bool_),
            }
        )
    return rows


class TrajectoryBuffer:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.terminateds: list[np.ndarray] = []
        self.truncateds: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []
        self.next_observations: list[np.ndarray] = []

    @property
    def num_steps(self) -> int:
        return len(self.observations)

    @property
    def num_transitions(self) -> int:
        if not self.observations:
            return 0
        return len(self.observations) * int(self.observations[0].shape[0])

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        next_observation: np.ndarray,
    ) -> None:
        terminated = np.asarray(terminated, dtype=np.bool_)
        truncated = np.asarray(truncated, dtype=np.bool_)
        self.observations.append(np.asarray(observation).copy())
        self.actions.append(np.asarray(action).copy())
        self.rewards.append(np.asarray(reward, dtype=np.float32).copy())
        self.terminateds.append(terminated.copy())
        self.truncateds.append(truncated.copy())
        self.dones.append((terminated | truncated).copy())
        self.next_observations.append(np.asarray(next_observation).copy())

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        save_trajectory_dataset(
            path=path,
            observations=np.stack(self.observations, axis=0),
            actions=np.stack(self.actions, axis=0),
            rewards=np.stack(self.rewards, axis=0),
            terminateds=np.stack(self.terminateds, axis=0),
            truncateds=np.stack(self.truncateds, axis=0),
            next_observations=np.stack(self.next_observations, axis=0),
            metadata=metadata,
        )


class TrajectoryRecorderCallback(BaseCallback):
    def __init__(
        self,
        save_path: str | Path | None = None,
        max_transitions: int | None = None,
        stop_when_full: bool = False,
        verbose: int = 0,
        log_fn: Callable[[dict[str, float | int], int], None] | None = None,
        log_interval: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.save_path = Path(save_path) if save_path is not None else None
        self.max_transitions = max_transitions
        self.stop_when_full = stop_when_full
        self.log_fn = log_fn
        self.log_interval = int(log_interval)
        self.buffer = TrajectoryBuffer()

    def _on_step(self) -> bool:
        if self.max_transitions is None or self.buffer.num_transitions < self.max_transitions:
            infos = self.locals.get("infos")
            terminateds, truncateds, _ = split_done_flags(self.locals["dones"], infos)
            rewards = np.asarray(self.locals["rewards"], dtype=np.float32)
            self.buffer.add(
                observation=self.model._last_obs,
                action=self.locals["actions"],
                reward=rewards,
                terminated=terminateds,
                truncated=truncateds,
                next_observation=resolve_terminal_next_observations(
                    self.locals["new_obs"],
                    infos,
                ),
            )
            if self.log_fn is not None and self.log_interval > 0 and self.n_calls % self.log_interval == 0:
                self.log_fn(
                    {
                        "num_timesteps": int(self.num_timesteps),
                        "trajectory_transitions": int(self.buffer.num_transitions),
                        "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
                    },
                    int(self.num_timesteps),
                )
        if self.max_transitions is None or not self.stop_when_full:
            return True
        return self.buffer.num_transitions < self.max_transitions

    def save(self, metadata: dict[str, Any] | None = None) -> None:
        if self.save_path is None:
            raise ValueError("save_path was not configured.")
        self.buffer.save(self.save_path, metadata=metadata)


def sample_action_batch(action_space: gym.Space, n_envs: int) -> np.ndarray:
    if isinstance(action_space, gym.spaces.Discrete):
        return np.asarray([action_space.sample() for _ in range(n_envs)], dtype=np.int64)
    if isinstance(action_space, gym.spaces.Box):
        return np.stack([action_space.sample() for _ in range(n_envs)], axis=0)
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return np.stack([action_space.sample() for _ in range(n_envs)], axis=0)
    if isinstance(action_space, gym.spaces.MultiBinary):
        return np.stack([action_space.sample() for _ in range(n_envs)], axis=0)
    raise TypeError(f"Unsupported action space: {type(action_space).__name__}")


def collect_random_trajectories(
    env: VecEnv,
    num_steps: int,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryBuffer:
    buffer = TrajectoryBuffer()
    observation = env.reset()
    for _ in range(num_steps):
        actions = sample_action_batch(env.action_space, env.num_envs)
        next_observation, rewards, dones, infos = env.step(actions)
        terminateds, truncateds, _ = split_done_flags(dones, infos)
        buffer.add(
            observation=observation,
            action=actions,
            reward=rewards,
            terminated=terminateds,
            truncated=truncateds,
            next_observation=resolve_terminal_next_observations(next_observation, infos),
        )
        observation = next_observation
    buffer.save(output_path, metadata=metadata)
    return buffer


class TransitionDataset(TorchDataset):
    def __init__(self, trajectory_files: Sequence[str | Path]):
        self.rows: list[dict[str, Any]] = []
        self.horizon = 1
        for path in trajectory_files:
            if is_supervised_transition_dataset(path):
                supervised_rows = load_supervised_transition_rows(path)
                self.rows.extend(supervised_rows)
                self.horizon = int(np.asarray(supervised_rows[0]["actions"]).shape[0])
            else:
                raw_rows = load_trajectory_rows(path)
                self.rows.extend(self._one_step_rows(raw_rows))
                self.horizon = 1
        if not self.rows:
            raise ValueError("No transitions were found.")

    @staticmethod
    def _one_step_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        one_step_rows = []
        for row in rows:
            one_step_rows.append(
                {
                    "observation": row["observation"],
                    "actions": np.expand_dims(np.asarray(row["action"]), axis=0),
                    "targets": np.expand_dims(np.asarray(row["next_observation"]), axis=0),
                    "dones": np.asarray([bool(row["done"])]),
                    "terminateds": np.asarray([bool(row["terminated"])]),
                    "truncateds": np.asarray([bool(row["truncated"])]),
                }
            )
        return one_step_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        actions = torch.as_tensor(row["actions"])
        targets = torch.stack([self._to_image_tensor(target) for target in np.asarray(row["targets"])], dim=0)
        dones = torch.as_tensor(np.asarray(row["dones"], dtype=np.bool_), dtype=torch.bool)
        terminateds = torch.as_tensor(np.asarray(row["terminateds"], dtype=np.bool_), dtype=torch.bool)
        truncateds = torch.as_tensor(np.asarray(row["truncateds"], dtype=np.bool_), dtype=torch.bool)
        item = {
            "observation": self._to_image_tensor(row["observation"]),
            "actions": actions,
            "targets": targets,
            "dones": dones,
            "terminateds": terminateds,
            "truncateds": truncateds,
        }
        if actions.shape[0] == 1:
            item["action"] = actions[0]
            item["target"] = targets[0]
            item["done"] = dones[0]
            item["terminated"] = terminateds[0]
            item["truncated"] = truncateds[0]
        return item

    @staticmethod
    def _to_image_tensor(array: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(array)
        if tensor.dtype == torch.uint8:
            return tensor.float().div(255.0)
        if not torch.is_floating_point(tensor):
            return tensor.float().div(255.0)
        if tensor.numel() > 0 and float(tensor.detach().max()) > 1.0:
            return tensor.float().div(255.0)
        return tensor.float()


def count_trajectory_file(path: str | Path) -> dict[str, int]:
    path = Path(path)
    if is_supervised_transition_dataset(path):
        dataset = ArrowDataset.load_from_disk(str(path))
        metadata = read_metadata(path)
        if len(dataset) == 0:
            return {
                "num_steps": 0,
                "num_envs": 0,
                "num_transitions": 0,
                "num_samples": 0,
                "num_terminated": 0,
                "num_truncated": 0,
                "num_done": 0,
                "num_trajectories": 0,
            }
        env_indices = np.asarray(dataset["env_index"], dtype=np.int64)
        terminateds = np.asarray(dataset["terminateds"], dtype=np.bool_)
        truncateds = np.asarray(dataset["truncateds"], dtype=np.bool_)
        dones = np.asarray(dataset["dones"], dtype=np.bool_)
        return {
            "num_steps": int(metadata.get("horizon", int(dones.shape[-1]))),
            "num_envs": int(np.unique(env_indices).size),
            "num_transitions": int(len(dataset)),
            "num_samples": int(len(dataset)),
            "num_terminated": int(terminateds.sum()),
            "num_truncated": int(truncateds.sum()),
            "num_done": int(dones.sum()),
            "num_trajectories": int(np.unique(env_indices).size),
        }

    if is_arrow_trajectory(path):
        dataset = ArrowDataset.load_from_disk(str(path))
        metadata = read_metadata(path)
        if len(dataset) == 0:
            return {
                "num_steps": 0,
                "num_envs": 0,
                "num_transitions": 0,
                "num_terminated": 0,
                "num_truncated": 0,
                "num_done": 0,
                "num_trajectories": 0,
            }
        step_indices = np.asarray(dataset["step_index"], dtype=np.int64)
        trajectory_ids = np.asarray(dataset["trajectory_id"], dtype=np.int64) if "trajectory_id" in dataset.column_names else np.asarray(dataset["env_index"], dtype=np.int64)
        terminateds = np.asarray(dataset["terminated"], dtype=np.bool_)
        truncateds = np.asarray(dataset["truncated"], dtype=np.bool_)
        dones = np.asarray(dataset["done"], dtype=np.bool_)
        return {
            "num_steps": int(metadata.get("num_steps", int(step_indices.max()) + 1)),
            "num_envs": int(metadata.get("num_envs", int(np.unique(trajectory_ids).size))),
            "num_transitions": int(len(dataset)),
            "num_terminated": int(terminateds.sum()),
            "num_truncated": int(truncateds.sum()),
            "num_done": int(dones.sum()),
            "num_trajectories": int(np.unique(trajectory_ids).size),
        }

    arrays = load_npz_trajectory_arrays(path)
    rewards = arrays["rewards"]
    num_steps = int(rewards.shape[0])
    num_envs = int(rewards.shape[1]) if rewards.ndim > 1 else 1
    return {
        "num_steps": num_steps,
        "num_envs": num_envs,
        "num_transitions": num_steps * num_envs,
        "num_terminated": int(arrays["terminateds"].sum()),
        "num_truncated": int(arrays["truncateds"].sum()),
        "num_done": int(arrays["dones"].sum()),
        "num_trajectories": num_envs,
    }


def write_trajectory_prefix(
    input_path: str | Path,
    output_path: str | Path,
    num_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    input_path = Path(input_path)
    rows = load_trajectory_rows(input_path)
    source_metadata = read_metadata(input_path)
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["trajectory_id"]), []).append(row)
    max_source_steps = max(len(group_rows) for group_rows in groups.values())
    selected_steps = max_source_steps if num_steps is None or num_steps <= 0 else min(num_steps, max_source_steps)

    selected_rows: list[dict[str, Any]] = []
    for trajectory_id in sorted(groups):
        trajectory_rows = sorted(groups[trajectory_id], key=lambda item: int(item["step_index"]))
        for step_index, row in enumerate(trajectory_rows[:selected_steps]):
            selected_row = dict(row)
            selected_row["step_index"] = int(step_index)
            selected_rows.append(selected_row)

    merged_metadata = dict(source_metadata)
    if metadata:
        merged_metadata.update(metadata)
    merged_metadata["source_trajectory_file"] = str(input_path)
    merged_metadata["selected_steps"] = int(selected_steps)
    merged_metadata["source_steps"] = int(max_source_steps)
    save_trajectory_rows(output_path, selected_rows, metadata=merged_metadata)
