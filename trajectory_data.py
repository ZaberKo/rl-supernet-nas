from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

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


def trajectory_rows(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    dones: np.ndarray,
    next_observations: np.ndarray,
):
    num_steps, num_envs = rewards.shape[:2]
    for step_index in range(num_steps):
        for env_index in range(num_envs):
            yield {
                "step_index": int(step_index),
                "env_index": int(env_index),
                "observation": observations[step_index, env_index],
                "action": scalar_or_array_value(actions[step_index, env_index]),
                "reward": float(rewards[step_index, env_index]),
                "terminated": bool(terminateds[step_index, env_index]),
                "truncated": bool(truncateds[step_index, env_index]),
                "done": bool(dones[step_index, env_index]),
                "next_observation": next_observations[step_index, env_index],
            }


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path, TRAJECTORY_METADATA_FILE).write_text(json.dumps(metadata, indent=2))


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
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    observations = np.asarray(observations)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards, dtype=np.float32)
    terminateds = np.asarray(terminateds, dtype=np.bool_)
    truncateds = np.asarray(truncateds, dtype=np.bool_)
    next_observations = np.asarray(next_observations)
    dones = terminateds | truncateds

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [num_steps, num_envs].")
    num_steps, num_envs = rewards.shape
    observation_shape = tuple(int(value) for value in observations.shape[2:])
    action_shape = tuple(int(value) for value in actions.shape[2:])

    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "trajectory_schema_version": 3,
            "trajectory_storage_format": TRAJECTORY_STORAGE_FORMAT,
            "num_steps": int(num_steps),
            "num_envs": int(num_envs),
            "num_transitions": int(num_steps * num_envs),
            "observation_shape": list(observation_shape),
            "observation_dtype": str(observations.dtype),
            "action_shape": list(action_shape),
            "action_dtype": str(actions.dtype),
            "reward_dtype": "float32",
        }
    )

    features = build_trajectory_features(
        observation_shape=observation_shape,
        observation_dtype=observations.dtype,
        action_shape=action_shape,
        action_dtype=actions.dtype,
    )
    dataset = ArrowDataset.from_generator(
        lambda: trajectory_rows(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            dones=dones,
            next_observations=next_observations,
        ),
        features=features,
    )
    dataset.save_to_disk(str(path))
    write_metadata(path, payload_metadata)


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


def load_arrow_trajectory_arrays(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    metadata = read_metadata(path)
    dataset = ArrowDataset.load_from_disk(str(path)).with_format("numpy")
    if len(dataset) == 0:
        raise ValueError(f"Trajectory dataset is empty: {path}")

    step_indices = np.asarray(dataset["step_index"], dtype=np.int64)
    env_indices = np.asarray(dataset["env_index"], dtype=np.int64)
    num_steps = int(metadata.get("num_steps", int(step_indices.max()) + 1))
    num_envs = int(metadata.get("num_envs", int(env_indices.max()) + 1))

    first = dataset[0]
    observation_dtype = np.dtype(metadata.get("observation_dtype", np.asarray(first["observation"]).dtype))
    action_dtype = np.dtype(metadata.get("action_dtype", np.asarray(first["action"]).dtype))
    observation_shape = tuple(int(value) for value in metadata.get("observation_shape", np.asarray(first["observation"]).shape))
    action_shape = tuple(int(value) for value in metadata.get("action_shape", np.asarray(first["action"]).shape))

    observations = np.empty((num_steps, num_envs, *observation_shape), dtype=observation_dtype)
    next_observations = np.empty_like(observations)
    actions = np.empty((num_steps, num_envs, *action_shape), dtype=action_dtype)
    rewards = np.empty((num_steps, num_envs), dtype=np.float32)
    terminateds = np.empty((num_steps, num_envs), dtype=np.bool_)
    truncateds = np.empty((num_steps, num_envs), dtype=np.bool_)
    dones = np.empty((num_steps, num_envs), dtype=np.bool_)

    seen = np.zeros((num_steps, num_envs), dtype=np.bool_)
    for row in dataset:
        step_index = int(row["step_index"])
        env_index = int(row["env_index"])
        observations[step_index, env_index] = np.asarray(row["observation"], dtype=observation_dtype)
        next_observations[step_index, env_index] = np.asarray(row["next_observation"], dtype=observation_dtype)
        actions[step_index, env_index] = np.asarray(row["action"], dtype=action_dtype)
        rewards[step_index, env_index] = np.float32(row["reward"])
        terminateds[step_index, env_index] = bool(row["terminated"])
        truncateds[step_index, env_index] = bool(row["truncated"])
        dones[step_index, env_index] = bool(row["done"])
        seen[step_index, env_index] = True

    if not bool(seen.all()):
        raise ValueError(f"Trajectory dataset has missing step/env rows: {path}")

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
    ):
        super().__init__(verbose=verbose)
        self.save_path = Path(save_path) if save_path is not None else None
        self.max_transitions = max_transitions
        self.stop_when_full = stop_when_full
        self.buffer = TrajectoryBuffer()

    def _on_step(self) -> bool:
        if self.max_transitions is None or self.buffer.num_transitions < self.max_transitions:
            infos = self.locals.get("infos")
            terminateds, truncateds, _ = split_done_flags(self.locals["dones"], infos)
            self.buffer.add(
                observation=self.model._last_obs,
                action=self.locals["actions"],
                reward=self.locals["rewards"],
                terminated=terminateds,
                truncated=truncateds,
                next_observation=resolve_terminal_next_observations(
                    self.locals["new_obs"],
                    infos,
                ),
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


class TrajectoryDataset(TorchDataset):
    def __init__(self, trajectory_files: Sequence[str | Path], horizon: int):
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.files: list[dict[str, np.ndarray]] = []
        self.indices: list[tuple[int, int, int]] = []
        for file_index, path in enumerate(trajectory_files):
            data = load_trajectory_arrays(path)
            self.files.append(data)
            dones = data["dones"]
            num_steps, num_envs = dones.shape[:2]
            for step in range(num_steps - self.horizon + 1):
                done_window = dones[step : step + self.horizon]
                for env_index in range(num_envs):
                    if not bool(done_window[:, env_index].any()):
                        self.indices.append((file_index, step, env_index))
        if not self.indices:
            raise ValueError("No valid trajectory windows were found.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_index, step, env_index = self.indices[index]
        data = self.files[file_index]
        observation = data["observations"][step, env_index]
        actions = data["actions"][step : step + self.horizon, env_index]
        targets = data["next_observations"][step : step + self.horizon, env_index]
        return {
            "observation": self._to_image_tensor(observation),
            "actions": torch.as_tensor(actions),
            "targets": self._to_image_tensor(targets),
        }

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
            }
        step_indices = np.asarray(dataset["step_index"], dtype=np.int64)
        env_indices = np.asarray(dataset["env_index"], dtype=np.int64)
        terminateds = np.asarray(dataset["terminated"], dtype=np.bool_)
        truncateds = np.asarray(dataset["truncated"], dtype=np.bool_)
        dones = np.asarray(dataset["done"], dtype=np.bool_)
        return {
            "num_steps": int(metadata.get("num_steps", int(step_indices.max()) + 1)),
            "num_envs": int(metadata.get("num_envs", int(env_indices.max()) + 1)),
            "num_transitions": int(len(dataset)),
            "num_terminated": int(terminateds.sum()),
            "num_truncated": int(truncateds.sum()),
            "num_done": int(dones.sum()),
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
    }


def write_trajectory_prefix(
    input_path: str | Path,
    output_path: str | Path,
    num_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    arrays = load_trajectory_arrays(input_path)
    total_steps = int(arrays["rewards"].shape[0])
    selected_steps = total_steps if num_steps is None or num_steps <= 0 else min(num_steps, total_steps)
    source_metadata = read_metadata(input_path)
    merged_metadata = dict(source_metadata)
    if metadata:
        merged_metadata.update(metadata)
    merged_metadata["source_trajectory_file"] = str(input_path)
    merged_metadata["selected_steps"] = selected_steps
    merged_metadata["source_steps"] = total_steps
    save_trajectory_dataset(
        path=output_path,
        observations=arrays["observations"][:selected_steps],
        actions=arrays["actions"][:selected_steps],
        rewards=arrays["rewards"][:selected_steps],
        terminateds=arrays["terminateds"][:selected_steps],
        truncateds=arrays["truncateds"][:selected_steps],
        next_observations=arrays["next_observations"][:selected_steps],
        metadata=merged_metadata,
    )
