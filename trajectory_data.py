from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Callable, Iterator, Sequence

import gymnasium as gym
import h5py
import numpy as np
import torch
from datasets import Dataset as ArrowDataset
from datasets import Features, Sequence as ArrowSequence, Value
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.data import Dataset as TorchDataset


TRAJECTORY_METADATA_FILE = "metadata.json"
SUPERVISED_DATASET_TYPE = "supervised_transition_samples"
HDF5_REPRESENTATION_STORAGE_FORMAT = "hdf5"
HDF5_REPRESENTATION_SCHEMA_VERSION = 1
HDF5_REPRESENTATION_CHUNK_SIZE = 1024
HDF5_REPRESENTATION_TARGET_CHUNK_BYTES = 8 * 1024 * 1024
HDF5_REPRESENTATION_COMPRESSION = "gzip"
HDF5_REPRESENTATION_COMPRESSION_OPTS = 4
HDF5_SAMPLE_ARRAYS = (
    "observation",
    "actions",
    "targets",
    "dones",
    "terminateds",
    "truncateds",
    "env_index",
    "source_step",
)
HDF5_FLAG_ARRAYS = ("dones", "terminateds", "truncateds")


def split_done_flags(
    dones: np.ndarray,
    infos: Sequence[dict[str, Any]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    done_array = np.asarray(dones, dtype=np.bool_)
    info_items = list(infos or [{} for _ in range(done_array.shape[0])])
    truncated = np.asarray(
        [bool(info.get("TimeLimit.truncated", False)) for info in info_items],
        dtype=np.bool_,
    )
    terminated = done_array & ~truncated
    return terminated, truncated


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


def scalar_or_array_value(value: np.ndarray | np.generic | Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path, TRAJECTORY_METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def read_hdf5_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or not h5py.is_hdf5(path):
        raise ValueError(f"Expected an HDF5 representation dataset: {path}")
    with h5py.File(path, "r") as dataset_file:
        if dataset_file.attrs.get("dataset_type") != SUPERVISED_DATASET_TYPE:
            raise ValueError(f"Expected a supervised HDF5 representation dataset: {path}")
        raw_metadata = dataset_file.attrs.get("metadata_json", "{}")
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(str(raw_metadata))
        metadata.setdefault("dataset_type", dataset_file.attrs.get("dataset_type", SUPERVISED_DATASET_TYPE))
        metadata.setdefault("trajectory_storage_format", dataset_file.attrs.get("trajectory_storage_format", HDF5_REPRESENTATION_STORAGE_FORMAT))
        metadata.setdefault("supervised_schema_version", int(dataset_file.attrs.get("supervised_schema_version", HDF5_REPRESENTATION_SCHEMA_VERSION)))
        metadata.setdefault("horizon", int(dataset_file.attrs.get("horizon", 1)))
        if "observation" in dataset_file:
            metadata.setdefault("num_samples", int(dataset_file["observation"].shape[0]))
            metadata.setdefault("observation_shape", [int(value) for value in dataset_file["observation"].shape[1:]])
            metadata.setdefault("observation_dtype", str(dataset_file["observation"].dtype))
        else:
            metadata.setdefault("num_samples", int(dataset_file.attrs.get("num_samples", 0)))
        if "actions" in dataset_file:
            action_shape = dataset_file["actions"].shape[2:]
            metadata.setdefault("action_shape", [int(value) for value in action_shape])
            metadata.setdefault("action_dtype", str(dataset_file["actions"].dtype))
    return metadata


def hdf5_base_metadata(
    horizon: int,
    chunk_size: int = HDF5_REPRESENTATION_CHUNK_SIZE,
    target_chunk_bytes: int = HDF5_REPRESENTATION_TARGET_CHUNK_BYTES,
) -> dict[str, Any]:
    return {
        "dataset_type": SUPERVISED_DATASET_TYPE,
        "supervised_schema_version": HDF5_REPRESENTATION_SCHEMA_VERSION,
        "trajectory_storage_format": HDF5_REPRESENTATION_STORAGE_FORMAT,
        "horizon": int(horizon),
        "chunk_size": int(chunk_size),
        "target_chunk_bytes": int(target_chunk_bytes),
        "compression": HDF5_REPRESENTATION_COMPRESSION,
        "compression_opts": HDF5_REPRESENTATION_COMPRESSION_OPTS,
    }


def write_hdf5_metadata(
    dataset_file: h5py.File,
    metadata: dict[str, Any],
    horizon: int,
    num_samples: int,
    chunk_size: int,
    target_chunk_bytes: int,
) -> None:
    payload = hdf5_base_metadata(horizon, chunk_size, target_chunk_bytes)
    payload.update(metadata)
    payload["num_samples"] = int(num_samples)
    for key in (
        "dataset_type",
        "supervised_schema_version",
        "trajectory_storage_format",
        "horizon",
        "num_samples",
        "chunk_size",
        "target_chunk_bytes",
        "compression",
        "compression_opts",
    ):
        dataset_file.attrs[key] = payload[key]
    dataset_file.attrs["metadata_json"] = json.dumps(payload)


def hdf5_compression_kwargs() -> dict[str, Any]:
    return {
        "compression": HDF5_REPRESENTATION_COMPRESSION,
        "compression_opts": HDF5_REPRESENTATION_COMPRESSION_OPTS,
    }


def hdf5_chunk_shape(values: np.ndarray, max_chunk_samples: int, target_chunk_bytes: int) -> tuple[int, ...]:
    sample_shape = tuple(int(value) for value in values.shape[1:])
    sample_bytes = max(1, int(np.dtype(values.dtype).itemsize * np.prod(sample_shape, dtype=np.int64)))
    chunk_samples = max(1, min(int(max_chunk_samples), int(target_chunk_bytes) // sample_bytes))
    return (chunk_samples, *sample_shape)


def empty_supervised_count(horizon: int) -> dict[str, int]:
    empty_flags = np.zeros((0, int(horizon)), dtype=np.bool_)
    return supervised_count(
        env_indices=np.asarray([], dtype=np.int64),
        dones=empty_flags,
        terminateds=empty_flags,
        truncateds=empty_flags,
        horizon=int(horizon),
    )


def supervised_count(
    env_indices: np.ndarray,
    dones: np.ndarray,
    terminateds: np.ndarray,
    truncateds: np.ndarray,
    horizon: int,
) -> dict[str, int]:
    env_indices = np.asarray(env_indices, dtype=np.int64)
    dones = np.asarray(dones, dtype=np.bool_)
    terminateds = np.asarray(terminateds, dtype=np.bool_)
    truncateds = np.asarray(truncateds, dtype=np.bool_)
    num_samples = int(dones.shape[0])
    return {
        "num_steps": int(horizon),
        "num_envs": int(np.unique(env_indices).size) if num_samples > 0 else 0,
        "num_transitions": num_samples,
        "num_samples": num_samples,
        "num_terminated": int(terminateds.sum()),
        "num_truncated": int(truncateds.sum()),
        "num_done": int(dones.sum()),
        "num_trajectories": int(np.unique(env_indices).size) if num_samples > 0 else 0,
    }


def read_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_file() and h5py.is_hdf5(path):
        return read_hdf5_metadata(path)
    if path.is_dir() and (path / "state.json").exists():
        metadata_path = path / TRAJECTORY_METADATA_FILE
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
        return {}
    raise ValueError(f"Unsupported dataset path: {path}")


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


def _resolve_source_steps(arrays: dict[str, np.ndarray], num_steps: int, num_envs: int) -> np.ndarray:
    source_steps = arrays.get("source_steps")
    if source_steps is None:
        return np.broadcast_to(
            np.arange(num_steps, dtype=np.int64)[:, None],
            (num_steps, num_envs),
        )
    source_steps = np.asarray(source_steps, dtype=np.int64)
    if source_steps.shape == (num_steps,):
        return np.broadcast_to(source_steps[:, None], (num_steps, num_envs))
    if source_steps.shape != (num_steps, num_envs):
        raise ValueError("source_steps must have shape [num_steps] or [num_steps, num_envs].")
    return source_steps


def _resolve_env_indices(arrays: dict[str, np.ndarray], num_envs: int) -> np.ndarray:
    env_indices = arrays.get("env_indices")
    if env_indices is None:
        return np.arange(num_envs, dtype=np.int64)
    env_indices = np.asarray(env_indices, dtype=np.int64)
    if env_indices.shape != (num_envs,):
        raise ValueError("env_indices must have shape [num_envs].")
    return env_indices


def iter_supervised_transition_samples_from_arrays(
    arrays: dict[str, np.ndarray],
    horizon: int,
    max_start_transitions: int | None = None,
) -> Iterator[dict[str, Any]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    observations = np.asarray(arrays["observations"])
    actions = np.asarray(arrays["actions"])
    terminateds = np.asarray(arrays["terminateds"], dtype=np.bool_)
    truncateds = np.asarray(arrays["truncateds"], dtype=np.bool_)
    next_observations = np.asarray(arrays["next_observations"])
    if observations.ndim < 2:
        raise ValueError("observations must start with [num_steps, num_envs].")
    num_steps, num_envs = int(observations.shape[0]), int(observations.shape[1])
    if actions.shape[:2] != (num_steps, num_envs):
        raise ValueError("actions must start with [num_steps, num_envs].")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations must match observations shape.")
    if terminateds.shape != (num_steps, num_envs) or truncateds.shape != (num_steps, num_envs):
        raise ValueError("done flags must have shape [num_steps, num_envs].")
    source_steps = _resolve_source_steps(arrays, num_steps, num_envs)
    env_indices = _resolve_env_indices(arrays, num_envs)

    window_count = num_steps - int(horizon) + 1
    if window_count <= 0:
        return

    start_budget = num_steps * num_envs
    if max_start_transitions is not None and max_start_transitions > 0:
        start_budget = min(start_budget, int(max_start_transitions))
    used_start_transitions = 0
    for env_index in range(num_envs):
        for start_index in range(window_count):
            if used_start_transitions >= start_budget:
                return
            end_index = start_index + int(horizon)
            sample_terminateds = terminateds[start_index:end_index, env_index]
            sample_truncateds = truncateds[start_index:end_index, env_index]
            source_step = int(source_steps[start_index, env_index])
            output_env_index = int(env_indices[env_index])
            yield {
                "sample_id": int(used_start_transitions),
                "source_trajectory_id": output_env_index,
                "step_index": source_step,
                "env_index": output_env_index,
                "source_step": source_step,
                "observation": np.asarray(observations[start_index, env_index]),
                "actions": np.asarray(actions[start_index:end_index, env_index]),
                "targets": np.asarray(next_observations[start_index:end_index, env_index]),
                "dones": np.asarray(sample_terminateds | sample_truncateds, dtype=np.bool_),
                "terminateds": np.asarray(sample_terminateds, dtype=np.bool_),
                "truncateds": np.asarray(sample_truncateds, dtype=np.bool_),
            }
            used_start_transitions += 1


def build_supervised_transition_samples_from_arrays(
    arrays: dict[str, np.ndarray],
    horizon: int,
    max_start_transitions: int | None = None,
) -> list[dict[str, Any]]:
    samples = list(
        iter_supervised_transition_samples_from_arrays(
            arrays,
            horizon=horizon,
            max_start_transitions=max_start_transitions,
        )
    )
    for sample_id, sample in enumerate(samples):
        sample["sample_id"] = int(sample_id)
    return samples


def write_supervised_transition_samples(
    samples: Sequence[dict[str, Any]],
    output_path: str | Path,
    horizon: int | None = None,
    metadata: dict[str, Any] | None = None,
    num_raw_transitions: int | None = None,
) -> None:
    if num_raw_transitions is None or num_raw_transitions <= 0:
        num_raw_transitions = len(samples)
    if num_raw_transitions <= 0:
        raise ValueError("At least one source transition or sample is required.")
    if not samples:
        raise ValueError("At least one supervised transition sample is required.")

    first = samples[0]
    inferred_horizon = int(np.asarray(first["actions"]).shape[0])
    if horizon is None:
        horizon = inferred_horizon
    if int(horizon) != inferred_horizon:
        raise ValueError("horizon does not match the sample action shape.")
    horizon = int(horizon)
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
            "num_raw_transitions": int(num_raw_transitions),
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
        for default_sample_id, sample in enumerate(samples):
            yield {
                "sample_id": int(sample.get("sample_id", default_sample_id)),
                "source_trajectory_id": int(sample.get("source_trajectory_id", sample.get("env_index", 0))),
                "step_index": int(sample.get("step_index", sample.get("source_step", default_sample_id))),
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


def write_supervised_transition_samples_from_hdf5_files(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    horizon: int | None = None,
    metadata: dict[str, Any] | None = None,
    num_raw_transitions: int | None = None,
) -> None:
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("At least one HDF5 input path is required.")

    input_metadata = [read_hdf5_metadata(path) for path in paths]
    total_samples = sum(int(item.get("num_samples", 0)) for item in input_metadata)
    if total_samples <= 0:
        raise ValueError("At least one supervised transition sample is required.")

    first_metadata = next(item for item in input_metadata if int(item.get("num_samples", 0)) > 0)
    input_horizon = int(first_metadata.get("horizon", horizon if horizon is not None else 1))
    if horizon is not None and int(horizon) != input_horizon:
        raise ValueError("horizon does not match the HDF5 input horizon.")
    for item in input_metadata:
        if int(item.get("num_samples", 0)) > 0 and int(item.get("horizon", input_horizon)) != input_horizon:
            raise ValueError("All HDF5 inputs must use the same horizon.")

    observation_shape = tuple(int(value) for value in first_metadata["observation_shape"])
    observation_dtype = np.dtype(first_metadata["observation_dtype"])
    action_shape = tuple(int(value) for value in first_metadata["action_shape"])
    action_dtype = np.dtype(first_metadata["action_dtype"])
    if num_raw_transitions is None or num_raw_transitions <= 0:
        num_raw_transitions = total_samples

    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "dataset_type": SUPERVISED_DATASET_TYPE,
            "supervised_schema_version": 1,
            "horizon": int(input_horizon),
            "num_samples": int(total_samples),
            "num_raw_transitions": int(num_raw_transitions),
            "observation_shape": list(observation_shape),
            "observation_dtype": str(observation_dtype),
            "action_shape": list(action_shape),
            "action_dtype": str(action_dtype),
            "hdf5_input_files": [str(path) for path in paths],
        }
    )
    features = build_supervised_transition_features(
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        horizon=input_horizon,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def row_generator():
        output_sample_id = 0
        for path in paths:
            for sample in iter_hdf5_supervised_transition_rows(path):
                yield {
                    "sample_id": int(output_sample_id),
                    "source_trajectory_id": int(sample.get("source_trajectory_id", sample.get("env_index", 0))),
                    "step_index": int(sample.get("step_index", sample.get("source_step", output_sample_id))),
                    "env_index": int(sample["env_index"]),
                    "observation": np.asarray(sample["observation"]),
                    "actions": np.asarray(sample["actions"]),
                    "targets": np.asarray(sample["targets"]),
                    "dones": np.asarray(sample["dones"], dtype=np.bool_),
                    "terminateds": np.asarray(sample["terminateds"], dtype=np.bool_),
                    "truncateds": np.asarray(sample["truncateds"], dtype=np.bool_),
                }
                output_sample_id += 1

    dataset = ArrowDataset.from_generator(row_generator, features=features)
    dataset.save_to_disk(str(output_path))
    write_metadata(output_path, payload_metadata)


def load_supervised_transition_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_dir() or not (path / "state.json").exists():
        raise ValueError(f"Expected an Arrow supervised transition dataset: {path}")
    if read_metadata(path).get("dataset_type") != SUPERVISED_DATASET_TYPE:
        raise ValueError(f"Expected a supervised transition dataset: {path}")
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


def load_hdf5_supervised_transition_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_hdf5_supervised_transition_rows(path))


def iter_hdf5_supervised_transition_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    read_hdf5_metadata(path)

    with h5py.File(path, "r") as dataset_file:
        if "observation" not in dataset_file:
            return
        num_samples = int(dataset_file["observation"].shape[0])
        env_indices = dataset_file["env_index"] if "env_index" in dataset_file else None
        source_steps = dataset_file["source_step"] if "source_step" in dataset_file else None
        for sample_index in range(num_samples):
            env_index = int(env_indices[sample_index]) if env_indices is not None else 0
            source_step = int(source_steps[sample_index]) if source_steps is not None else sample_index
            yield {
                "sample_id": int(sample_index),
                "source_trajectory_id": int(env_index),
                "step_index": int(source_step),
                "env_index": int(env_index),
                "source_step": int(source_step),
                "observation": np.asarray(dataset_file["observation"][sample_index]),
                "actions": np.asarray(dataset_file["actions"][sample_index]),
                "targets": np.asarray(dataset_file["targets"][sample_index]),
                "dones": np.asarray(dataset_file["dones"][sample_index], dtype=np.bool_),
                "terminateds": np.asarray(dataset_file["terminateds"][sample_index], dtype=np.bool_),
                "truncateds": np.asarray(dataset_file["truncateds"][sample_index], dtype=np.bool_),
            }


class Hdf5RepresentationWriter:
    def __init__(
        self,
        path: str | Path,
        horizon: int,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = HDF5_REPRESENTATION_CHUNK_SIZE,
        target_chunk_bytes: int = HDF5_REPRESENTATION_TARGET_CHUNK_BYTES,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if target_chunk_bytes <= 0:
            raise ValueError("target_chunk_bytes must be positive.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.horizon = int(horizon)
        self.chunk_size = int(chunk_size)
        self.target_chunk_bytes = int(target_chunk_bytes)
        self.num_samples = 0
        self._closed = False
        self._initialized = False
        self.metadata: dict[str, Any] = hdf5_base_metadata(
            self.horizon,
            self.chunk_size,
            self.target_chunk_bytes,
        )
        if metadata:
            self.metadata.update(metadata)
        self.file = h5py.File(self.path, "w")
        self._write_metadata()

    def _write_metadata(self) -> None:
        write_hdf5_metadata(
            self.file,
            self.metadata,
            horizon=self.horizon,
            num_samples=self.num_samples,
            chunk_size=self.chunk_size,
            target_chunk_bytes=self.target_chunk_bytes,
        )

    def _create_dataset(self, name: str, values: np.ndarray) -> None:
        sample_shape = tuple(int(value) for value in values.shape[1:])
        self.file.create_dataset(
            name,
            shape=(0, *sample_shape),
            maxshape=(None, *sample_shape),
            chunks=hdf5_chunk_shape(values, self.chunk_size, self.target_chunk_bytes),
            dtype=values.dtype,
            **hdf5_compression_kwargs(),
        )

    def _initialize(self, arrays: dict[str, np.ndarray]) -> None:
        observations = arrays["observation"]
        actions = arrays["actions"]
        observation_shape = tuple(int(value) for value in observations.shape[1:])
        action_shape = tuple(int(value) for value in actions.shape[2:])
        self.metadata.update(
            {
                "observation_shape": list(observation_shape),
                "observation_dtype": str(observations.dtype),
                "action_shape": list(action_shape),
                "action_dtype": str(actions.dtype),
            }
        )
        for name in HDF5_SAMPLE_ARRAYS:
            self._create_dataset(name, arrays[name])
        self._initialized = True

    def _expected_shapes(self, arrays: dict[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
        num_samples = int(arrays["observation"].shape[0])
        return {
            "actions": (num_samples, self.horizon, *arrays["actions"].shape[2:]),
            "targets": (num_samples, self.horizon, *arrays["observation"].shape[1:]),
            **{name: (num_samples, self.horizon) for name in HDF5_FLAG_ARRAYS},
            "env_index": (num_samples,),
            "source_step": (num_samples,),
        }

    def _normalize_arrays(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        dones: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray,
        env_indices: np.ndarray,
        source_steps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        arrays = {
            "observation": np.asarray(observations),
            "actions": np.asarray(actions),
            "targets": np.asarray(targets),
            "dones": np.asarray(dones, dtype=np.bool_),
            "terminateds": np.asarray(terminateds, dtype=np.bool_),
            "truncateds": np.asarray(truncateds, dtype=np.bool_),
            "env_index": np.asarray(env_indices, dtype=np.int32),
            "source_step": np.asarray(source_steps, dtype=np.int64),
        }
        num_samples = int(arrays["observation"].shape[0])
        if num_samples <= 0:
            return arrays
        for name, expected_shape in self._expected_shapes(arrays).items():
            if arrays[name].shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {arrays[name].shape}.")
        return arrays

    def append(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        dones: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray,
        env_indices: np.ndarray,
        source_steps: np.ndarray,
    ) -> None:
        if self._closed:
            raise ValueError("Cannot append to a closed HDF5 representation writer.")
        arrays = self._normalize_arrays(
            observations,
            actions,
            targets,
            dones,
            terminateds,
            truncateds,
            env_indices,
            source_steps,
        )
        num_new = int(arrays["observation"].shape[0])
        if num_new <= 0:
            return
        if not self._initialized:
            self._initialize(arrays)

        start = int(self.num_samples)
        end = start + num_new
        for name in HDF5_SAMPLE_ARRAYS:
            values = arrays[name]
            dataset = self.file[name]
            dataset.resize((end, *dataset.shape[1:]))
            dataset[start:end] = values
        self.num_samples = end
        self._write_metadata()
        self.file.flush()

    def append_samples(self, samples: Sequence[dict[str, Any]]) -> None:
        if not samples:
            return
        self.append(
            observations=np.stack([np.asarray(sample["observation"]) for sample in samples], axis=0),
            actions=np.stack([np.asarray(sample["actions"]) for sample in samples], axis=0),
            targets=np.stack([np.asarray(sample["targets"]) for sample in samples], axis=0),
            dones=np.stack([np.asarray(sample["dones"], dtype=np.bool_) for sample in samples], axis=0),
            terminateds=np.stack([np.asarray(sample["terminateds"], dtype=np.bool_) for sample in samples], axis=0),
            truncateds=np.stack([np.asarray(sample["truncateds"], dtype=np.bool_) for sample in samples], axis=0),
            env_indices=np.asarray([int(sample["env_index"]) for sample in samples], dtype=np.int32),
            source_steps=np.asarray(
                [int(sample.get("source_step", sample.get("step_index", sample_index))) for sample_index, sample in enumerate(samples)],
                dtype=np.int64,
            ),
        )

    def close(self, metadata: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        if metadata:
            self.metadata.update(metadata)
        self._write_metadata()
        self.file.flush()
        self.metadata["actual_bytes"] = int(self.path.stat().st_size) if self.path.exists() else 0
        self._write_metadata()
        self.file.flush()
        self.file.close()
        self._closed = True


class TrajectoryRecorderCallback(BaseCallback):
    def __init__(
        self,
        save_path: str | Path,
        horizon: int = 1,
        sample_ratio: float = 0.005,
        sample_seed: int = 0,
        max_samples: int | None = None,
        verbose: int = 0,
        log_fn: Callable[[dict[str, float | int], int], None] | None = None,
        log_interval: int = 0,
    ):
        super().__init__(verbose=verbose)
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if not 0.0 <= sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in [0, 1].")
        if max_samples is not None and max_samples <= 0:
            max_samples = None
        self.save_path = Path(save_path)
        self.horizon = int(horizon)
        self.sample_ratio = float(sample_ratio)
        self.sample_seed = int(sample_seed)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.log_fn = log_fn
        self.log_interval = int(log_interval)
        self.writer = Hdf5RepresentationWriter(self.save_path, horizon=self.horizon)
        self.sample_rng = random.Random(self.sample_seed)
        self.eligible_samples = 0
        self.raw_transitions = 0
        self.next_source_step = 0
        self.tail_by_env: dict[int, dict[str, np.ndarray]] = {}
        self.rollout_dones: list[np.ndarray] = []
        self.rollout_terminateds: list[np.ndarray] = []
        self.rollout_truncateds: list[np.ndarray] = []
        self.rollout_terminal_next_observations: list[dict[int, np.ndarray]] = []

    @property
    def num_samples(self) -> int:
        return int(self.writer.num_samples)

    def _on_rollout_start(self) -> None:
        self.rollout_dones = []
        self.rollout_terminateds = []
        self.rollout_truncateds = []
        self.rollout_terminal_next_observations = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = np.asarray(self.locals["dones"], dtype=np.bool_)
        terminateds, truncateds = split_done_flags(dones, infos)
        terminal_next_observations: dict[int, np.ndarray] = {}
        for env_index, info in enumerate(infos or []):
            terminal_observation = info.get("terminal_observation")
            if terminal_observation is not None:
                terminal_next_observations[int(env_index)] = np.asarray(terminal_observation).copy()
        self.rollout_dones.append(dones.copy())
        self.rollout_terminateds.append(terminateds.copy())
        self.rollout_truncateds.append(truncateds.copy())
        self.rollout_terminal_next_observations.append(terminal_next_observations)
        return True

    def _rollout_actions(self, actions: np.ndarray) -> np.ndarray:
        if isinstance(self.model.action_space, gym.spaces.Discrete) and actions.ndim >= 3 and actions.shape[-1] == 1:
            return actions[..., 0].astype(np.int64, copy=False)
        return actions

    def _build_next_observations(self, observations: np.ndarray) -> np.ndarray:
        next_observations = np.empty_like(observations)
        if observations.shape[0] > 1:
            next_observations[:-1] = observations[1:]
        final_next_observation = np.asarray(self.locals.get("new_obs", self.model._last_obs))
        next_observations[-1] = final_next_observation
        for step_index, terminal_next_observations in enumerate(self.rollout_terminal_next_observations):
            for env_index, terminal_observation in terminal_next_observations.items():
                next_observations[step_index, env_index] = terminal_observation
        return next_observations

    def _update_tail(self, env_index: int, sequence: dict[str, np.ndarray]) -> None:
        update_transition_tail(self.tail_by_env, env_index, sequence, self.horizon)

    def _sequence_for_env(
        self,
        env_index: int,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray,
        source_steps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        sequence = {
            "observations": observations[:, env_index],
            "actions": actions[:, env_index],
            "next_observations": next_observations[:, env_index],
            "dones": dones[:, env_index],
            "terminateds": terminateds[:, env_index],
            "truncateds": truncateds[:, env_index],
            "source_steps": source_steps,
        }
        return sequence_with_optional_tail(self.tail_by_env, env_index, sequence)

    def _on_rollout_end(self) -> None:
        n_steps = len(self.rollout_dones)
        if n_steps <= 0:
            return
        rollout_buffer = self.model.rollout_buffer
        observations = np.asarray(rollout_buffer.observations[:n_steps])
        actions = self._rollout_actions(np.asarray(rollout_buffer.actions[:n_steps]))
        dones = np.stack(self.rollout_dones, axis=0)
        terminateds = np.stack(self.rollout_terminateds, axis=0)
        truncateds = np.stack(self.rollout_truncateds, axis=0)
        next_observations = self._build_next_observations(observations)
        source_steps = np.arange(self.next_source_step, self.next_source_step + n_steps, dtype=np.int64)
        self.raw_transitions += int(n_steps * observations.shape[1])

        kept_samples: list[dict[str, Any]] = []

        for env_index in range(int(observations.shape[1])):
            sequence = self._sequence_for_env(
                env_index=env_index,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                dones=dones,
                terminateds=terminateds,
                truncateds=truncateds,
                source_steps=source_steps,
            )
            sample_iter = iter_supervised_transition_samples_from_arrays(
                {
                    "observations": sequence["observations"][:, None],
                    "actions": sequence["actions"][:, None],
                    "next_observations": sequence["next_observations"][:, None],
                    "terminateds": sequence["terminateds"][:, None],
                    "truncateds": sequence["truncateds"][:, None],
                    "source_steps": sequence["source_steps"],
                    "env_indices": np.asarray([env_index], dtype=np.int64),
                },
                horizon=self.horizon,
            )
            for sample in sample_iter:
                self.eligible_samples += 1
                if self.max_samples is not None and self.num_samples + len(kept_samples) >= self.max_samples:
                    continue
                if self.sample_ratio < 1.0 and self.sample_rng.random() >= self.sample_ratio:
                    continue
                kept_samples.append(sample)
            self._update_tail(env_index, sequence)

        self.writer.append_samples(kept_samples)
        self.next_source_step += n_steps
        if self.log_fn is not None and self.log_interval > 0 and self.n_calls % self.log_interval == 0:
            self.log_fn(
                {
                    "num_timesteps": int(self.num_timesteps),
                    "representation_samples": int(self.num_samples),
                    "representation_eligible_windows": int(self.eligible_samples),
                    "representation_sample_ratio": float(self.sample_ratio),
                },
                int(self.num_timesteps),
            )

    def save(self, metadata: dict[str, Any] | None = None) -> None:
        payload = dict(metadata or {})
        payload.update(
            {
                "num_eligible_samples": int(self.eligible_samples),
                "num_raw_transitions": int(self.raw_transitions),
                "num_samples": int(self.num_samples),
                "sample_ratio": float(self.sample_ratio),
                "sample_seed": int(self.sample_seed),
                "max_samples": int(self.max_samples) if self.max_samples is not None else None,
                "horizon": int(self.horizon),
            }
        )
        self.writer.close(metadata=payload)


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


def collect_random_transition_arrays(
    env: VecEnv,
    num_steps: int,
    initial_observation: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    terminateds: list[np.ndarray] = []
    truncateds: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    observation = initial_observation
    for _ in range(num_steps):
        action = sample_action_batch(env.action_space, env.num_envs)
        next_observation, _, dones, infos = env.step(action)
        terminated, truncated = split_done_flags(dones, infos)
        observations.append(np.asarray(observation).copy())
        actions.append(np.asarray(action).copy())
        terminateds.append(terminated.copy())
        truncateds.append(truncated.copy())
        next_observations.append(resolve_terminal_next_observations(next_observation, infos))
        observation = next_observation
    arrays = {
        "observations": np.stack(observations, axis=0),
        "actions": np.stack(actions, axis=0),
        "terminateds": np.stack(terminateds, axis=0),
        "truncateds": np.stack(truncateds, axis=0),
        "next_observations": np.stack(next_observations, axis=0),
    }
    return arrays, observation


def update_transition_tail(
    tail_by_env: dict[int, dict[str, np.ndarray]],
    env_index: int,
    sequence: dict[str, np.ndarray],
    horizon: int,
) -> None:
    if horizon <= 1:
        tail_by_env.pop(env_index, None)
        return
    tail_length = min(int(horizon) - 1, int(sequence["observations"].shape[0]))
    tail_by_env[env_index] = {
        key: np.asarray(value[-tail_length:]).copy()
        for key, value in sequence.items()
    }


def sequence_with_optional_tail(
    tail_by_env: dict[int, dict[str, np.ndarray]],
    env_index: int,
    sequence: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    tail = tail_by_env.get(env_index)
    if tail is None:
        return sequence
    return {
        key: np.concatenate([tail[key], value], axis=0)
        for key, value in sequence.items()
    }


def collect_random_supervised_transition_samples_to_hdf5(
    env: VecEnv,
    output_path: str | Path,
    horizon: int,
    max_samples: int,
    rollout_steps: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, int]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")

    writer = Hdf5RepresentationWriter(output_path, horizon=horizon, metadata=metadata)
    observation = env.reset()
    tail_by_env: dict[int, dict[str, np.ndarray]] = {}
    source_step = 0
    raw_transitions = 0
    try:
        while writer.num_samples < int(max_samples):
            remaining = int(max_samples) - writer.num_samples
            remaining_steps = max(1, (remaining + env.num_envs - 1) // env.num_envs)
            chunk_steps = max(1, min(int(rollout_steps), remaining_steps + int(horizon) - 1))
            arrays, observation = collect_random_transition_arrays(env, chunk_steps, observation)
            source_steps = np.arange(source_step, source_step + chunk_steps, dtype=np.int64)
            raw_transitions += int(chunk_steps * env.num_envs)
            source_step += chunk_steps

            kept_samples: list[dict[str, Any]] = []
            for env_index in range(env.num_envs):
                sequence = {
                    "observations": arrays["observations"][:, env_index],
                    "actions": arrays["actions"][:, env_index],
                    "next_observations": arrays["next_observations"][:, env_index],
                    "terminateds": arrays["terminateds"][:, env_index],
                    "truncateds": arrays["truncateds"][:, env_index],
                    "source_steps": source_steps,
                }
                sequence = sequence_with_optional_tail(tail_by_env, env_index, sequence)
                remaining = int(max_samples) - writer.num_samples - len(kept_samples)
                if remaining > 0:
                    kept_samples.extend(
                        iter_supervised_transition_samples_from_arrays(
                            {
                                "observations": sequence["observations"][:, None],
                                "actions": sequence["actions"][:, None],
                                "next_observations": sequence["next_observations"][:, None],
                                "terminateds": sequence["terminateds"][:, None],
                                "truncateds": sequence["truncateds"][:, None],
                                "source_steps": sequence["source_steps"],
                                "env_indices": np.asarray([env_index], dtype=np.int64),
                            },
                            horizon=horizon,
                            max_start_transitions=remaining,
                        )
                    )
                update_transition_tail(tail_by_env, env_index, sequence, horizon)
            writer.append_samples(kept_samples)
    finally:
        writer.close(
            {
                "num_raw_transitions": int(raw_transitions),
                "target_random_samples": int(max_samples),
                "random_rollout_steps": int(rollout_steps),
            }
        )
    return int(writer.num_samples), int(raw_transitions)


class TransitionDataset(TorchDataset):
    def __init__(self, trajectory_files: Sequence[str | Path]):
        self.rows: list[dict[str, Any]] = []
        self.horizon: int | None = None
        for path in trajectory_files:
            path_rows = load_supervised_transition_rows(path)
            path_horizon = int(np.asarray(path_rows[0]["actions"]).shape[0])
            if self.horizon is None:
                self.horizon = path_horizon
            elif self.horizon != path_horizon:
                raise ValueError("All transition datasets must use the same horizon.")
            self.rows.extend(path_rows)
        if not self.rows:
            raise ValueError("No transitions were found.")
        self.horizon = int(self.horizon)

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
    if path.is_file() and h5py.is_hdf5(path):
        metadata = read_metadata(path)
        horizon = int(metadata.get("horizon", 1))
        with h5py.File(path, "r") as dataset_file:
            if "observation" not in dataset_file:
                return empty_supervised_count(horizon)
            return supervised_count(
                env_indices=np.asarray(dataset_file["env_index"], dtype=np.int64),
                dones=np.asarray(dataset_file["dones"], dtype=np.bool_),
                terminateds=np.asarray(dataset_file["terminateds"], dtype=np.bool_),
                truncateds=np.asarray(dataset_file["truncateds"], dtype=np.bool_),
                horizon=horizon,
            )

    if path.is_dir() and (path / "state.json").exists():
        metadata = read_metadata(path)
        if metadata.get("dataset_type") != SUPERVISED_DATASET_TYPE:
            raise ValueError(f"Expected a supervised transition dataset: {path}")
        dataset = ArrowDataset.load_from_disk(str(path))
        horizon = int(metadata.get("horizon", 0))
        if len(dataset) == 0:
            return empty_supervised_count(horizon)
        dones = np.asarray(dataset["dones"], dtype=np.bool_)
        if horizon <= 0:
            horizon = int(dones.shape[-1])
        return supervised_count(
            env_indices=np.asarray(dataset["env_index"], dtype=np.int64),
            dones=dones,
            terminateds=np.asarray(dataset["terminateds"], dtype=np.bool_),
            truncateds=np.asarray(dataset["truncateds"], dtype=np.bool_),
            horizon=horizon,
        )

    raise ValueError(f"Unsupported dataset path: {path}")
