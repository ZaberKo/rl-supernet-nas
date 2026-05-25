from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

try:
    import ale_py  # noqa: F401
except ImportError:
    pass
from gymnasium import spaces
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.preprocessing import is_image_space_channels_first
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecMonitor, VecNormalize, VecTransposeImage


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        self.skip = int(skip)
        if self.skip <= 0:
            raise ValueError("Frame skip must be positive.")

    def step(self, action: Any):
        total_reward = 0.0
        observation = None
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info


class ResizeImageObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, image_size: int):
        super().__init__(env)
        if len(env.observation_space.shape) != 3:
            raise ValueError("ResizeImageObservation expects image observations.")
        channels = int(env.observation_space.shape[-1])
        self.image_size = int(image_size)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size, self.image_size, channels),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.uint8)
        try:
            import cv2

            resized = cv2.resize(
                observation,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            from PIL import Image

            resized = np.asarray(
                Image.fromarray(observation).resize(
                    (self.image_size, self.image_size),
                    resample=Image.BILINEAR,
                )
            )
        if resized.ndim == 2:
            resized = resized[..., None]
        return resized.astype(np.uint8, copy=False)


class GrayscaleImageObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if len(env.observation_space.shape) != 3:
            raise ValueError("GrayscaleImageObservation expects image observations.")
        height, width = env.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.uint8)
        if observation.ndim != 3:
            raise ValueError("GrayscaleImageObservation expects HWC observations.")
        if observation.shape[-1] == 1:
            return observation
        red = observation[..., 0].astype(np.float32)
        green = observation[..., 1].astype(np.float32)
        blue = observation[..., 2].astype(np.float32)
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        return np.rint(gray).clip(0, 255).astype(np.uint8)[..., None]


def is_atari_env(env_id: str) -> bool:
    return env_id.startswith("ALE/") or env_id.endswith("NoFrameskip-v4")


def apply_time_limit(env: gym.Env, max_episode_steps: int | None) -> gym.Env:
    if max_episode_steps is None:
        return env
    max_steps = int(max_episode_steps)
    if max_steps <= 0:
        return env
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)


def remove_outer_time_limit(env: gym.Env) -> gym.Env:
    if isinstance(env, gym.wrappers.TimeLimit):
        return env.env
    return env


def make_single_atari_env(
    env_id: str,
    seed: int,
    image_size: int = 84,
    max_episode_steps: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    noop_max: int = 30,
    frame_skip: int = 4,
    terminal_on_life_loss: bool = True,
    clip_reward: bool = True,
    action_repeat_probability: float = 0.0,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        if not is_atari_env(env_id):
            raise ValueError("The Atari wrapper requires an Atari environment.")
        kwargs: dict[str, Any] = dict(env_kwargs or {})
        kwargs["frameskip"] = 1
        kwargs["repeat_action_probability"] = 0.0
        env = gym.make(env_id, **kwargs)
        env = remove_outer_time_limit(env)
        screen_size = int(image_size)
        if screen_size <= 0:
            raise ValueError("image_size must be positive for Atari environments.")
        env = AtariWrapper(
            env,
            noop_max=int(noop_max),
            frame_skip=int(frame_skip),
            screen_size=screen_size,
            terminal_on_life_loss=bool(terminal_on_life_loss),
            clip_reward=bool(clip_reward),
            action_repeat_probability=float(action_repeat_probability),
        )
        env = apply_time_limit(env, max_episode_steps)
        env.reset(seed=seed)
        return env

    return _init


def make_single_box2d_env(
    env_id: str,
    seed: int,
    image_size: int = 64,
    max_episode_steps: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    frame_skip: int = 1,
    grayscale_observation: bool = False,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        kwargs: dict[str, Any] = dict(env_kwargs or {})
        env = gym.make(env_id, **kwargs)
        env = remove_outer_time_limit(env)
        if frame_skip > 1:
            env = FrameSkipWrapper(env, skip=frame_skip)
        if grayscale_observation:
            env = GrayscaleImageObservation(env)
        if image_size > 0 and env.observation_space.shape[:2] != (image_size, image_size):
            env = ResizeImageObservation(env, image_size=image_size)
        env = apply_time_limit(env, max_episode_steps)
        env.reset(seed=seed)
        return env

    return _init


def make_vec_env_from_factories(
    env_fns: list[Callable[[], gym.Env]],
    vector_env_type: str = "dummy",
    frame_stack: int = 1,
    normalize_observation: bool = False,
    normalize_reward: bool = False,
    normalize_clip_obs: float = 10.0,
    normalize_gamma: float = 0.99,
) -> VecEnv:
    n_envs = len(env_fns)
    if n_envs <= 0:
        raise ValueError("At least one environment factory is required.")
    if frame_stack <= 0:
        raise ValueError("frame_stack must be positive.")
    if vector_env_type == "dummy" or n_envs <= 1:
        env = DummyVecEnv(env_fns)
    elif vector_env_type == "subproc":
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        raise ValueError(f"Unsupported vector_env_type: {vector_env_type}")
    env = VecMonitor(env)
    observation_space = env.observation_space
    if (
        len(observation_space.shape) == 3
        and not is_image_space_channels_first(observation_space)
    ):
        env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
    if normalize_observation or normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=normalize_observation,
            norm_reward=normalize_reward,
            clip_obs=float(normalize_clip_obs),
            gamma=float(normalize_gamma),
        )
    return env


def make_atari_vec_env(
    env_id: str,
    n_envs: int,
    seed: int,
    image_size: int = 84,
    vector_env_type: str = "dummy",
    frame_stack: int = 1,
    max_episode_steps: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    noop_max: int = 30,
    frame_skip: int = 4,
    terminal_on_life_loss: bool = True,
    clip_reward: bool = True,
    action_repeat_probability: float = 0.0,
    normalize_observation: bool = False,
    normalize_reward: bool = False,
    normalize_clip_obs: float = 10.0,
    normalize_gamma: float = 0.99,
) -> VecEnv:
    env_fns = [
        make_single_atari_env(
            env_id=env_id,
            seed=seed + rank,
            image_size=image_size,
            max_episode_steps=max_episode_steps,
            env_kwargs=env_kwargs,
            noop_max=noop_max,
            frame_skip=frame_skip,
            terminal_on_life_loss=terminal_on_life_loss,
            clip_reward=clip_reward,
            action_repeat_probability=action_repeat_probability,
        )
        for rank in range(n_envs)
    ]
    return make_vec_env_from_factories(
        env_fns=env_fns,
        vector_env_type=vector_env_type,
        frame_stack=frame_stack,
        normalize_observation=normalize_observation,
        normalize_reward=normalize_reward,
        normalize_clip_obs=normalize_clip_obs,
        normalize_gamma=normalize_gamma,
    )


def make_box2d_vec_env(
    env_id: str,
    n_envs: int,
    seed: int,
    image_size: int = 64,
    vector_env_type: str = "dummy",
    frame_stack: int = 1,
    max_episode_steps: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    frame_skip: int = 1,
    grayscale_observation: bool = False,
    normalize_observation: bool = False,
    normalize_reward: bool = False,
    normalize_clip_obs: float = 10.0,
    normalize_gamma: float = 0.99,
) -> VecEnv:
    env_fns = [
        make_single_box2d_env(
            env_id=env_id,
            seed=seed + rank,
            image_size=image_size,
            max_episode_steps=max_episode_steps,
            env_kwargs=env_kwargs,
            frame_skip=frame_skip,
            grayscale_observation=grayscale_observation,
        )
        for rank in range(n_envs)
    ]
    return make_vec_env_from_factories(
        env_fns=env_fns,
        vector_env_type=vector_env_type,
        frame_stack=frame_stack,
        normalize_observation=normalize_observation,
        normalize_reward=normalize_reward,
        normalize_clip_obs=normalize_clip_obs,
        normalize_gamma=normalize_gamma,
    )
