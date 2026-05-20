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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecMonitor, VecTransposeImage


class RenderObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        env.reset()
        frame = env.render()
        if frame is None:
            raise ValueError("The environment did not return an rgb_array frame.")
        frame = np.asarray(frame, dtype=np.uint8)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=frame.shape,
            dtype=np.uint8,
        )

    def reset(self, **kwargs: Any):
        _, info = self.env.reset(**kwargs)
        return self._get_frame(), info

    def step(self, action: Any):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_frame(), reward, terminated, truncated, info

    def _get_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise ValueError("The environment did not return an rgb_array frame.")
        return np.asarray(frame, dtype=np.uint8)


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


def is_atari_env(env_id: str) -> bool:
    return env_id.startswith("ALE/") or env_id.endswith("NoFrameskip-v4")


def make_single_vision_env(
    env_id: str,
    seed: int,
    image_size: int = 64,
    use_render_observation: bool = True,
    atari_wrapper: str = "none",
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        wrapper_name = atari_wrapper.lower()
        kwargs: dict[str, Any] = {}
        if wrapper_name == "sb3":
            if not is_atari_env(env_id):
                raise ValueError("The sb3 Atari wrapper requires an Atari environment.")
            kwargs["frameskip"] = 1
            kwargs["repeat_action_probability"] = 0.0
        elif use_render_observation:
            kwargs["render_mode"] = "rgb_array"
        env = gym.make(env_id, **kwargs)
        if wrapper_name == "sb3":
            env = AtariWrapper(env)
        elif wrapper_name != "none":
            raise ValueError(f"Unsupported atari_wrapper: {atari_wrapper}")
        elif use_render_observation:
            env = RenderObservationWrapper(env)
        if image_size > 0 and env.observation_space.shape[:2] != (image_size, image_size):
            env = ResizeImageObservation(env, image_size=image_size)
        env.reset(seed=seed)
        return env

    return _init


def make_vision_vec_env(
    env_id: str,
    n_envs: int,
    seed: int,
    image_size: int = 64,
    use_render_observation: bool = True,
    vector_env_type: str = "dummy",
    frame_stack: int = 1,
    atari_wrapper: str = "none",
) -> VecEnv:
    env_fns = [
        make_single_vision_env(
            env_id=env_id,
            seed=seed + rank,
            image_size=image_size,
            use_render_observation=use_render_observation,
            atari_wrapper=atari_wrapper,
        )
        for rank in range(n_envs)
    ]
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
    return env


def get_vision_spaces(
    env_id: str,
    seed: int,
    image_size: int = 64,
    use_render_observation: bool = True,
    vector_env_type: str = "dummy",
    frame_stack: int = 1,
    atari_wrapper: str = "none",
) -> tuple[spaces.Space, spaces.Space]:
    env = make_vision_vec_env(
        env_id=env_id,
        n_envs=1,
        seed=seed,
        image_size=image_size,
        use_render_observation=use_render_observation,
        vector_env_type=vector_env_type,
        frame_stack=frame_stack,
        atari_wrapper=atari_wrapper,
    )
    try:
        return env.observation_space, env.action_space
    finally:
        env.close()
