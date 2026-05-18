from typing import Tuple

import gymnasium as gym


def make_maniskill_env(env_id: str = "PickCube-v1", obs_mode: str = "rgb", render_mode: str = "rgb_array"):
    try:
        import mani_skill.envs  # noqa: F401
    except ImportError as exc:
        raise ImportError("maniskill is required for real environment data collection") from exc

    env = gym.make(env_id, obs_mode=obs_mode, render_mode=render_mode)
    return env


def infer_action_dim(env) -> int:
    shape: Tuple[int, ...] = env.action_space.shape
    return int(shape[0])
