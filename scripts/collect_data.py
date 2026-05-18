import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.make_env import make_maniskill_env
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


def _resize_rgb(frame: np.ndarray, image_size: int) -> np.ndarray:
    if frame.shape[0] == image_size and frame.shape[1] == image_size:
        return frame
    return cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _extract_frame(obs, env, image_size: int) -> np.ndarray:
    def as_numpy_image(arr) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr[..., :3]

    if isinstance(obs, dict):
        if "rgb" in obs:
            rgb = obs["rgb"]
            if isinstance(rgb, dict):
                # Pick one camera stream if multiple exist.
                rgb = next(iter(rgb.values()))
            rgb = as_numpy_image(rgb)
            return _resize_rgb(rgb[..., :3], image_size)
    frame = env.render()
    frame = as_numpy_image(frame)
    return _resize_rgb(frame[..., :3], image_size)


def collect_dummy_data(output_dir: str, num_episodes: int, max_steps: int, image_size: int, action_dim: int = 7):
    ensure_dir(output_dir)
    rng = np.random.default_rng(42)
    for ep in tqdm(range(num_episodes), desc="Collecting dummy episodes"):
        frames = []
        actions = []
        rewards = []
        dones = []

        pos = rng.uniform(8, image_size - 8, size=2)
        vel = rng.uniform(-2.0, 2.0, size=2)
        frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)
        frames.append(frame)

        for t in range(max_steps):
            action = rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            vel += 0.3 * action[:2]
            vel = np.clip(vel, -3.0, 3.0)
            pos += vel
            pos = np.clip(pos, 5, image_size - 5)

            frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)
            frames.append(frame)
            actions.append(action)
            rewards.append(float(-np.linalg.norm(pos - image_size / 2)))
            dones.append(float(t == max_steps - 1))

        np.savez_compressed(
            os.path.join(output_dir, f"episode_{ep:05d}.npz"),
            frames=np.stack(frames, axis=0),
            actions=np.stack(actions, axis=0).astype(np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )


def collect_maniskill_data(output_dir: str, num_episodes: int, max_steps: int, image_size: int):
    ensure_dir(output_dir)
    env = make_maniskill_env()
    for ep in tqdm(range(num_episodes), desc="Collecting ManiSkill episodes"):
        obs, _ = env.reset(seed=ep)
        frame = _extract_frame(obs, env, image_size)

        frames = [frame]
        actions = []
        rewards = []
        dones = []

        for _ in range(max_steps):
            action = env.action_space.sample().astype(np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_frame = _extract_frame(next_obs, env, image_size)

            frames.append(next_frame)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(float(done))

            if done:
                break

        np.savez_compressed(
            os.path.join(output_dir, f"episode_{ep:05d}.npz"),
            frames=np.stack(frames, axis=0),
            actions=np.stack(actions, axis=0).astype(np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_dummy_data", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.use_dummy_data:
        collect_dummy_data(args.output_dir, args.num_episodes, args.max_steps, args.image_size)
    else:
        collect_maniskill_data(args.output_dir, args.num_episodes, args.max_steps, args.image_size)


if __name__ == "__main__":
    main()
