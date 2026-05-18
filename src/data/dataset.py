import glob
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ActionStats:
    mean: np.ndarray
    std: np.ndarray


def _to_chw_float01(image_hwc: np.ndarray) -> np.ndarray:
    x = image_hwc.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))


def _collect_episode_paths(data_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz episode files found in: {data_dir}")
    return paths


def build_indices(
    data_dir: str, val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], ActionStats]:
    rng = np.random.default_rng(seed)
    episode_paths = _collect_episode_paths(data_dir)
    rng.shuffle(episode_paths)

    split = int(len(episode_paths) * (1.0 - val_ratio))
    train_eps = episode_paths[:split]
    val_eps = episode_paths[split:] if split < len(episode_paths) else episode_paths[-1:]

    train_idx = []
    val_idx = []
    all_train_actions = []

    for path in train_eps:
        data = np.load(path)
        actions = data["actions"].astype(np.float32)
        for t in range(actions.shape[0]):
            train_idx.append((path, t))
        all_train_actions.append(actions)

    for path in val_eps:
        data = np.load(path)
        actions = data["actions"].astype(np.float32)
        for t in range(actions.shape[0]):
            val_idx.append((path, t))

    cat_actions = np.concatenate(all_train_actions, axis=0)
    mean = cat_actions.mean(axis=0)
    std = cat_actions.std(axis=0) + 1e-6
    stats = ActionStats(mean=mean, std=std)
    return train_idx, val_idx, stats


def save_indices(
    train_idx: List[Tuple[str, int]],
    val_idx: List[Tuple[str, int]],
    stats: ActionStats,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_index.pkl"), "wb") as f:
        pickle.dump(train_idx, f)
    with open(os.path.join(out_dir, "val_index.pkl"), "wb") as f:
        pickle.dump(val_idx, f)
    np.savez(
        os.path.join(out_dir, "action_stats.npz"),
        mean=stats.mean.astype(np.float32),
        std=stats.std.astype(np.float32),
    )


def load_indices(index_dir: str):
    with open(os.path.join(index_dir, "train_index.pkl"), "rb") as f:
        train_idx = pickle.load(f)
    with open(os.path.join(index_dir, "val_index.pkl"), "rb") as f:
        val_idx = pickle.load(f)
    stats = np.load(os.path.join(index_dir, "action_stats.npz"))
    action_stats = ActionStats(mean=stats["mean"], std=stats["std"])
    return train_idx, val_idx, action_stats


class TransitionDataset(Dataset):
    def __init__(
        self,
        index_list: List[Tuple[str, int]],
        action_stats: ActionStats,
    ) -> None:
        self.index_list = index_list
        self.action_mean = action_stats.mean.astype(np.float32)
        self.action_std = action_stats.std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, t = self.index_list[idx]
        data = np.load(path)
        frames = data["frames"]  # (T+1,H,W,3)
        actions = data["actions"]  # (T,A)
        rewards = data["rewards"]
        dones = data["dones"]

        x_t = _to_chw_float01(frames[t])
        x_tp1 = _to_chw_float01(frames[t + 1])
        a_t = actions[t].astype(np.float32)
        a_t = (a_t - self.action_mean) / self.action_std
        r_t = np.array([rewards[t]], dtype=np.float32)
        d_t = np.array([dones[t]], dtype=np.float32)

        return {
            "x_t": torch.from_numpy(x_t),
            "a_t": torch.from_numpy(a_t),
            "x_tp1": torch.from_numpy(x_tp1),
            "r_t": torch.from_numpy(r_t),
            "d_t": torch.from_numpy(d_t),
        }
