import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.world_model import WorldModel
from src.utils.io import ensure_dir, load_checkpoint
from src.utils.video import write_video


def to_tensor_image(frame_hwc: np.ndarray) -> torch.Tensor:
    x = frame_hwc.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def denorm_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (actions - mean[None, :]) / std[None, :]


def concat_gt_pred(gt: np.ndarray, pred: np.ndarray, step_text: str) -> np.ndarray:
    gap = np.full((gt.shape[0], 8, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([gt, gap, pred], axis=1)
    cv2.putText(canvas, step_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (8, gt.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Pred",
        (gt.shape[1] + 16, gt.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    latent_dim = cfg["model"]["latent_dim"]
    action_dim = int(ckpt["action_dim"])
    hidden_dim = cfg["model"]["hidden_dim"]

    model = WorldModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    action_mean = ckpt["action_mean"].astype(np.float32)
    action_std = ckpt["action_std"].astype(np.float32)

    episode_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not episode_files:
        raise FileNotFoundError(f"No episode files in {args.data_dir}")
    ep_path = episode_files[min(args.episode_idx, len(episode_files) - 1)]
    data = np.load(ep_path)
    frames = data["frames"]
    actions = data["actions"].astype(np.float32)

    horizon = min(args.horizon, actions.shape[0])
    x0 = to_tensor_image(frames[0])
    a_seq = denorm_actions(actions[:horizon], action_mean, action_std)
    a_seq = torch.from_numpy(a_seq).unsqueeze(0)  # (1, H, A)

    with torch.no_grad():
        preds = model.rollout(x0, a_seq)  # (1,H,3,64,64)
    preds = preds.squeeze(0).permute(0, 2, 3, 1).numpy()
    preds = (preds * 255.0).clip(0, 255).astype(np.uint8)

    frames_out = []
    for t in range(horizon):
        gt = frames[t + 1][..., :3]
        pred = preds[t]
        frames_out.append(concat_gt_pred(gt, pred, step_text=f"t+{t+1}"))

    ensure_dir("outputs/rollout_videos")
    output_path = args.output or os.path.join("outputs/rollout_videos", f"rollout_ep{args.episode_idx:03d}.mp4")
    write_video(frames_out, output_path, fps=8)
    print(f"Saved rollout video: {output_path}")


if __name__ == "__main__":
    main()
