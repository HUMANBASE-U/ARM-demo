import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.losses.losses import compute_losses
from src.models.world_model import WorldModel


def to_rgb_u8(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame[..., :3]


def as_np3(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    return x[:3].astype(np.float32)


def preprocess_frame(frame: np.ndarray, image_size: int = 64) -> np.ndarray:
    img = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def build_corners(z: float) -> Dict[str, np.ndarray]:
    return {
        "top_left": np.array([-0.18, +0.18, z], dtype=np.float32),
        "top_right": np.array([+0.18, +0.18, z], dtype=np.float32),
        "bottom_left": np.array([-0.18, -0.18, z], dtype=np.float32),
        "bottom_right": np.array([+0.18, -0.18, z], dtype=np.float32),
    }


def forbidden_hit(ee_xyz: np.ndarray, center_xy: np.ndarray, radius: float, z_max: float) -> bool:
    return bool(np.linalg.norm(ee_xyz[:2] - center_xy) <= radius and ee_xyz[2] <= z_max)


def collect_dataset(
    env,
    episodes: int,
    max_steps: int,
    horizon_steps: int,
    image_size: int,
    forbidden_center_xy: np.ndarray,
    forbidden_radius: float,
    forbidden_z_max: float,
) -> Tuple[TensorDataset, Dict]:
    x_t_list, a_t_list, x_tp1_list = [], [], []
    risk_list, r_t_list, d_t_list = [], [], []
    total_hits = 0
    for ep in range(episodes):
        env.reset(seed=ep)
        cube0 = as_np3(env.unwrapped.cube.pose.p)
        corners = list(build_corners(cube0[2]).values())
        target = corners[ep % 4]

        frames: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        ee_traj: List[np.ndarray] = []
        done = False
        for _ in range(max_steps):
            ee = as_np3(env.unwrapped.agent.tcp.pose.p)
            cube = as_np3(env.unwrapped.cube.pose.p)
            # Candidate controller: move toward cube then toward corner.
            toward = (cube - ee) if np.linalg.norm(cube[:2] - ee[:2]) > 0.06 else (target - ee)
            dxyz = np.clip(toward / 0.04, -1.0, 1.0)
            action = np.array([dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            frame = to_rgb_u8(env.render())
            frames.append(preprocess_frame(frame, image_size=image_size))
            ee_traj.append(ee.copy())
            actions.append(action.copy())
            _, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            if done:
                break

        if len(frames) < 2:
            continue

        for t in range(len(frames) - 1):
            y = 0.0
            for h in range(1, horizon_steps + 1):
                idx = min(t + h, len(ee_traj) - 1)
                if forbidden_hit(ee_traj[idx], forbidden_center_xy, forbidden_radius, forbidden_z_max):
                    y = 1.0
                    break
            total_hits += int(y > 0.5)
            x_t_list.append(frames[t])
            x_tp1_list.append(frames[t + 1])
            a_t_list.append(actions[t])
            risk_list.append([y])
            r_t_list.append([0.0])
            d_t_list.append([1.0 if done and (t == len(frames) - 2) else 0.0])

    x_t = torch.tensor(np.asarray(x_t_list), dtype=torch.float32)
    x_tp1 = torch.tensor(np.asarray(x_tp1_list), dtype=torch.float32)
    a_t = torch.tensor(np.asarray(a_t_list), dtype=torch.float32)
    y_risk = torch.tensor(np.asarray(risk_list), dtype=torch.float32)
    r_t = torch.tensor(np.asarray(r_t_list), dtype=torch.float32)
    d_t = torch.tensor(np.asarray(d_t_list), dtype=torch.float32)
    ds = TensorDataset(x_t, a_t, x_tp1, y_risk, r_t, d_t)
    meta = {
        "num_samples": int(len(x_t_list)),
        "num_risky_labels": int(total_hits),
        "action_dim": int(a_t.shape[-1]),
        "image_size": int(image_size),
        "horizon_steps": int(horizon_steps),
    }
    return ds, meta


@dataclass
class TrainStats:
    steps: List[int]
    wm_total: List[float]
    wm_pred: List[float]
    wm_latent: List[float]
    risk_bce: List[float]
    total: List[float]


class RiskHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def save_curve(stats: TrainStats, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = np.array(stats.steps, dtype=np.int32)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(x, stats.wm_total)
    ax[0, 0].set_title("WM Total")
    ax[0, 0].set_xlabel("Training step")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 1].plot(x, stats.wm_pred, label="pred")
    ax[0, 1].plot(x, stats.wm_latent, label="latent")
    ax[0, 1].legend()
    ax[0, 1].set_title("WM Components")
    ax[0, 1].set_xlabel("Training step")
    ax[0, 1].set_ylabel("Loss")
    ax[1, 0].plot(x, stats.risk_bce)
    ax[1, 0].set_title("Risk BCE")
    ax[1, 0].set_xlabel("Training step")
    ax[1, 0].set_ylabel("Loss")
    ax[1, 1].plot(x, stats.total)
    ax[1, 1].set_title("Safety Total Loss")
    ax[1, 1].set_xlabel("Training step")
    ax[1, 1].set_ylabel("Loss")
    for a in ax.reshape(-1):
        a.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--max_steps", type=int, default=90)
    p.add_argument("--horizon_steps", type=int, default=30)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--alpha_risk", type=float, default=1.0)
    p.add_argument("--forbidden_x", type=float, default=0.0)
    p.add_argument("--forbidden_y", type=float, default=0.0)
    p.add_argument("--forbidden_radius", type=float, default=0.08)
    p.add_argument("--forbidden_z_max", type=float, default=0.18)
    p.add_argument("--output_dir", type=str, default="outputs/safety_wm_v1")
    p.add_argument("--checkpoint", type=str, default="checkpoints/safety_wm_v1.pt")
    args = p.parse_args()

    import mani_skill.envs  # noqa: F401
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("PickCube-v1", obs_mode="rgb", control_mode="pd_ee_delta_pose", render_mode="rgb_array")

    center_xy = np.array([args.forbidden_x, args.forbidden_y], dtype=np.float32)
    ds, meta = collect_dataset(
        env=env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        horizon_steps=args.horizon_steps,
        image_size=args.image_size,
        forbidden_center_xy=center_xy,
        forbidden_radius=args.forbidden_radius,
        forbidden_z_max=args.forbidden_z_max,
    )
    env.close()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    wm = WorldModel(latent_dim=args.latent_dim, action_dim=meta["action_dim"], hidden_dim=256).to(device)
    risk_head = RiskHead(args.latent_dim).to(device)
    opt = torch.optim.Adam(list(wm.parameters()) + list(risk_head.parameters()), lr=3e-4)
    bce = nn.BCEWithLogitsLoss()

    stats = TrainStats([], [], [], [], [], [])
    step_i = 0
    for ep in range(args.epochs):
        wm.train()
        risk_head.train()
        sum_wm_t, sum_pred, sum_lat, sum_rb, sum_all = 0.0, 0.0, 0.0, 0.0, 0.0
        cnt = 0
        for x_t, a_t, x_tp1, y_risk, r_t, d_t in loader:
            x_t = x_t.to(device)
            a_t = a_t.to(device)
            x_tp1 = x_tp1.to(device)
            y_risk = y_risk.to(device)
            r_t = r_t.to(device)
            d_t = d_t.to(device)

            out = wm(x_t, a_t, x_tp1)
            losses = compute_losses(
                out,
                x_t=x_t,
                x_tp1=x_tp1,
                r_t=r_t,
                d_t=d_t,
                lambda_recon=1.0,
                lambda_pred=1.0,
                lambda_latent=0.5,
                lambda_reward=0.0,
                lambda_done=0.0,
            )
            risk_logit = risk_head(out["z_tp1_pred"])
            l_risk = bce(risk_logit, y_risk)
            total = losses["total"] + args.alpha_risk * l_risk

            opt.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(list(wm.parameters()) + list(risk_head.parameters()), 1.0)
            opt.step()

            sum_wm_t += float(losses["total"].item())
            sum_pred += float(losses["pred"].item())
            sum_lat += float(losses["latent"].item())
            sum_rb += float(l_risk.item())
            sum_all += float(total.item())
            cnt += 1
            step_i += 1

        stats.steps.append(step_i)
        stats.wm_total.append(sum_wm_t / max(cnt, 1))
        stats.wm_pred.append(sum_pred / max(cnt, 1))
        stats.wm_latent.append(sum_lat / max(cnt, 1))
        stats.risk_bce.append(sum_rb / max(cnt, 1))
        stats.total.append(sum_all / max(cnt, 1))
        print(
            f"[epoch {ep+1}/{args.epochs}] step={step_i} "
            f"wm={stats.wm_total[-1]:.4f} risk={stats.risk_bce[-1]:.4f} total={stats.total[-1]:.4f}"
        )

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save(
        {
            "wm_state_dict": wm.state_dict(),
            "risk_state_dict": risk_head.state_dict(),
            "latent_dim": args.latent_dim,
            "action_dim": meta["action_dim"],
            "image_size": args.image_size,
            "horizon_steps": args.horizon_steps,
            "forbidden_center_xy": center_xy.tolist(),
            "forbidden_radius": args.forbidden_radius,
            "forbidden_z_max": args.forbidden_z_max,
        },
        args.checkpoint,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    curve = os.path.join(args.output_dir, "loss_curves.png")
    save_curve(stats, curve)
    summary = {
        "checkpoint": args.checkpoint,
        "loss_curve": curve,
        "dataset_meta": meta,
        "final_losses": {
            "wm_total": stats.wm_total[-1],
            "wm_pred": stats.wm_pred[-1],
            "wm_latent": stats.wm_latent[-1],
            "risk_bce": stats.risk_bce[-1],
            "safety_total": stats.total[-1],
        },
    }
    with open(os.path.join(args.output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved: {os.path.join(args.output_dir, 'result.json')}")


if __name__ == "__main__":
    main()
