import argparse
import json
import os
import sys
from typing import Tuple

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.world_model import WorldModel


def to_rgb_u8(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame[..., :3]


def preprocess_frame(frame: np.ndarray, image_size: int = 64) -> np.ndarray:
    img = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def as_np3(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    return x[:3].astype(np.float32)


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


def build_corners(z: float):
    return {
        "top_left": np.array([-0.18, +0.18, z], dtype=np.float32),
        "top_right": np.array([+0.18, +0.18, z], dtype=np.float32),
        "bottom_left": np.array([-0.18, -0.18, z], dtype=np.float32),
        "bottom_right": np.array([+0.18, -0.18, z], dtype=np.float32),
    }


def estimate_risk(
    wm: WorldModel,
    risk_head: nn.Module,
    x_t: torch.Tensor,
    a_t: torch.Tensor,
    horizon_steps: int,
    n_samples: int = 5,
    z_noise_std: float = 0.02,
) -> float:
    wm.eval()
    risk_head.eval()
    with torch.no_grad():
        z0 = wm.encoder(x_t)
        risks = []
        for _ in range(n_samples):
            z = z0 + z_noise_std * torch.randn_like(z0)
            rmax = 0.0
            for _h in range(horizon_steps):
                z, _, _ = wm.dynamics(z, a_t)
                r = torch.sigmoid(risk_head(z)).item()
                rmax = max(rmax, r)
            risks.append(rmax)
    return float(np.mean(risks))


def draw_overlay(frame: np.ndarray, cube_xy: np.ndarray, ee_xy: np.ndarray, center_xy: np.ndarray, radius: float, risk: float, stopped: bool):
    out = frame.copy()
    h, w = out.shape[:2]
    x0, y0 = w - 190, 10
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + 180, y0 + 180), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.40, out, 0.60, 0, out)
    cv2.rectangle(out, (x0, y0), (x0 + 180, y0 + 180), (220, 220, 220), 1)

    def map_xy(p):
        px = int(x0 + (p[0] + 0.25) / 0.50 * 180)
        py = int(y0 + (0.25 - (p[1] + 0.25)) / 0.50 * 180)
        return px, py

    cx, cy = map_xy(center_xy)
    rr = int(max(4, radius / 0.50 * 180))
    zone_layer = out.copy()
    cv2.circle(zone_layer, (cx, cy), rr, (0, 0, 255), -1)
    cv2.addWeighted(zone_layer, 0.25, out, 0.75, 0, out)
    cv2.circle(out, (cx, cy), rr, (0, 0, 255), 2)

    eex, eey = map_xy(ee_xy)
    cbx, cby = map_xy(cube_xy)
    cv2.circle(out, (eex, eey), 4, (255, 255, 0), -1)
    cv2.circle(out, (cbx, cby), 4, (0, 255, 0), -1)
    cv2.putText(out, f"Risk={risk:.3f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if stopped:
        cv2.putText(out, "RISK ALERT: future collision predicted. STOP.", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/safety_wm_v1.pt")
    p.add_argument("--output_dir", type=str, default="outputs/safety_demo_v1")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--risk_threshold", type=float, default=0.52)
    p.add_argument("--horizon_steps", type=int, default=30)  # 3 seconds if control is ~10Hz
    p.add_argument("--n_samples", type=int, default=5)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = WorldModel(latent_dim=int(ckpt["latent_dim"]), action_dim=int(ckpt["action_dim"]), hidden_dim=256).to(device)
    wm.load_state_dict(ckpt["wm_state_dict"], strict=False)
    risk_head = RiskHead(int(ckpt["latent_dim"])).to(device)
    risk_head.load_state_dict(ckpt["risk_state_dict"], strict=False)

    center_xy = np.array(ckpt["forbidden_center_xy"], dtype=np.float32)
    radius = float(ckpt["forbidden_radius"])
    image_size = int(ckpt.get("image_size", 64))
    horizon_steps = int(args.horizon_steps if args.horizon_steps > 0 else ckpt.get("horizon_steps", 30))

    import mani_skill.envs  # noqa: F401
    env = gym.make("PickCube-v1", obs_mode="rgb", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
    results = []

    for ep in range(args.episodes):
        env.reset(seed=2026 + ep)
        cube = as_np3(env.unwrapped.cube.pose.p)
        corners = list(build_corners(cube[2]).values())
        target = corners[ep % 4]
        frames = []
        stopped = False
        max_r_seen = 0.0
        for t in range(140):
            frame = to_rgb_u8(env.render())
            x_t_np = preprocess_frame(frame, image_size=image_size)
            x_t = torch.tensor(x_t_np, dtype=torch.float32, device=device).unsqueeze(0)

            ee = as_np3(env.unwrapped.agent.tcp.pose.p)
            cube = as_np3(env.unwrapped.cube.pose.p)
            to_target = (cube - ee) if np.linalg.norm(cube[:2] - ee[:2]) > 0.06 else (target - ee)
            dxyz = np.clip(to_target / 0.04, -1.0, 1.0)
            action = np.array([dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            a_t = torch.tensor(action[None, :], dtype=torch.float32, device=device)

            risk = estimate_risk(wm, risk_head, x_t, a_t, horizon_steps=horizon_steps, n_samples=args.n_samples)
            max_r_seen = max(max_r_seen, risk)
            if risk > args.risk_threshold:
                stopped = True
                # stop action and send warning signal; no replanning.
                warn = draw_overlay(frame, cube[:2], ee[:2], center_xy, radius, risk, stopped=True)
                for _ in range(25):
                    frames.append(warn.copy())
                break

            env.step(action)
            ee2 = as_np3(env.unwrapped.agent.tcp.pose.p)
            cube2 = as_np3(env.unwrapped.cube.pose.p)
            vis = draw_overlay(frame, cube2[:2], ee2[:2], center_xy, radius, risk, stopped=False)
            frames.append(vis)

        video = os.path.join(args.output_dir, "videos", f"safety_ep_{ep:03d}.mp4")
        imageio.mimsave(video, frames, fps=10)
        results.append(
            {
                "episode": ep,
                "stopped": bool(stopped),
                "max_risk": float(max_r_seen),
                "video": video,
            }
        )

    env.close()
    out = {
        "risk_threshold": args.risk_threshold,
        "horizon_steps": horizon_steps,
        "summary": results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"saved: {os.path.join(args.output_dir, 'result.json')}")


if __name__ == "__main__":
    main()
