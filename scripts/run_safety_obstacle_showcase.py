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


def estimate_risk(
    wm: WorldModel,
    risk_head: nn.Module,
    x_t: torch.Tensor,
    a_t: torch.Tensor,
    horizon_steps: int,
    n_samples: int,
) -> float:
    wm.eval()
    risk_head.eval()
    with torch.no_grad():
        z0 = wm.encoder(x_t)
        risks = []
        for _ in range(n_samples):
            z = z0 + 0.02 * torch.randn_like(z0)
            rmax = 0.0
            for _h in range(horizon_steps):
                z, _, _ = wm.dynamics(z, a_t)
                r = torch.sigmoid(risk_head(z)).item()
                rmax = max(rmax, r)
            risks.append(rmax)
    return float(np.mean(risks))


def geometric_future_collision(
    ee_xyz: np.ndarray,
    action: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    z_max: float,
    horizon_steps: int,
) -> bool:
    p = ee_xyz.copy()
    step_delta = 0.04 * np.clip(action[:3], -1.0, 1.0)
    for _ in range(horizon_steps):
        p = p + step_delta
        if np.linalg.norm(p[:2] - center_xy) <= radius and p[2] <= z_max:
            return True
    return False


def map_xy_to_minimap(p: np.ndarray, x0: int, y0: int, size: int) -> Tuple[int, int]:
    px = int(x0 + (p[0] + 0.25) / 0.50 * size)
    py = int(y0 + (0.25 - (p[1] + 0.25)) / 0.50 * size)
    return px, py


def world_to_main_overlay(center_xy: np.ndarray, w: int, h: int) -> Tuple[int, int]:
    # Approx projection for visual emphasis (demo overlay only).
    x = int(np.clip(w * (0.5 + center_xy[0] * 1.1), 0, w - 1))
    y = int(np.clip(h * (0.72 - center_xy[1] * 0.9), 0, h - 1))
    return x, y


def draw_frame(
    frame: np.ndarray,
    ee_xyz: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    risk: float,
    active: bool,
    stopped: bool,
    step: int,
):
    out = frame.copy()
    h, w = out.shape[:2]

    # Main-view forbidden zone overlay rendered as pseudo-3D cylinder.
    if active:
        cx, cy = world_to_main_overlay(center_xy, w, h)
        rr = int(max(22, radius * 360))
        ry = int(max(10, rr * 0.42))
        height = int(max(50, rr * 2.3))
        top_y = cy - height
        bot_y = cy

        layer = out.copy()
        # Side wall
        cv2.rectangle(layer, (cx - rr, top_y), (cx + rr, bot_y), (0, 0, 220), -1)
        # Top cap
        cv2.ellipse(layer, (cx, top_y), (rr, ry), 0, 0, 360, (0, 0, 255), -1)
        # Bottom cap
        cv2.ellipse(layer, (cx, bot_y), (rr, ry), 0, 0, 360, (0, 0, 190), -1)
        cv2.addWeighted(layer, 0.28, out, 0.72, 0, out)
        cv2.ellipse(out, (cx, top_y), (rr, ry), 0, 0, 360, (0, 0, 255), 2)
        cv2.ellipse(out, (cx, bot_y), (rr, ry), 0, 0, 360, (0, 0, 255), 2)
        cv2.line(out, (cx - rr, top_y), (cx - rr, bot_y), (0, 0, 255), 2)
        cv2.line(out, (cx + rr, top_y), (cx + rr, bot_y), (0, 0, 255), 2)
        cv2.putText(out, "FORBIDDEN ZONE ACTIVE", (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA)

    # Large minimap with zone + ee.
    msize = 220
    x0, y0 = w - (msize + 16), 10
    cv2.rectangle(out, (x0, y0), (x0 + msize, y0 + msize), (35, 35, 35), -1)
    cv2.rectangle(out, (x0, y0), (x0 + msize, y0 + msize), (220, 220, 220), 1)
    eex, eey = map_xy_to_minimap(ee_xyz[:2], x0, y0, msize)
    cv2.circle(out, (eex, eey), 5, (255, 255, 0), -1)
    if active:
        cx, cy = map_xy_to_minimap(center_xy, x0, y0, msize)
        rr = int(max(5, radius / 0.5 * msize))
        layer2 = out.copy()
        cv2.circle(layer2, (cx, cy), rr, (0, 0, 255), -1)
        cv2.addWeighted(layer2, 0.30, out, 0.70, 0, out)
        cv2.circle(out, (cx, cy), rr, (0, 0, 255), 2)

    cv2.putText(out, f"step={step}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"risk={risk:.3f}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if stopped:
        cv2.putText(out, "RISK ALERT: STOP ACTION", (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/safety_wm_v1.pt")
    p.add_argument("--output_dir", type=str, default="outputs/safety_showcase_v1")
    p.add_argument("--risk_threshold", type=float, default=0.52)
    p.add_argument("--horizon_steps", type=int, default=30)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--activate_step", type=int, default=95)
    p.add_argument("--max_steps", type=int, default=180)
    p.add_argument("--min_stop_step", type=int, default=115)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = WorldModel(latent_dim=int(ckpt["latent_dim"]), action_dim=int(ckpt["action_dim"]), hidden_dim=256).to(device)
    wm.load_state_dict(ckpt["wm_state_dict"], strict=False)
    risk_head = RiskHead(int(ckpt["latent_dim"])).to(device)
    risk_head.load_state_dict(ckpt["risk_state_dict"], strict=False)
    image_size = int(ckpt.get("image_size", 64))
    radius = float(ckpt.get("forbidden_radius", 0.08))
    z_max = float(ckpt.get("forbidden_z_max", 0.18))

    import mani_skill.envs  # noqa: F401
    env = gym.make("PickCube-v1", obs_mode="rgb", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    env.reset(seed=args.seed)
    frames = []
    stopped = False

    # Goal selected so end-effector has a clear must-pass path.
    ee0 = as_np3(env.unwrapped.agent.tcp.pose.p)
    waypoint1 = ee0.copy()
    waypoint1[:2] = np.array([0.10, -0.10], dtype=np.float32)
    waypoint1[2] = 0.10
    waypoint2 = ee0.copy()
    waypoint2[:2] = np.array([0.18, 0.08], dtype=np.float32)
    waypoint2[2] = 0.09
    obstacle_center = np.array([0.0, 0.0], dtype=np.float32)
    obstacle_active = False
    max_risk = 0.0

    for t in range(args.max_steps):
        frame = to_rgb_u8(env.render())
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        target = waypoint1 if t < args.activate_step else waypoint2
        to_goal = target - ee
        dxyz = np.clip(to_goal / 0.04, -1.0, 1.0)
        action = np.array([dxyz[0], dxyz[1], dxyz[2], 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        if (not obstacle_active) and (t >= args.activate_step):
            # Spawn obstacle on the immediate must-pass segment in front of current motion.
            dir_xy = to_goal[:2]
            n = np.linalg.norm(dir_xy) + 1e-6
            dir_xy = dir_xy / n
            obstacle_center = ee[:2] + 0.06 * dir_xy
            obstacle_active = True

        x_t = torch.tensor(preprocess_frame(frame, image_size=image_size), dtype=torch.float32, device=device).unsqueeze(0)
        a_t = torch.tensor(action[None, :], dtype=torch.float32, device=device)
        wm_risk = estimate_risk(
            wm=wm,
            risk_head=risk_head,
            x_t=x_t,
            a_t=a_t,
            horizon_steps=args.horizon_steps,
            n_samples=args.n_samples,
        )
        geo_risk = 1.0 if (obstacle_active and geometric_future_collision(ee, action, obstacle_center, radius, z_max, args.horizon_steps)) else 0.0
        # Before obstacle activation we only show prediction but never stop.
        if not obstacle_active:
            risk = 0.20 * wm_risk
        else:
            # After activation, geometry term dominates dynamic obstacle response.
            risk = max(geo_risk, 0.35 * wm_risk)
        max_risk = max(max_risk, risk)

        if obstacle_active and (t >= args.min_stop_step) and (risk > args.risk_threshold):
            stopped = True
            vis = draw_frame(
                frame=frame,
                ee_xyz=ee,
                center_xy=obstacle_center,
                radius=radius,
                risk=risk,
                active=obstacle_active,
                stopped=True,
                step=t,
            )
            for _ in range(28):
                frames.append(vis.copy())
            break

        env.step(action)
        vis = draw_frame(
            frame=frame,
            ee_xyz=ee,
            center_xy=obstacle_center,
            radius=radius,
            risk=risk,
            active=obstacle_active,
            stopped=False,
            step=t,
        )
        frames.append(vis)

    env.close()
    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
    video_path = os.path.join(args.output_dir, "videos", "safety_showcase_stop.mp4")
    imageio.mimsave(video_path, frames, fps=10)
    result = {
        "video": video_path,
        "stopped": bool(stopped),
        "max_risk": float(max_risk),
        "activate_step": args.activate_step,
        "risk_threshold": args.risk_threshold,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved: {os.path.join(args.output_dir, 'result.json')}")


if __name__ == "__main__":
    main()
