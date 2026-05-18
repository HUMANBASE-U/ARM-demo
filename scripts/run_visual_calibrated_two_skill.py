import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn

GLOBAL_VIEW_SIZE = 256
NEAR_VIEW_SIZE = 160
NEAR_CROP_RATIO = 0.36
ALIGN_SIM_THRESHOLD = 0.82
ALIGN_REQUIRED_STREAK = 2
ALIGN_SIM_GRASP_OK = 0.72
GEOM_XY_GRASP_OK = 0.028
GEOM_Z_MIN = -0.008
GEOM_Z_MAX = 0.070


class GraspVisualActor(nn.Module):
    def __init__(self, obs_dim: int = 8, act_dim: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return torch.tanh(self.mu(h))


class PlaceVisualActor(nn.Module):
    def __init__(self, obs_dim: int = 10, act_dim: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return torch.tanh(self.mu(h))


def as_np3(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    return x[:3].astype(np.float32)


def to_rgb_u8(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame[..., :3]


def red_cube_uv(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (165, 80, 60), (179, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    ys, xs = np.where(mask > 0)
    h, w = img.shape[:2]
    if len(xs) == 0:
        return np.array([0.5, 0.5], dtype=np.float32)
    return np.array([float(xs.mean() / w), float(ys.mean() / h)], dtype=np.float32)


def grip_open_ratio(env) -> float:
    q = env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy().astype(np.float32)
    return float(np.clip(np.mean(q[-2:]) / 0.04, 0.0, 1.0))


def build_corners(z: float, span_x: float = 0.18, span_y: float = 0.18) -> Dict[str, np.ndarray]:
    return {
        "top_left": np.array([-span_x, +span_y, z], dtype=np.float32),
        "top_right": np.array([+span_x, +span_y, z], dtype=np.float32),
        "bottom_left": np.array([-span_x, -span_y, z], dtype=np.float32),
        "bottom_right": np.array([+span_x, -span_y, z], dtype=np.float32),
    }


def llm_call(base_url: str, api_key: str, model: str, rules: str, payload: Dict, timeout_s: int) -> Dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "temperature": 0, "messages": [{"role": "system", "content": rules}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        s = text.lower()
        if ("left" in s or "左" in s) and ("top" in s or "上" in s):
            corner = "top_left"
        elif ("right" in s or "右" in s) and ("top" in s or "上" in s):
            corner = "top_right"
        elif ("left" in s or "左" in s) and ("bottom" in s or "下" in s):
            corner = "bottom_left"
        elif ("right" in s or "右" in s) and ("bottom" in s or "下" in s):
            corner = "bottom_right"
        else:
            corner = "top_left"
        return {"target_corner": corner}


def load_grasp_actor(path: str, device: torch.device) -> GraspVisualActor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = GraspVisualActor(obs_dim=int(ckpt.get("obs_dim", 8)), act_dim=int(ckpt.get("act_dim", 4))).to(device)
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()
    return net


def load_place_actor(path: str, device: torch.device) -> PlaceVisualActor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = PlaceVisualActor(obs_dim=10, act_dim=3).to(device)
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()
    return net


def action7(dx: float, dy: float, dz: float, grip: float, rz: float = 0.0) -> np.ndarray:
    return np.array([dx, dy, dz, 0.0, 0.0, rz, grip], dtype=np.float32)


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_error_cube_vs_gripper(env) -> float:
    ee_q = env.unwrapped.agent.tcp.pose.q[0].detach().cpu().numpy().astype(np.float32)
    cube_q = env.unwrapped.cube.pose.q[0].detach().cpu().numpy().astype(np.float32)
    yaw_ee = yaw_from_quat_wxyz(ee_q)
    yaw_cube = yaw_from_quat_wxyz(cube_q)
    return wrap_angle(yaw_cube - yaw_ee)


def close_and_squeeze(
    env,
    touch_xyz: np.ndarray,
    frames: List[np.ndarray],
    target_xy: np.ndarray,
    radius: float,
    skill_prefix: str,
):
    # Phase A: settle at touch with open gripper.
    for _ in range(6):
        yaw_err = yaw_error_cube_vs_gripper(env)
        rz_cmd = float(np.clip(1.6 * yaw_err, -0.6, 0.6))
        execute_to_point(
            env,
            touch_xyz,
            1.0,
            1,
            frames,
            target_xy,
            radius,
            f"{skill_prefix}_settle_open",
            speed_scale=0.25,
            rz_cmd=rz_cmd,
        )
    # Phase B: explicit close while maintaining pose.
    for i in range(10):
        yaw_err = yaw_error_cube_vs_gripper(env)
        rz_cmd = float(np.clip(1.4 * yaw_err, -0.5, 0.5))
        g = float(1.0 - 2.0 * (i + 1) / 10.0)  # 1 -> -1
        execute_to_point(
            env,
            touch_xyz,
            g,
            1,
            frames,
            target_xy,
            radius,
            f"{skill_prefix}_close",
            speed_scale=0.18,
            rz_cmd=rz_cmd,
        )
    # Phase C: squeeze hold.
    for _ in range(8):
        yaw_err = yaw_error_cube_vs_gripper(env)
        rz_cmd = float(np.clip(1.2 * yaw_err, -0.4, 0.4))
        execute_to_point(
            env,
            touch_xyz,
            -1.0,
            1,
            frames,
            target_xy,
            radius,
            f"{skill_prefix}_squeeze",
            speed_scale=0.12,
            rz_cmd=rz_cmd,
        )


def extract_dual_views(obs, env, near_tracker: Optional["NearViewTracker"] = None) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(obs, dict) and "rgb" in obs:
        rgb = obs["rgb"]
        if isinstance(rgb, dict):
            keys = list(rgb.keys())
            arrs = []
            for k in keys:
                img = rgb[k]
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()
                if img.ndim == 4:
                    img = img[0]
                if img.dtype != np.uint8:
                    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                arrs.append(img[..., :3])
            if len(arrs) >= 2:
                g = cv2.resize(arrs[0], (GLOBAL_VIEW_SIZE, GLOBAL_VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
                if near_tracker is None:
                    n = cv2.resize(arrs[1], (NEAR_VIEW_SIZE, NEAR_VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
                else:
                    base = cv2.resize(arrs[0], (GLOBAL_VIEW_SIZE, GLOBAL_VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
                    c = _left_face_center_uv(env, base, near_tracker)
                    h, w = base.shape[:2]
                    win = int(max(36, min(h, w) * NEAR_CROP_RATIO))
                    x0 = int(np.clip(c[0] - win // 2, 0, w - win))
                    y0 = int(np.clip(c[1] - win // 2, 0, h - win))
                    n = base[y0 : y0 + win, x0 : x0 + win]
                    n = cv2.resize(n, (NEAR_VIEW_SIZE, NEAR_VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
                return g, n
            if len(arrs) == 1:
                fr = cv2.resize(arrs[0], (GLOBAL_VIEW_SIZE, GLOBAL_VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
                h, w = fr.shape[:2]
                if near_tracker is None:
                    uv = red_cube_uv(fr)
                    center = np.array([float(np.clip(uv[0] * w, 0, w - 1)), float(np.clip(uv[1] * h - 0.10 * h, 0, h - 1))], dtype=np.float32)
                else:
                    center = _left_face_center_uv(env, fr, near_tracker)
                win = int(max(36, min(h, w) * NEAR_CROP_RATIO))
                x0 = int(np.clip(center[0] - win // 2, 0, w - win))
                y0 = int(np.clip(center[1] - win // 2, 0, h - win))
                near = fr[y0 : y0 + win, x0 : x0 + win]
                near = cv2.resize(near, (NEAR_VIEW_SIZE, NEAR_VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
                return fr, near
    fr = to_rgb_u8(env.render())
    fr = cv2.resize(fr, (GLOBAL_VIEW_SIZE, GLOBAL_VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
    h, w = fr.shape[:2]
    if near_tracker is None:
        uv = red_cube_uv(fr)
        center = np.array([float(np.clip(uv[0] * w, 0, w - 1)), float(np.clip(uv[1] * h - 0.12 * h, 0, h - 1))], dtype=np.float32)
    else:
        center = _left_face_center_uv(env, fr, near_tracker)
    win = int(max(36, min(h, w) * NEAR_CROP_RATIO))
    x0 = int(np.clip(center[0] - win // 2, 0, w - win))
    y0 = int(np.clip(center[1] - win // 2, 0, h - win))
    near = fr[y0 : y0 + win, x0 : x0 + win]
    near = cv2.resize(near, (NEAR_VIEW_SIZE, NEAR_VIEW_SIZE), interpolation=cv2.INTER_LINEAR)
    return fr, near


def vis_feature_from_views(view1: np.ndarray, view2: np.ndarray, env) -> np.ndarray:
    uv1 = red_cube_uv(view1)
    uv2 = red_cube_uv(view2)
    ee = as_np3(env.unwrapped.agent.tcp.pose.p)
    cube = as_np3(env.unwrapped.cube.pose.p)
    rel = cube - ee
    grip = np.array([grip_open_ratio(env)], dtype=np.float32)
    return np.concatenate([uv1, uv2, rel, grip], axis=0).astype(np.float32)  # 8D


def overlay_target_circle(frame: np.ndarray, cube_xy: np.ndarray, target_xy: np.ndarray, radius: float, skill: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    # minimap overlay
    x0, y0 = w - 170, 10
    cv2.rectangle(out, (x0, y0), (x0 + 160, y0 + 160), (40, 40, 40), -1)
    cv2.rectangle(out, (x0, y0), (x0 + 160, y0 + 160), (220, 220, 220), 1)

    def map_xy(p):
        # world map range [-0.25,0.25]
        px = int(x0 + (p[0] + 0.25) / 0.50 * 160)
        py = int(y0 + (0.25 - (p[1] + 0.25)) / 0.50 * 160)
        return px, py

    cpx, cpy = map_xy(cube_xy)
    tpx, tpy = map_xy(target_xy)
    rr = int(max(3, radius / 0.50 * 160))
    cv2.circle(out, (tpx, tpy), rr, (0, 255, 0), 2)
    cv2.circle(out, (cpx, cpy), 4, (0, 0, 255), -1)
    dist = float(np.linalg.norm(cube_xy - target_xy))
    cv2.putText(out, f"{skill}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"xy-dist={dist:.3f}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def make_line_points(start: np.ndarray, end: np.ndarray, n: int) -> List[np.ndarray]:
    pts = []
    for i in range(1, n + 1):
        a = i / n
        pts.append(((1 - a) * start + a * end).astype(np.float32))
    return pts


def execute_to_point(
    env,
    target_xyz: np.ndarray,
    grip_cmd: float,
    steps: int,
    frames: List[np.ndarray],
    target_xy: np.ndarray,
    radius: float,
    skill_name: str,
    speed_scale: float = 1.0,
    rz_cmd: float = 0.0,
):
    for _ in range(steps):
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        d = np.clip(speed_scale * (target_xyz - ee) / 0.04, -1.0, 1.0)
        env.step(action7(float(d[0]), float(d[1]), float(d[2]), float(np.clip(grip_cmd, -1.0, 1.0)), rz=float(np.clip(rz_cmd, -1.0, 1.0))))
        cube = as_np3(env.unwrapped.cube.pose.p)
        fr = to_rgb_u8(env.render())
        frames.append(overlay_target_circle(fr, cube[:2], target_xy, radius, skill_name))
        if np.linalg.norm(ee - target_xyz) < 0.015:
            break


def align_checkpoint(
    env,
    waypoint: np.ndarray,
    target_feat: np.ndarray,
    frames: List[np.ndarray],
    target_xy: np.ndarray,
    radius: float,
    skill_name: str,
    near_tracker: Optional["NearViewTracker"],
    force_grip_cmd: Optional[float] = None,
    max_steps: int = 30,
):
    streak = 0
    for _ in range(max_steps):
        grip_cmd = grip_open_ratio(env) * 2 - 1 if force_grip_cmd is None else force_grip_cmd
        execute_to_point(env, waypoint, grip_cmd, 1, frames, target_xy, radius, skill_name, speed_scale=0.35)
        v1, v2 = extract_dual_views(None, env, near_tracker=near_tracker)
        feat = vis_feature_from_views(v1, v2, env)
        sim = float(np.exp(-6.0 * np.linalg.norm(feat - target_feat)))
        if sim > ALIGN_SIM_THRESHOLD:
            streak += 1
        else:
            streak = 0
        if streak >= ALIGN_REQUIRED_STREAK:
            break


def run_grasp_skill(
    env,
    grasp_actor: GraspVisualActor,
    target_xy: np.ndarray,
    radius: float,
    device: torch.device,
    templates: Optional[List[np.ndarray]],
    frames: List[np.ndarray],
    near_tracker: Optional["NearViewTracker"],
) -> Tuple[bool, List[np.ndarray], int]:
    table_z = float(as_np3(env.unwrapped.cube.pose.p)[2])
    ee0 = as_np3(env.unwrapped.agent.tcp.pose.p)
    cube0 = as_np3(env.unwrapped.cube.pose.p)
    approach_dir = cube0[:2] - ee0[:2]
    n_dir = float(np.linalg.norm(approach_dir))
    if n_dir < 1e-6:
        approach_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        approach_dir = approach_dir / n_dir
    # Better calibrated grasp point: avoid back-side grasp, target cube center with small view-direction bias.
    grasp_xy = cube0[:2] - 0.008 * approach_dir
    hover = np.array([grasp_xy[0], grasp_xy[1], 0.09], dtype=np.float32)
    touch = np.array([grasp_xy[0], grasp_xy[1], float(cube0[2] + 0.016)], dtype=np.float32)

    # attempt multiple lines until grasp success
    new_templates = templates[:] if templates is not None else None
    total_steps = 0
    correction_xy = np.zeros((2,), dtype=np.float32)
    for attempt in range(4):
        # failure handling: immediately open gripper before retry
        if attempt > 0:
            for _ in range(15):
                env.step(action7(0.0, 0.0, 0.0, 1.0))
                cube = as_np3(env.unwrapped.cube.pose.p)
                fr = to_rgb_u8(env.render())
                frames.append(overlay_target_circle(fr, cube[:2], target_xy, radius, "grasp_retry_open"))
                total_steps += 1

        # adaptive correction from previous failed grasp, not fixed repeated descent
        offset = np.array([correction_xy[0], correction_xy[1], 0.0], dtype=np.float32)
        hover_a = hover + offset
        touch_a = touch + offset

        line_pts = make_line_points(ee0, hover_a, 12) + make_line_points(hover_a, touch_a, 10)
        cidx = [len(line_pts) // 3, (2 * len(line_pts)) // 3, len(line_pts) - 1]
        captured = []
        for i, wp in enumerate(line_pts):
            # Before alignment is complete, gripper is forced fully open.
            ee = as_np3(env.unwrapped.agent.tcp.pose.p)
            cube = as_np3(env.unwrapped.cube.pose.p)
            grip_open = np.array([grip_open_ratio(env)], dtype=np.float32)
            obs_rl = np.concatenate([ee, cube, grip_open, np.array([0.0], dtype=np.float32)], axis=0).astype(np.float32)
            ot = torch.tensor(obs_rl, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu = grasp_actor(ot).squeeze(0).detach().cpu().numpy().astype(np.float32)
            d_model = 0.5 * (wp - ee) + 0.5 * 0.06 * np.clip(mu[:3], -1.0, 1.0)
            gcmd = 1.0
            yaw_err = yaw_error_cube_vs_gripper(env)
            rz_cmd = float(np.clip(1.6 * yaw_err, -0.55, 0.55))
            act = action7(
                float(np.clip(d_model[0] / 0.04, -1, 1)),
                float(np.clip(d_model[1] / 0.04, -1, 1)),
                float(np.clip(d_model[2] / 0.04, -1, 1)),
                gcmd,
                rz=rz_cmd,
            )
            obs, _, _, _, _ = env.step(act)
            v1, v2 = extract_dual_views(obs, env, near_tracker=near_tracker)
            feat = vis_feature_from_views(v1, v2, env)
            if i in cidx:
                captured.append(feat.copy())
                if new_templates is not None and len(new_templates) >= len(captured):
                    align_checkpoint(
                        env,
                        wp,
                        new_templates[len(captured) - 1],
                        frames,
                        target_xy,
                        radius,
                        "grasp_calib",
                        near_tracker=near_tracker,
                        force_grip_cmd=1.0,
                    )
            cube_now = as_np3(env.unwrapped.cube.pose.p)
            fr = to_rgb_u8(env.render())
            frames.append(overlay_target_circle(fr, cube_now[:2], target_xy, radius, "grasp_skill"))
            total_steps += 1

        # calibrated pre-grasp hover: hold open gripper, align near-view to reduce unstable off-center grasp.
        stable = 0
        best_sim = 0.0
        best_geom = 1e9
        for _ in range(15):
            yaw_err = yaw_error_cube_vs_gripper(env)
            rz_cmd = float(np.clip(1.8 * yaw_err, -0.65, 0.65))
            execute_to_point(env, touch, 1.0, 1, frames, target_xy, radius, "grasp_pre_align", speed_scale=0.28, rz_cmd=rz_cmd)
            v1, v2 = extract_dual_views(None, env, near_tracker=near_tracker)
            feat = vis_feature_from_views(v1, v2, env)
            rel = as_np3(env.unwrapped.cube.pose.p) - as_np3(env.unwrapped.agent.tcp.pose.p)
            visual_ok = False
            sim = 0.0
            if len(captured) > 0:
                sim = float(np.exp(-6.0 * np.linalg.norm(feat - captured[-1])))
                visual_ok = sim > ALIGN_SIM_GRASP_OK
            best_sim = max(best_sim, sim)
            geom_ok = abs(float(rel[0])) < GEOM_XY_GRASP_OK and abs(float(rel[1])) < GEOM_XY_GRASP_OK and GEOM_Z_MIN < float(rel[2]) < GEOM_Z_MAX
            best_geom = min(best_geom, float(np.linalg.norm(rel[:2])))
            if visual_ok and geom_ok:
                stable += 1
            else:
                stable = 0
            if stable >= ALIGN_REQUIRED_STREAK:
                break

        good_enough = (best_sim > 0.66) and (best_geom < 0.035)
        if (stable < ALIGN_REQUIRED_STREAK) and (not good_enough):
            # Not aligned enough: apply calibration correction for next attempt (instead of repeating same path).
            rel = as_np3(env.unwrapped.cube.pose.p) - as_np3(env.unwrapped.agent.tcp.pose.p)
            correction_xy += np.clip(np.array([rel[0], rel[1]], dtype=np.float32), -0.012, 0.012)
            continue

        # Critical bugfix: explicit close-at-touch stage once alignment is good enough.
        close_and_squeeze(env, touch, frames, target_xy, radius, "grasp")
        total_steps += 24

        # lift to verify grasp success
        for _ in range(15):
            env.step(action7(0.0, 0.0, 0.6, -1.0))
            cube_now = as_np3(env.unwrapped.cube.pose.p)
            fr = to_rgb_u8(env.render())
            frames.append(overlay_target_circle(fr, cube_now[:2], target_xy, radius, "grasp_lift_check"))
            total_steps += 1
        cube_lift = as_np3(env.unwrapped.cube.pose.p)
        grasp_ok = bool(cube_lift[2] > table_z + 0.025)
        if grasp_ok:
            if new_templates is None:
                new_templates = captured[:3]
            return True, (new_templates if new_templates is not None else captured[:3]), total_steps
        # failed close/lift => calibration for next attempt
        rel = as_np3(env.unwrapped.cube.pose.p) - as_np3(env.unwrapped.agent.tcp.pose.p)
        correction_xy += np.clip(np.array([rel[0], rel[1]], dtype=np.float32), -0.012, 0.012)
    return False, (new_templates if new_templates is not None else []), total_steps


def run_place_skill(
    env,
    place_actor: PlaceVisualActor,
    target_xyz: np.ndarray,
    target_xy: np.ndarray,
    radius: float,
    device: torch.device,
    templates: Optional[List[np.ndarray]],
    frames: List[np.ndarray],
    near_tracker: Optional["NearViewTracker"],
) -> Tuple[List[np.ndarray], int]:
    ee0 = as_np3(env.unwrapped.agent.tcp.pose.p)
    hover = target_xyz.copy()
    hover[2] = max(target_xyz[2] + 0.12, 0.12)
    down = target_xyz.copy()
    down[2] = target_xyz[2] + 0.02
    line_pts = make_line_points(ee0, hover, 14) + make_line_points(hover, down, 12)
    cidx = [len(line_pts) // 3, (2 * len(line_pts)) // 3, len(line_pts) - 1]
    captured = []
    total = 0
    for i, wp in enumerate(line_pts):
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        cube = as_np3(env.unwrapped.cube.pose.p)
        obs_rl = np.concatenate([ee, cube, target_xyz, np.array([1.0], dtype=np.float32)], axis=0).astype(np.float32)
        ot = torch.tensor(obs_rl, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu = place_actor(ot).squeeze(0).detach().cpu().numpy().astype(np.float32)
        d_model = 0.5 * (wp - ee) + 0.5 * 0.06 * np.clip(mu, -1.0, 1.0)
        act = action7(float(np.clip(d_model[0] / 0.04, -1, 1)), float(np.clip(d_model[1] / 0.04, -1, 1)), float(np.clip(d_model[2] / 0.04, -1, 1)), -1.0)
        obs, _, _, _, _ = env.step(act)
        v1, v2 = extract_dual_views(obs, env, near_tracker=near_tracker)
        feat = vis_feature_from_views(v1, v2, env)
        if i in cidx:
            captured.append(feat.copy())
            if templates is not None and len(templates) >= len(captured):
                align_checkpoint(
                    env,
                    wp,
                    templates[len(captured) - 1],
                    frames,
                    target_xy,
                    radius,
                    "place_calib",
                    near_tracker=near_tracker,
                )
        cube_now = as_np3(env.unwrapped.cube.pose.p)
        fr = to_rgb_u8(env.render())
        frames.append(overlay_target_circle(fr, cube_now[:2], target_xy, radius, "place_skill"))
        total += 1

    # release
    for _ in range(16):
        env.step(action7(0.0, 0.0, 0.0, 1.0))
        cube_now = as_np3(env.unwrapped.cube.pose.p)
        fr = to_rgb_u8(env.render())
        frames.append(overlay_target_circle(fr, cube_now[:2], target_xy, radius, "place_release"))
        total += 1
    if templates is None:
        templates = captured[:3]
    return templates, total


@dataclass
class EpisodeOut:
    success: bool
    steps: int
    video: str
    target_corner: str


@dataclass
class NearViewTracker:
    center_uv: Optional[np.ndarray] = None


def _left_face_center_uv(env, frame: np.ndarray, tracker: NearViewTracker) -> np.ndarray:
    h, w = frame.shape[:2]
    cube_uv = red_cube_uv(frame)
    cube_px = np.array([cube_uv[0] * w, cube_uv[1] * h], dtype=np.float32)
    ee = as_np3(env.unwrapped.agent.tcp.pose.p)
    cube = as_np3(env.unwrapped.cube.pose.p)
    v = cube[:2] - ee[:2]
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        v = np.array([1.0, 0.0], dtype=np.float32)
    else:
        v = v / nv
    # Left side from arm->cube view direction.
    left_world = np.array([-v[1], v[0]], dtype=np.float32)
    left_img = np.array([left_world[0], -left_world[1]], dtype=np.float32)
    offset_px = 0.12 * min(h, w)
    center = cube_px + offset_px * left_img
    if tracker.center_uv is None:
        smoothed = center
    else:
        smoothed = 0.75 * tracker.center_uv + 0.25 * center
    smoothed[0] = float(np.clip(smoothed[0], 0, w - 1))
    smoothed[1] = float(np.clip(smoothed[1], 0, h - 1))
    tracker.center_uv = smoothed
    return smoothed


def save_metrics_curve(results: List[EpisodeOut], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = np.arange(1, len(results) + 1)
    s = np.array([1.0 if r.success else 0.0 for r in results], dtype=np.float32)
    cum = np.cumsum(s) / x
    st = np.array([r.steps for r in results], dtype=np.float32)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, s, marker="o", label="episode success")
    ax[0].plot(x, cum, marker="s", label="cumulative")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title("Success Curve")
    ax[0].grid(alpha=0.2)
    ax[0].legend()
    ax[1].plot(x, st, marker="o")
    ax[1].set_title("Steps")
    ax[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def run_episode(
    env,
    target_name: str,
    corners: Dict[str, np.ndarray],
    grasp_actor: GraspVisualActor,
    place_actor: PlaceVisualActor,
    device: torch.device,
    success_radius: float,
    out_dir: str,
    ep_idx: int,
    grasp_templates: Optional[List[np.ndarray]],
    place_templates: Optional[List[np.ndarray]],
) -> Tuple[EpisodeOut, List[np.ndarray], List[np.ndarray]]:
    target = corners[target_name]
    target_xy = target[:2]
    frames = []
    total_steps = 0
    near_tracker = NearViewTracker()

    ok, grasp_templates, c = run_grasp_skill(
        env=env,
        grasp_actor=grasp_actor,
        target_xy=target_xy,
        radius=success_radius,
        device=device,
        templates=grasp_templates,
        frames=frames,
        near_tracker=near_tracker,
    )
    total_steps += c

    if ok:
        place_templates, c2 = run_place_skill(
            env=env,
            place_actor=place_actor,
            target_xyz=target,
            target_xy=target_xy,
            radius=success_radius,
            device=device,
            templates=place_templates,
            frames=frames,
            near_tracker=near_tracker,
        )
        total_steps += c2

    cube = as_np3(env.unwrapped.cube.pose.p)
    success = bool(np.linalg.norm(cube[:2] - target_xy) <= success_radius)
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
    video = os.path.join(out_dir, "videos", f"ep_{ep_idx:03d}.mp4")
    imageio.mimsave(video, frames, fps=10)
    return EpisodeOut(success=success, steps=total_steps, video=video, target_corner=target_name), grasp_templates, place_templates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_rules_path", type=str, default="LLM-rules")
    parser.add_argument("--grasp_checkpoint", type=str, default="checkpoints/grasp_visual_actor_v1.pt")
    parser.add_argument("--place_checkpoint", type=str, default="checkpoints/move_to_visual_actor_v3.pt")
    parser.add_argument("--success_radius", type=float, default=0.08)
    parser.add_argument("--output_dir", type=str, default="outputs/llm_full_v7_leftface_dual")
    parser.add_argument("--output_json", type=str, default="outputs/llm_full_v7_leftface_dual/result.json")
    parser.add_argument("--llm_timeout_s", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_url = args.api_base_url or os.environ.get("LLM_API_BASE_URL")
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not base_url or not api_key:
        raise ValueError("Missing API settings")

    with open(args.llm_rules_path, "r", encoding="utf-8") as f:
        rules = f.read()

    import mani_skill.envs  # noqa: F401
    env = gym.make("PickCube-v1", obs_mode="rgb", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    grasp_actor = load_grasp_actor(args.grasp_checkpoint, device=device)
    place_actor = load_place_actor(args.place_checkpoint, device=device)

    instructions = [
        "把cube抓起来，放到左上角",
        "把cube抓起来，放到右上角",
        "把cube抓起来，放到左下角",
        "把cube抓起来，放到右下角",
        "Pick up cube then place top left",
        "Pick up cube then place top right",
        "Pick up cube then place bottom left",
        "Pick up cube then place bottom right",
    ]
    grasp_templates, place_templates = None, None
    results: List[EpisodeOut] = []
    os.makedirs(args.output_dir, exist_ok=True)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        cube = as_np3(env.unwrapped.cube.pose.p)
        corners = build_corners(cube[2])
        instruction = instructions[ep % len(instructions)]
        payload = {
            "instruction": instruction,
            "corners": {k: v.tolist() for k, v in corners.items()},
            "robot": {"ee_pos": as_np3(env.unwrapped.agent.tcp.pose.p).tolist()},
            "cube": {"pos": cube.tolist()},
        }
        llm_out = llm_call(base_url=base_url, api_key=api_key, model=args.model, rules=rules, payload=payload, timeout_s=args.llm_timeout_s)
        corner = llm_out.get("target_corner", ["top_left", "top_right", "bottom_left", "bottom_right"][ep % 4])
        if corner not in corners:
            corner = ["top_left", "top_right", "bottom_left", "bottom_right"][ep % 4]

        print(f"[Episode {ep+1}/{args.episodes}] target={corner} skills=2(fixed)")
        out, grasp_templates, place_templates = run_episode(
            env=env,
            target_name=corner,
            corners=corners,
            grasp_actor=grasp_actor,
            place_actor=place_actor,
            device=device,
            success_radius=args.success_radius,
            out_dir=args.output_dir,
            ep_idx=ep,
            grasp_templates=grasp_templates,
            place_templates=place_templates,
        )
        results.append(out)
        print(f"  success={out.success} steps={out.steps} video={out.video}")

    env.close()
    success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results])) if results else 0.0
    metrics_curve = os.path.join(args.output_dir, "metrics_curve.png")
    save_metrics_curve(results, metrics_curve)
    summary = {
        "success_rate": success_rate,
        "episodes": args.episodes,
        "grasp_checkpoint": args.grasp_checkpoint,
        "place_checkpoint": args.place_checkpoint,
        "metrics_curve": metrics_curve,
        "results": [
            {
                "target_corner": r.target_corner,
                "success": r.success,
                "steps": r.steps,
                "video": r.video,
            }
            for r in results
        ],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved result summary: {args.output_json}")
    print(f"Success rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
