import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_rgb_uint8(frame) -> np.ndarray:
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


def red_cube_centroid(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (165, 80, 60), (179, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = frame.shape[:2]
        return np.array([w / 2, h / 2], dtype=np.float32)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def build_corners(z: float, span_x: float = 0.18, span_y: float = 0.18) -> Dict[str, np.ndarray]:
    return {
        "top_left": np.array([-span_x, +span_y, z], dtype=np.float32),
        "top_right": np.array([+span_x, +span_y, z], dtype=np.float32),
        "bottom_left": np.array([-span_x, -span_y, z], dtype=np.float32),
        "bottom_right": np.array([+span_x, -span_y, z], dtype=np.float32),
    }


def make_obs_vec(ee: np.ndarray, cube: np.ndarray, target_corner: np.ndarray, snap_idx: int, num_snaps: int) -> np.ndarray:
    s = np.array([snap_idx / max(num_snaps, 1)], dtype=np.float32)
    return np.concatenate([ee, cube, target_corner, s], axis=0).astype(np.float32)


def skill_action(delta_xyz: np.ndarray) -> np.ndarray:
    # control_mode=pd_ee_delta_pose => 7D action
    d = np.clip(delta_xyz / 0.05, -1.0, 1.0)
    return np.array([d[0], d[1], d[2], 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def heuristic_delta(ee: np.ndarray, cube: np.ndarray, target_corner: np.ndarray) -> np.ndarray:
    to_goal = target_corner[:2] - cube[:2]
    n = np.linalg.norm(to_goal) + 1e-6
    dir2 = to_goal / n
    behind = cube.copy()
    behind[:2] = cube[:2] - 0.05 * dir2
    behind[2] = 0.05
    if np.linalg.norm(ee[:2] - behind[:2]) > 0.04:
        tgt = behind
    else:
        tgt = cube.copy()
        tgt[:2] = cube[:2] + 0.08 * dir2
        tgt[2] = 0.04
    return (tgt - ee).astype(np.float32)


def visual_feature(env, frame: np.ndarray) -> np.ndarray:
    cube_uv = red_cube_centroid(frame)
    h, w = frame.shape[:2]
    cube_uv = np.array([cube_uv[0] / w, cube_uv[1] / h], dtype=np.float32)
    ee = as_np3(env.unwrapped.agent.tcp.pose.p)
    qpos = env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy().astype(np.float32)
    arm = qpos[:4]
    return np.concatenate([cube_uv, ee, arm], axis=0).astype(np.float32)  # 9d


def choose_snapshots(features: List[np.ndarray], min_gap: int = 12, max_k: int = 3) -> List[np.ndarray]:
    n = len(features)
    if n == 0:
        return []
    feats = np.stack(features, axis=0)
    if n <= max_k:
        return [f.copy() for f in feats]
    # turning-score from cube uv motion
    uv = feats[:, :2]
    vel = uv[1:] - uv[:-1]
    score = np.zeros((n,), dtype=np.float32)
    for i in range(2, n):
        a = vel[i - 2]
        b = vel[i - 1]
        na = np.linalg.norm(a) + 1e-6
        nb = np.linalg.norm(b) + 1e-6
        cosv = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        score[i] = 1.0 - cosv
    candidates = np.argsort(-score).tolist()
    chosen = []
    for idx in candidates:
        if idx < min_gap or idx > n - min_gap:
            continue
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(idx)
        if len(chosen) >= max_k:
            break
    if len(chosen) == 0:
        chosen = [n // 3, 2 * n // 3]
    chosen = sorted(chosen)[:max_k]
    return [feats[i].copy() for i in chosen]


def heuristic_push_episode(env, target_corner: np.ndarray, horizon: int = 140, collect_demo: bool = False):
    obs, _ = env.reset()
    frames, features = [], []
    demos = []
    success = False
    snap_idx = 0
    num_snaps = 1
    for _ in range(horizon):
        frame = to_rgb_uint8(env.render())
        cube = as_np3(env.unwrapped.cube.pose.p)
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        to_goal = target_corner[:2] - cube[:2]
        n = np.linalg.norm(to_goal) + 1e-6
        dir2 = to_goal / n
        behind = cube.copy()
        behind[:2] = cube[:2] - 0.05 * dir2
        behind[2] = 0.05

        if np.linalg.norm(ee[:2] - behind[:2]) > 0.04:
            target = behind
        else:
            target = cube.copy()
            target[:2] = cube[:2] + 0.08 * dir2
            target[2] = 0.04

        delta = target - ee
        action = skill_action(delta)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(frame)
        features.append(visual_feature(env, frame))
        if collect_demo:
            demos.append(
                {
                    "obs": make_obs_vec(ee, cube, target_corner, snap_idx, num_snaps),
                    "act": np.clip(delta / 0.05, -1.0, 1.0).astype(np.float32),
                }
            )
        if np.linalg.norm(cube[:2] - target_corner[:2]) < 0.06:
            success = True
            break
        if terminated or truncated:
            break
    return {"success": success, "frames": frames, "features": features, "demos": demos}


def build_snapshot_library(env, attempts: int = 60) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
    z = float(as_np3(env.unwrapped.cube.pose.p)[2]) if hasattr(env.unwrapped, "cube") else 0.02
    corners = list(build_corners(z).values())
    successful = []
    demo_pool = []
    for _ in range(attempts):
        tgt = corners[np.random.randint(0, len(corners))]
        out = heuristic_push_episode(env, tgt, horizon=160, collect_demo=True)
        if out["success"]:
            successful.append(out)
            demo_pool.extend(out["demos"])
            if len(successful) >= 5:
                break
    if not successful:
        # fallback: create synthetic anchors from one rollout
        tgt = corners[0]
        out = heuristic_push_episode(env, tgt, horizon=100, collect_demo=True)
        return choose_snapshots(out["features"], min_gap=10, max_k=3), out["demos"]
    best = max(successful, key=lambda x: len(x["features"]))
    return choose_snapshots(best["features"], min_gap=12, max_k=3), demo_pool


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 13, act_dim: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)
        self.value = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.6))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = torch.tanh(self.mu(h))
        value = self.value(h).squeeze(-1)
        std = torch.exp(self.log_std).clamp(1e-3, 0.8)
        return mu, std, value


def pretrain_actor_with_demos(model: ActorCritic, demos: List[Dict[str, np.ndarray]], device: torch.device, epochs: int = 8, batch_size: int = 256) -> None:
    if len(demos) == 0:
        return
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    obs = torch.tensor(np.stack([d["obs"] for d in demos], axis=0), dtype=torch.float32, device=device)
    act = torch.tensor(np.stack([d["act"] for d in demos], axis=0), dtype=torch.float32, device=device)
    n = obs.size(0)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        loss_sum, cnt = 0.0, 0
        for s in range(0, n, batch_size):
            j = idx[s : s + batch_size]
            mu, _, _ = model(obs[j])
            loss = F.mse_loss(mu, act[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            cnt += 1
        print(f"[pretrain] epoch={ep+1}/{epochs} mse={loss_sum/max(cnt,1):.5f}")


def make_obs(env, target_corner: np.ndarray, snap_idx: int, num_snaps: int) -> np.ndarray:
    ee = as_np3(env.unwrapped.agent.tcp.pose.p)
    cube = as_np3(env.unwrapped.cube.pose.p)
    return make_obs_vec(ee, cube, target_corner, snap_idx, num_snaps)


def ppo_gae(rew, val, done, gamma=0.99, lam=0.95):
    n = len(rew)
    adv = np.zeros((n,), dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        nv = 0.0 if t == n - 1 else val[t + 1]
        mask = 1.0 - done[t]
        delta = rew[t] + gamma * nv * mask - val[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    ret = adv + val
    return adv, ret


@dataclass
class TrainStats:
    step_counts: List[int]
    policy_losses: List[float]
    value_losses: List[float]
    total_losses: List[float]
    mean_returns: List[float]
    success_rates: List[float]


def train_actor(
    env,
    snapshots: List[np.ndarray],
    total_steps: int,
    device: torch.device,
    success_radius: float,
    init_model: ActorCritic = None,
) -> Tuple[ActorCritic, TrainStats]:
    model = init_model.to(device) if init_model is not None else ActorCritic(obs_dim=10, act_dim=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = TrainStats([], [], [], [], [], [])

    rollout = 2048
    ppo_epochs = 8
    mb = 256
    clip_eps = 0.2
    ent_coef = 0.01
    val_coef = 0.5

    step_count = 0
    while step_count < total_steps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []
        ep_returns = []
        ep_success = []

        while len(obs_buf) < rollout:
            obs, _ = env.reset()
            cube0 = as_np3(env.unwrapped.cube.pose.p)
            corners = build_corners(cube0[2])
            target_name = list(corners.keys())[np.random.randint(0, 4)]
            target_corner = corners[target_name]
            snap_idx = 0
            num_snaps = max(len(snapshots), 1)

            ep_ret = 0.0
            prev_dist = float(np.linalg.norm(cube0[:2] - target_corner[:2]))
            done = False
            t = 0
            while not done and len(obs_buf) < rollout:
                ob = make_obs(env, target_corner, snap_idx, num_snaps)
                ot = torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    mu, std, v = model(ot)
                    dist = torch.distributions.Normal(mu, std)
                    a = dist.sample()
                    lp = dist.log_prob(a).sum(-1)
                a_np = a.squeeze(0).cpu().numpy().astype(np.float32)

                ee = as_np3(env.unwrapped.agent.tcp.pose.p)
                cube_now = as_np3(env.unwrapped.cube.pose.p)
                delta_model = 0.06 * np.clip(a_np, -1.0, 1.0)
                delta_heur = np.clip(heuristic_delta(ee, cube_now, target_corner), -0.08, 0.08)
                delta_cmd = 0.35 * delta_model + 0.65 * delta_heur
                action = skill_action(delta_cmd)
                _, _, terminated, truncated, _ = env.step(action)
                frame = to_rgb_uint8(env.render())
                feat = visual_feature(env, frame)

                cube = as_np3(env.unwrapped.cube.pose.p)
                dist_to_goal = float(np.linalg.norm(cube[:2] - target_corner[:2]))
                # Required rewards:
                r_base = -dist_to_goal
                r_prog = prev_dist - dist_to_goal
                r_time = -0.005
                r_succ = 1.0 if dist_to_goal < success_radius else 0.0

                # visual snapshot critic reward
                if len(snapshots) > 0:
                    sfeat = snapshots[min(snap_idx, len(snapshots) - 1)]
                    sim = float(np.exp(-6.0 * np.linalg.norm(feat - sfeat)))
                    r_vis = 0.2 * sim
                    if sim > 0.80:
                        snap_idx = min(snap_idx + 1, len(snapshots) - 1)
                else:
                    r_vis = 0.0

                reward = r_base + 0.8 * r_prog + r_time + r_succ + r_vis
                prev_dist = dist_to_goal
                t += 1
                done = (dist_to_goal < success_radius) or (t >= 180) or terminated or truncated

                obs_buf.append(ob)
                act_buf.append(a_np)
                logp_buf.append(float(lp.item()))
                rew_buf.append(float(reward))
                done_buf.append(float(done))
                val_buf.append(float(v.item()))
                ep_ret += reward
                step_count += 1

            ep_returns.append(ep_ret)
            ep_success.append(float(prev_dist < success_radius))

        adv, ret = ppo_gae(np.array(rew_buf, np.float32), np.array(val_buf, np.float32), np.array(done_buf, np.float32))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)
        old_lp_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        idx = np.arange(obs_t.size(0))
        pl_sum, vl_sum, tl_sum, cnt = 0.0, 0.0, 0.0, 0
        for _ in range(ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), mb):
                j = idx[s : s + mb]
                b_obs = obs_t[j]
                b_act = act_t[j]
                b_oldlp = old_lp_t[j]
                b_adv = adv_t[j]
                b_ret = ret_t[j]

                mu, std, v = model(b_obs)
                d = torch.distributions.Normal(mu, std)
                lp = d.log_prob(b_act).sum(-1)
                ratio = torch.exp(lp - b_oldlp)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = F.mse_loss(v, b_ret)
                ent = d.entropy().sum(-1).mean()
                loss = pol_loss + val_coef * val_loss - ent_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                pl_sum += float(pol_loss.item())
                vl_sum += float(val_loss.item())
                tl_sum += float(loss.item())
                cnt += 1

        stats.policy_losses.append(pl_sum / max(cnt, 1))
        stats.value_losses.append(vl_sum / max(cnt, 1))
        stats.total_losses.append(tl_sum / max(cnt, 1))
        stats.mean_returns.append(float(np.mean(ep_returns) if ep_returns else 0.0))
        stats.success_rates.append(float(np.mean(ep_success) if ep_success else 0.0))
        stats.step_counts.append(step_count)
        print(
            f"[train] steps={step_count} total_loss={stats.total_losses[-1]:.4f} "
            f"ret={stats.mean_returns[-1]:.3f} succ={stats.success_rates[-1]:.3f}"
        )

    return model, stats


def draw_target_info(frame: np.ndarray, target_name: str, target_xyz: np.ndarray, dist: float) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, f"Target: {target_name}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"Target xyz: [{target_xyz[0]:.2f}, {target_xyz[1]:.2f}, {target_xyz[2]:.2f}]", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Cube dist to target: {dist:.3f}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def evaluate_and_record(env, model: ActorCritic, snapshots: List[np.ndarray], episodes: int, success_radius: float, out_dir: str, device: torch.device):
    ensure_dir(out_dir)
    results = []
    for ep in range(episodes):
        env.reset(seed=ep + 1000)
        cube0 = as_np3(env.unwrapped.cube.pose.p)
        corners = build_corners(cube0[2])
        target_name = list(corners.keys())[ep % 4]
        target = corners[target_name]
        snap_idx = 0
        frames = []
        done = False
        t = 0
        while not done and t < 200:
            ob = make_obs(env, target, snap_idx, max(len(snapshots), 1))
            ot = torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, _, _ = model(ot)
            a = mu.squeeze(0).cpu().numpy().astype(np.float32)
            ee = as_np3(env.unwrapped.agent.tcp.pose.p)
            cube_now = as_np3(env.unwrapped.cube.pose.p)
            delta_model = 0.06 * np.clip(a, -1.0, 1.0)
            delta_heur = np.clip(heuristic_delta(ee, cube_now, target), -0.08, 0.08)
            delta_cmd = 0.35 * delta_model + 0.65 * delta_heur
            env.step(skill_action(delta_cmd))
            frame = to_rgb_uint8(env.render())
            cube = as_np3(env.unwrapped.cube.pose.p)
            dist = float(np.linalg.norm(cube[:2] - target[:2]))

            if len(snapshots) > 0:
                feat = visual_feature(env, frame)
                sfeat = snapshots[min(snap_idx, len(snapshots) - 1)]
                sim = float(np.exp(-6.0 * np.linalg.norm(feat - sfeat)))
                if sim > 0.80:
                    snap_idx = min(snap_idx + 1, len(snapshots) - 1)

            frames.append(draw_target_info(frame, target_name, target, dist))
            done = dist < success_radius
            t += 1
        success = done
        video_path = os.path.join(out_dir, f"eval_ep_{ep:03d}.mp4")
        imageio.mimsave(video_path, frames, fps=15)
        results.append({"episode": ep, "target": target_name, "success": success, "steps": t, "video": video_path})
    return results


def save_loss_curves(stats: TrainStats, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    x = np.array(stats.step_counts, dtype=np.int32) if len(stats.step_counts) > 0 else np.arange(1, len(stats.total_losses) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(x, stats.total_losses)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Training steps")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].plot(x, stats.policy_losses)
    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Training steps")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].plot(x, stats.value_losses)
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Training steps")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 1].plot(x, stats.success_rates, label="train success")
    axes[1, 1].plot(x, stats.mean_returns, label="mean return")
    axes[1, 1].legend()
    axes[1, 1].set_title("Success/Return")
    axes[1, 1].set_xlabel("Training steps")
    axes[1, 1].set_ylabel("Value")
    for a in axes.reshape(-1):
        a.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps", type=int, default=80000)
    parser.add_argument("--success_radius", type=float, default=0.08)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/visual_move_to")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/move_to_visual_actor.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    import mani_skill.envs  # noqa: F401

    env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array")

    print("building visual snapshot library ...")
    snapshots, demos = build_snapshot_library(env, attempts=120)
    print(f"snapshot_count={len(snapshots)}")
    print(f"demo_samples={len(demos)}")

    print("training move_to visual actor ...")
    warm_model = ActorCritic(obs_dim=10, act_dim=3).to(device)
    pretrain_actor_with_demos(warm_model, demos=demos, device=device, epochs=10, batch_size=256)
    model, stats = train_actor(
        env=env,
        snapshots=snapshots,
        total_steps=args.total_steps,
        device=device,
        success_radius=args.success_radius,
        init_model=warm_model,
    )

    ensure_dir(os.path.dirname(args.checkpoint_path))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "snapshots": [s.tolist() for s in snapshots],
            "obs_dim": 10,
            "act_dim": 3,
        },
        args.checkpoint_path,
    )

    ensure_dir(args.output_dir)
    loss_curve = os.path.join(args.output_dir, "loss_curves.png")
    save_loss_curves(stats, loss_curve)

    print("evaluating and recording videos ...")
    video_dir = os.path.join(args.output_dir, "videos")
    results = evaluate_and_record(
        env=env,
        model=model,
        snapshots=snapshots,
        episodes=args.eval_episodes,
        success_radius=args.success_radius,
        out_dir=video_dir,
        device=device,
    )
    env.close()

    success_rate = float(np.mean([r["success"] for r in results])) if results else 0.0
    summary = {
        "success_rate": success_rate,
        "eval_episodes": args.eval_episodes,
        "loss_curve": loss_curve,
        "checkpoint": args.checkpoint_path,
        "results": results,
    }
    summary_path = os.path.join(args.output_dir, "result.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved summary: {summary_path}")
    print(f"success_rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
