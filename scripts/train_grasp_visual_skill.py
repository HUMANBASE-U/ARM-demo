import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

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


def grip_open_ratio(env) -> float:
    qpos = env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy().astype(np.float32)
    return float(np.clip(np.mean(qpos[-2:]) / 0.04, 0.0, 1.0))


def build_obs(env, snap_idx: int, total_snaps: int) -> np.ndarray:
    ee = as_np3(env.unwrapped.agent.tcp.pose.p)
    cube = as_np3(env.unwrapped.cube.pose.p)
    g = np.array([grip_open_ratio(env)], dtype=np.float32)
    t = np.array([snap_idx / max(total_snaps, 1)], dtype=np.float32)
    return np.concatenate([ee, cube, g, t], axis=0).astype(np.float32)  # 8 dim


def visual_feat(env, frame: np.ndarray) -> np.ndarray:
    uv = red_cube_centroid(frame)
    h, w = frame.shape[:2]
    uv = np.array([uv[0] / w, uv[1] / h], dtype=np.float32)
    ee = as_np3(env.unwrapped.agent.tcp.pose.p)
    g = np.array([grip_open_ratio(env)], dtype=np.float32)
    return np.concatenate([uv, ee, g], axis=0).astype(np.float32)  # 6 dim


def choose_snapshots(features: List[np.ndarray], max_k: int = 3, min_gap: int = 10) -> List[np.ndarray]:
    if len(features) == 0:
        return []
    f = np.stack(features, axis=0)
    n = f.shape[0]
    if n <= max_k:
        return [x.copy() for x in f]
    uv = f[:, :2]
    vel = uv[1:] - uv[:-1]
    score = np.zeros((n,), dtype=np.float32)
    for i in range(2, n):
        a = vel[i - 2]
        b = vel[i - 1]
        na = np.linalg.norm(a) + 1e-6
        nb = np.linalg.norm(b) + 1e-6
        score[i] = 1.0 - float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    cand = np.argsort(-score).tolist()
    idx = []
    for c in cand:
        if c < min_gap or c > n - min_gap:
            continue
        if all(abs(c - p) >= min_gap for p in idx):
            idx.append(c)
        if len(idx) >= max_k:
            break
    if len(idx) == 0:
        idx = [n // 3, (2 * n) // 3]
    idx = sorted(idx)[:max_k]
    return [f[i].copy() for i in idx]


def heuristic_grasp_rollout(env, horizon: int = 120):
    env.reset()
    feats = []
    frames = []
    table_z = float(as_np3(env.unwrapped.cube.pose.p)[2])
    success = False
    for t in range(horizon):
        fr = to_rgb_u8(env.render())
        frames.append(fr)
        feats.append(visual_feat(env, fr))

        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        cube = as_np3(env.unwrapped.cube.pose.p)
        d = cube - ee
        grip_cmd = 1.0 if t < 25 else -1.0
        act = np.array(
            [
                float(np.clip(d[0] / 0.04, -1.0, 1.0)),
                float(np.clip(d[1] / 0.04, -1.0, 1.0)),
                float(np.clip((d[2] + 0.02) / 0.04, -1.0, 1.0)),
                0.0,
                0.0,
                0.0,
                float(np.clip(grip_cmd, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        env.step(act)
        if t > 35:
            for _ in range(8):
                env.step(np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))
            cube2 = as_np3(env.unwrapped.cube.pose.p)
            if cube2[2] > table_z + 0.025:
                success = True
                break
    return {"success": success, "features": feats, "frames": frames}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 8, act_dim: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)
        self.value = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = torch.tanh(self.mu(h))
        v = self.value(h).squeeze(-1)
        std = torch.exp(self.log_std).clamp(1e-3, 0.8)
        return mu, std, v


def gae(rew, val, done, gamma=0.99, lam=0.95):
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
class Stats:
    steps: List[int]
    total_loss: List[float]
    policy_loss: List[float]
    value_loss: List[float]
    entropy_loss: List[float]
    reward_base: List[float]
    reward_progress: List[float]
    reward_time: List[float]
    reward_done: List[float]
    reward_visual: List[float]
    ep_success: List[float]
    ep_return: List[float]


def train(
    env,
    snapshots: List[np.ndarray],
    total_steps: int,
    device: torch.device,
    rollout_steps: int = 2048,
    ppo_epochs: int = 8,
    minibatch_size: int = 256,
) -> Tuple[ActorCritic, Stats]:
    model = ActorCritic(obs_dim=8, act_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = Stats([], [], [], [], [], [], [], [], [], [], [], [])

    rollout = int(rollout_steps)
    ppo_epochs = int(ppo_epochs)
    mb = int(minibatch_size)
    clip_eps = 0.2
    ent_coef = 0.01
    val_coef = 0.5
    step_count = 0

    while step_count < total_steps:
        ob_buf, ac_buf, lp_buf, rw_buf, dn_buf, vl_buf = [], [], [], [], [], []
        rb_buf, rp_buf, rt_buf, rd_buf, rv_buf = [], [], [], [], []
        ep_rets, ep_succ = [], []
        while len(ob_buf) < rollout:
            env.reset()
            table_z = float(as_np3(env.unwrapped.cube.pose.p)[2])
            prev_dist = float(np.linalg.norm(as_np3(env.unwrapped.agent.tcp.pose.p) - as_np3(env.unwrapped.cube.pose.p)))
            snap_idx = 0
            ret = 0.0
            done = False
            t = 0
            while not done and len(ob_buf) < rollout:
                ob = build_obs(env, snap_idx, len(snapshots))
                ot = torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    mu, std, v = model(ot)
                    dist = torch.distributions.Normal(mu, std)
                    a = dist.sample()
                    lp = dist.log_prob(a).sum(-1)
                a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
                ee = as_np3(env.unwrapped.agent.tcp.pose.p)
                cube = as_np3(env.unwrapped.cube.pose.p)
                dxyz = 0.06 * np.clip(a_np[:3], -1.0, 1.0)
                grip = float(np.clip(a_np[3], -1.0, 1.0))
                act = np.array(
                    [
                        float(np.clip(dxyz[0] / 0.04, -1.0, 1.0)),
                        float(np.clip(dxyz[1] / 0.04, -1.0, 1.0)),
                        float(np.clip(dxyz[2] / 0.04, -1.0, 1.0)),
                        0.0,
                        0.0,
                        0.0,
                        grip,
                    ],
                    dtype=np.float32,
                )
                env.step(act)
                fr = to_rgb_u8(env.render())
                feat = visual_feat(env, fr)
                ee2 = as_np3(env.unwrapped.agent.tcp.pose.p)
                cube2 = as_np3(env.unwrapped.cube.pose.p)
                dist_now = float(np.linalg.norm(ee2 - cube2))

                # required reward terms
                r_base = -dist_now
                r_prog = prev_dist - dist_now
                r_time = -0.005
                # success: cube lifted (grasp done)
                lift_ok = float(cube2[2] > table_z + 0.025)
                r_done = 1.0 if lift_ok > 0.5 else 0.0

                # visual snapshot critic reward (includes gripper state)
                if len(snapshots) > 0:
                    sfeat = snapshots[min(snap_idx, len(snapshots) - 1)]
                    sim = float(np.exp(-8.0 * np.linalg.norm(feat - sfeat)))
                    r_vis = 0.25 * sim
                    if sim > 0.82:
                        snap_idx = min(snap_idx + 1, len(snapshots) - 1)
                else:
                    r_vis = 0.0

                reward = r_base + 0.8 * r_prog + r_time + r_done + r_vis
                prev_dist = dist_now
                t += 1
                done = (lift_ok > 0.5) or (t >= 160)

                ob_buf.append(ob)
                ac_buf.append(a_np)
                lp_buf.append(float(lp.item()))
                rw_buf.append(float(reward))
                rb_buf.append(float(r_base))
                rp_buf.append(float(r_prog))
                rt_buf.append(float(r_time))
                rd_buf.append(float(r_done))
                rv_buf.append(float(r_vis))
                dn_buf.append(float(done))
                vl_buf.append(float(v.item()))
                ret += reward
                step_count += 1

            ep_rets.append(ret)
            ep_succ.append(float(done))

        adv, ret = gae(np.array(rw_buf, np.float32), np.array(vl_buf, np.float32), np.array(dn_buf, np.float32))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ob_t = torch.tensor(np.array(ob_buf), dtype=torch.float32, device=device)
        ac_t = torch.tensor(np.array(ac_buf), dtype=torch.float32, device=device)
        lp_t = torch.tensor(np.array(lp_buf), dtype=torch.float32, device=device)
        ad_t = torch.tensor(adv, dtype=torch.float32, device=device)
        rt_t = torch.tensor(ret, dtype=torch.float32, device=device)

        idx = np.arange(ob_t.size(0))
        pl, vl, tl, el, cnt = 0.0, 0.0, 0.0, 0.0, 0
        for _ in range(ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), mb):
                j = idx[s : s + mb]
                b_obs, b_act, b_oldlp = ob_t[j], ac_t[j], lp_t[j]
                b_adv, b_ret = ad_t[j], rt_t[j]
                mu, std, v = model(b_obs)
                d = torch.distributions.Normal(mu, std)
                lp_new = d.log_prob(b_act).sum(-1)
                ratio = torch.exp(lp_new - b_oldlp)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                p_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(v, b_ret)
                ent = d.entropy().sum(-1).mean()
                loss = p_loss + val_coef * v_loss - ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                pl += float(p_loss.item())
                vl += float(v_loss.item())
                tl += float(loss.item())
                el += float(ent.item())
                cnt += 1

        stats.total_loss.append(tl / max(cnt, 1))
        stats.policy_loss.append(pl / max(cnt, 1))
        stats.value_loss.append(vl / max(cnt, 1))
        stats.entropy_loss.append(el / max(cnt, 1))
        stats.reward_base.append(float(np.mean(rb_buf) if rb_buf else 0.0))
        stats.reward_progress.append(float(np.mean(rp_buf) if rp_buf else 0.0))
        stats.reward_time.append(float(np.mean(rt_buf) if rt_buf else 0.0))
        stats.reward_done.append(float(np.mean(rd_buf) if rd_buf else 0.0))
        stats.reward_visual.append(float(np.mean(rv_buf) if rv_buf else 0.0))
        stats.ep_success.append(float(np.mean(ep_succ) if ep_succ else 0.0))
        stats.ep_return.append(float(np.mean(ep_rets) if ep_rets else 0.0))
        stats.steps.append(step_count)
        print(f"[train] steps={step_count} total={stats.total_loss[-1]:.4f} succ={stats.ep_success[-1]:.3f} ret={stats.ep_return[-1]:.3f}")

    return model, stats


def save_curve(stats: Stats, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    x = np.array(stats.steps, dtype=np.int32) if len(stats.steps) > 0 else np.arange(1, len(stats.total_loss) + 1)
    fig, ax = plt.subplots(3, 3, figsize=(13, 10))
    ax[0, 0].plot(x, stats.total_loss); ax[0, 0].set_title("Total Loss"); ax[0, 0].set_xlabel("Training steps"); ax[0, 0].set_ylabel("Loss")
    ax[0, 1].plot(x, stats.policy_loss); ax[0, 1].set_title("Policy Loss"); ax[0, 1].set_xlabel("Training steps"); ax[0, 1].set_ylabel("Loss")
    ax[0, 2].plot(x, stats.value_loss); ax[0, 2].set_title("Value Loss"); ax[0, 2].set_xlabel("Training steps"); ax[0, 2].set_ylabel("Loss")
    ax[1, 0].plot(x, stats.entropy_loss); ax[1, 0].set_title("Entropy"); ax[1, 0].set_xlabel("Training steps"); ax[1, 0].set_ylabel("Entropy")
    ax[1, 1].plot(x, stats.reward_base); ax[1, 1].set_title("Reward Base"); ax[1, 1].set_xlabel("Training steps"); ax[1, 1].set_ylabel("Reward")
    ax[1, 2].plot(x, stats.reward_progress); ax[1, 2].set_title("Reward Progress"); ax[1, 2].set_xlabel("Training steps"); ax[1, 2].set_ylabel("Reward")
    ax[2, 0].plot(x, stats.reward_time, label="time")
    ax[2, 0].plot(x, stats.reward_done, label="done")
    ax[2, 0].plot(x, stats.reward_visual, label="visual")
    ax[2, 0].legend(); ax[2, 0].set_title("Reward Components"); ax[2, 0].set_xlabel("Training steps"); ax[2, 0].set_ylabel("Reward")
    ax[2, 1].plot(x, stats.ep_success, label="success")
    ax[2, 1].plot(x, stats.ep_return, label="return")
    ax[2, 1].legend(); ax[2, 1].set_title("Success/Return"); ax[2, 1].set_xlabel("Training steps"); ax[2, 1].set_ylabel("Value")
    ax[2, 2].axis("off")
    for a in ax.reshape(-1):
        a.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def eval_policy(env, model: ActorCritic, episodes: int, device: torch.device, out_dir: str):
    ensure_dir(out_dir)
    res = []
    for ep in range(episodes):
        env.reset(seed=1000 + ep)
        table_z = float(as_np3(env.unwrapped.cube.pose.p)[2])
        frames = []
        success = False
        for t in range(180):
            ob = build_obs(env, 0, 1)
            ot = torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, _, _ = model(ot)
            a = mu.squeeze(0).cpu().numpy().astype(np.float32)
            dxyz = 0.06 * np.clip(a[:3], -1.0, 1.0)
            grip = float(np.clip(a[3], -1.0, 1.0))
            act = np.array(
                [
                    float(np.clip(dxyz[0] / 0.04, -1.0, 1.0)),
                    float(np.clip(dxyz[1] / 0.04, -1.0, 1.0)),
                    float(np.clip(dxyz[2] / 0.04, -1.0, 1.0)),
                    0.0,
                    0.0,
                    0.0,
                    grip,
                ],
                dtype=np.float32,
            )
            env.step(act)
            fr = to_rgb_u8(env.render())
            cube = as_np3(env.unwrapped.cube.pose.p)
            ee = as_np3(env.unwrapped.agent.tcp.pose.p)
            g = grip_open_ratio(env)
            cv2.putText(fr, f"grip_open={g:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(fr, f"ee-cube={np.linalg.norm(ee-cube):.3f}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(fr, f"cube_z={cube[2]:.3f}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            frames.append(fr)
            if cube[2] > table_z + 0.025:
                success = True
                break
        path = os.path.join(out_dir, f"grasp_eval_{ep:03d}.mp4")
        imageio.mimsave(path, frames, fps=10)
        res.append({"episode": ep, "success": bool(success), "steps": len(frames), "video": path})
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/grasp_visual_v1")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/grasp_visual_actor_v1.pt")
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--ppo_epochs", type=int, default=8)
    parser.add_argument("--minibatch_size", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    import mani_skill.envs  # noqa: F401

    env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    snaps = []
    for _ in range(80):
        out = heuristic_grasp_rollout(env, horizon=140)
        if out["success"]:
            snaps = choose_snapshots(out["features"], max_k=3, min_gap=10)
            if len(snaps) > 0:
                break
    print(f"snapshot_count={len(snaps)}")
    model, stats = train(
        env,
        snapshots=snaps,
        total_steps=args.total_steps,
        device=device,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
    )

    ensure_dir(os.path.dirname(args.checkpoint))
    torch.save({"state_dict": model.state_dict(), "obs_dim": 8, "act_dim": 4, "snapshots": [s.tolist() for s in snaps]}, args.checkpoint)

    ensure_dir(args.output_dir)
    curve = os.path.join(args.output_dir, "loss_curves.png")
    save_curve(stats, curve)
    videos = os.path.join(args.output_dir, "videos")
    eval_res = eval_policy(env, model, episodes=args.eval_episodes, device=device, out_dir=videos)
    env.close()

    succ = float(np.mean([1.0 if r["success"] else 0.0 for r in eval_res])) if eval_res else 0.0
    summary = {
        "success_rate": succ,
        "checkpoint": args.checkpoint,
        "loss_curve": curve,
        "steps": stats.steps,
        "loss_series": {
            "total": stats.total_loss,
            "policy": stats.policy_loss,
            "value": stats.value_loss,
            "entropy": stats.entropy_loss,
            "reward_base": stats.reward_base,
            "reward_progress": stats.reward_progress,
            "reward_time": stats.reward_time,
            "reward_done": stats.reward_done,
            "reward_visual": stats.reward_visual,
            "ep_success": stats.ep_success,
            "ep_return": stats.ep_return,
        },
        "results": eval_res,
    }
    out_json = os.path.join(args.output_dir, "result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary: {out_json}")
    print(f"success_rate: {succ:.3f}")


if __name__ == "__main__":
    main()
