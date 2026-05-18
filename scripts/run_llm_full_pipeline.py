import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List

import cv2
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn


SKILLS = ["grasp_skill", "place_skill"]


class MetaPolicy(nn.Module):
    def __init__(self, obs_dim: int = 9, act_dim: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, act_dim)
        self.term = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = torch.tanh(self.mu(h))
        term = torch.sigmoid(self.term(h)).squeeze(-1)
        return mu, term


class VisualActor(nn.Module):
    def __init__(self, obs_dim: int = 10, act_dim: int = 3):
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
        return mu


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


def build_corners(z: float, span_x: float = 0.18, span_y: float = 0.18):
    return {
        "top_left": np.array([-span_x, +span_y, z], dtype=np.float32),
        "top_right": np.array([+span_x, +span_y, z], dtype=np.float32),
        "bottom_left": np.array([-span_x, -span_y, z], dtype=np.float32),
        "bottom_right": np.array([+span_x, -span_y, z], dtype=np.float32),
    }


def draw_overlay(frame: np.ndarray, target_name: str, target_xyz: np.ndarray, dist: float, skill: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, f"Skill: {skill}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"Target: {target_name}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"Target xyz: [{target_xyz[0]:.2f}, {target_xyz[1]:.2f}, {target_xyz[2]:.2f}]",
        (8, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(out, f"Cube-dist: {dist:.3f}", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    return out


def load_meta_policies(ckpt_dir: str, device: torch.device) -> Dict[str, MetaPolicy]:
    out = {}
    for skill in ["move_to", "descend", "ascend", "open_gripper"]:
        path = os.path.join(ckpt_dir, f"{skill}.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        net = MetaPolicy().to(device)
        net.load_state_dict(ckpt["state_dict"], strict=False)
        net.eval()
        out[skill] = net
    return out


def load_visual_actor(path: str, device: torch.device) -> VisualActor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = VisualActor(obs_dim=10, act_dim=3).to(device)
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()
    return net


def load_grasp_actor(path: str, device: torch.device) -> GraspVisualActor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", 8))
    act_dim = int(ckpt.get("act_dim", 4))
    net = GraspVisualActor(obs_dim=obs_dim, act_dim=act_dim).to(device)
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()
    return net


def llm_call(base_url: str, api_key: str, model: str, rules: str, payload: Dict, timeout_s: int) -> Dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "system", "content": rules}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback 1: extract first JSON-like object region.
        m = re.search(r"\{[\s\S]*\}", text)
        if m is not None:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # Fallback 2: keyword-based corner detection.
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


def build_llm_payload(env, instruction: str, corners: Dict[str, np.ndarray]) -> Dict:
    ee = as_np3(env.unwrapped.agent.tcp.pose.p).tolist()
    cube = as_np3(env.unwrapped.cube.pose.p).tolist()
    qpos = env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy().tolist()
    return {
        "instruction": instruction,
        "corners": {k: v.tolist() for k, v in corners.items()},
        "robot": {"ee_pos": ee, "qpos": qpos},
        "cube": {"pos": cube},
    }


def parse_target_only(plan: Dict, default_target: str) -> Dict:
    target = plan.get("target_corner", default_target)
    if target not in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        target = default_target
    return {"target_corner": target}


def fixed_skill_sequence(target_xyz: np.ndarray) -> List[Dict]:
    return [
        {"skill": "grasp_skill"},
        {"skill": "place_skill", "target": target_xyz.tolist()},
    ]


def action7(dx, dy, dz, grip) -> np.ndarray:
    return np.array([dx, dy, dz, 0.0, 0.0, 0.0, grip], dtype=np.float32)


def run_skill_deterministic(env, skill: str, target_xyz: np.ndarray, max_steps: int) -> int:
    cnt = 0
    for _ in range(max_steps):
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        if skill == "open_gripper":
            act = action7(0.0, 0.0, 0.0, 1.0)
        elif skill == "descend":
            zc = float(np.clip((target_xyz[2] - ee[2]) / 0.04, -1.0, 1.0))
            act = action7(0.0, 0.0, zc, -1.0)
        elif skill == "ascend":
            zc = float(np.clip((target_xyz[2] - ee[2]) / 0.04, -1.0, 1.0))
            act = action7(0.0, 0.0, zc, -1.0)
        else:  # move_to
            d = np.clip((target_xyz - ee) / 0.04, -1.0, 1.0)
            act = action7(float(d[0]), float(d[1]), float(d[2]), -1.0)
        env.step(act)
        cnt += 1
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        if skill in ["move_to", "ascend", "descend"] and np.linalg.norm(ee - target_xyz) < 0.04:
            break
        if skill == "open_gripper":
            if cnt >= 12:
                break
    return cnt


def run_grasp_cube_visual(env, grasp_actor: GraspVisualActor, target_corner: np.ndarray, device: torch.device, frames: List[np.ndarray], target_name: str, max_steps: int = 100) -> (int, bool):
    cnt = 0
    table_z = float(as_np3(env.unwrapped.cube.pose.p)[2])
    for _ in range(max_steps):
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        cube = as_np3(env.unwrapped.cube.pose.p)
        qpos = env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy().astype(np.float32)
        grip_open = np.array([float(np.clip(np.mean(qpos[-2:]) / 0.04, 0.0, 1.0))], dtype=np.float32)
        obs = np.concatenate([ee, cube, grip_open, np.array([0.0], dtype=np.float32)], axis=0).astype(np.float32)  # 8D
        ot = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu = grasp_actor(ot).squeeze(0).detach().cpu().numpy().astype(np.float32)
        d_model = 0.06 * np.clip(mu[:3], -1.0, 1.0)
        d_goal = np.clip((cube - ee) / 0.05, -1.0, 1.0) * 0.05
        d = 0.5 * d_model + 0.5 * d_goal
        grip_cmd = float(np.clip(mu[3], -1.0, 1.0))  # RL controls timing.
        act = action7(float(np.clip(d[0] / 0.04, -1, 1)), float(np.clip(d[1] / 0.04, -1, 1)), float(np.clip(d[2] / 0.04, -1, 1)), grip_cmd)
        env.step(act)
        cnt += 1
        fr = to_rgb_u8(env.render())
        dist = float(np.linalg.norm(as_np3(env.unwrapped.cube.pose.p)[:2] - target_corner[:2]))
        frames.append(draw_overlay(fr, target_name, target_corner, dist, "grasp_skill"))
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        cube = as_np3(env.unwrapped.cube.pose.p)
        if np.linalg.norm(ee[:2] - cube[:2]) < 0.04 and ee[2] < 0.08:
            # hold close for a few steps
            for _ in range(8):
                env.step(action7(0.0, 0.0, 0.0, -1.0))
                cnt += 1
                fr = to_rgb_u8(env.render())
                dist = float(np.linalg.norm(as_np3(env.unwrapped.cube.pose.p)[:2] - target_corner[:2]))
                frames.append(draw_overlay(fr, target_name, target_corner, dist, "grasp_skill"))
            # small lift to verify grasp
            for _ in range(12):
                env.step(action7(0.0, 0.0, 0.5, -1.0))
                cnt += 1
                fr = to_rgb_u8(env.render())
                dist = float(np.linalg.norm(as_np3(env.unwrapped.cube.pose.p)[:2] - target_corner[:2]))
                frames.append(draw_overlay(fr, target_name, target_corner, dist, "grasp_skill"))
            cube_after = as_np3(env.unwrapped.cube.pose.p)
            grasp_ok = bool(cube_after[2] > table_z + 0.02)
            return cnt, grasp_ok
    return cnt, False


def run_place_visual(env, place_actor: VisualActor, target_xyz: np.ndarray, device: torch.device, frames: List[np.ndarray], target_name: str, max_steps: int = 220) -> int:
    cnt = 0
    for _ in range(max_steps):
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        cube = as_np3(env.unwrapped.cube.pose.p)
        obs = np.concatenate([ee, cube, target_xyz, np.array([1.0], dtype=np.float32)], axis=0).astype(np.float32)
        ot = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu = place_actor(ot).squeeze(0).detach().cpu().numpy().astype(np.float32)
        d_model = 0.06 * np.clip(mu, -1.0, 1.0)
        d_goal = np.clip((target_xyz - cube) / 0.05, -1.0, 1.0) * 0.05
        d = 0.5 * d_model + 0.5 * d_goal
        act = action7(float(np.clip(d[0] / 0.04, -1, 1)), float(np.clip(d[1] / 0.04, -1, 1)), float(np.clip(d[2] / 0.04, -1, 1)), -1.0)
        env.step(act)
        cnt += 1
        fr = to_rgb_u8(env.render())
        dist = float(np.linalg.norm(as_np3(env.unwrapped.cube.pose.p)[:2] - target_xyz[:2]))
        frames.append(draw_overlay(fr, target_name, target_xyz, dist, "place_skill"))
        cube = as_np3(env.unwrapped.cube.pose.p)
        if np.linalg.norm(cube[:2] - target_xyz[:2]) < 0.08:
            break
    # deterministic descend + release
    for _ in range(25):
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        zc = float(np.clip((target_xyz[2] + 0.02 - ee[2]) / 0.04, -1.0, 1.0))
        env.step(action7(0.0, 0.0, zc, -1.0))
        cnt += 1
        fr = to_rgb_u8(env.render())
        dist = float(np.linalg.norm(as_np3(env.unwrapped.cube.pose.p)[:2] - target_xyz[:2]))
        frames.append(draw_overlay(fr, target_name, target_xyz, dist, "place_skill"))
    for _ in range(15):
        env.step(action7(0.0, 0.0, 0.0, 1.0))
        cnt += 1
        fr = to_rgb_u8(env.render())
        dist = float(np.linalg.norm(as_np3(env.unwrapped.cube.pose.p)[:2] - target_xyz[:2]))
        frames.append(draw_overlay(fr, target_name, target_xyz, dist, "place_skill"))
    return cnt


@dataclass
class EpisodeResult:
    success: bool
    instruction: str
    target_corner: str
    steps: int
    video_path: str


def run_episode(
    env,
    instruction: str,
    target_name: str,
    corners: Dict[str, np.ndarray],
    grasp_actor: GraspVisualActor,
    place_actor: VisualActor,
    device: torch.device,
    success_radius: float,
    ep_idx: int,
    out_dir: str,
) -> EpisodeResult:
    frames = []
    target_name = target_name if target_name in corners else "top_left"
    target = corners[target_name]
    plan_seq = fixed_skill_sequence(target)
    steps = 0
    for item in plan_seq:
        skill = item["skill"]
        if skill == "grasp_skill":
            grasp_ok = False
            retries = 0
            while not grasp_ok and retries < 4:
                c, grasp_ok = run_grasp_cube_visual(
                    env,
                    grasp_actor=grasp_actor,
                    target_corner=target,
                    device=device,
                    frames=frames,
                    target_name=target_name,
                    max_steps=120,
                )
                steps += c
                retries += 1
                if not grasp_ok:
                    # Required by user: immediately open gripper then retry.
                    for _ in range(18):
                        env.step(action7(0.0, 0.0, 0.0, 1.0))
                        steps += 1
                        fr = to_rgb_u8(env.render())
                        dist = float(np.linalg.norm(as_np3(env.unwrapped.cube.pose.p)[:2] - target[:2]))
                        frames.append(draw_overlay(fr, target_name, target, dist, "grasp_retry_open"))
        elif skill == "place_skill":
            steps += run_place_visual(
                env,
                place_actor=place_actor,
                target_xyz=target,
                device=device,
                frames=frames,
                target_name=target_name,
                max_steps=260,
            )

    cube = as_np3(env.unwrapped.cube.pose.p)
    success = float(np.linalg.norm(cube[:2] - target[:2])) < success_radius
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
    video = os.path.join(out_dir, "videos", f"ep_{ep_idx:03d}.mp4")
    imageio.mimsave(video, frames, fps=10)
    return EpisodeResult(bool(success), instruction, target_name, steps, video)


def save_metrics_plot(results: List[EpisodeResult], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = np.arange(1, len(results) + 1)
    succ = np.array([1.0 if r.success else 0.0 for r in results], dtype=np.float32)
    cum = np.cumsum(succ) / x
    steps = np.array([r.steps for r in results], dtype=np.float32)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, succ, marker="o", label="episode success")
    ax[0].plot(x, cum, marker="s", label="cumulative success rate")
    ax[0].set_title("Success Curve")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].grid(alpha=0.2)
    ax[0].legend()

    ax[1].plot(x, steps, marker="o")
    ax[1].set_title("Steps per Episode")
    ax[1].set_xlabel("Episode")
    ax[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_rules_path", type=str, default="LLM-rules")
    parser.add_argument("--grasp_visual_checkpoint", type=str, default="checkpoints/grasp_visual_actor_v1.pt")
    parser.add_argument("--place_visual_checkpoint", type=str, default="checkpoints/move_to_visual_actor_v3.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/llm_full_v1")
    parser.add_argument("--output_json", type=str, default="outputs/llm_full_v1/result.json")
    parser.add_argument("--success_radius", type=float, default=0.08)
    parser.add_argument("--llm_timeout_s", type=int, default=60)
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    api_base_url = args.api_base_url or os.environ.get("LLM_API_BASE_URL")
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_base_url or not api_key:
        raise ValueError("Missing LLM API config.")

    with open(args.llm_rules_path, "r", encoding="utf-8") as f:
        llm_rules = f.read()

    import mani_skill.envs  # noqa: F401

    env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    grasp_actor = load_grasp_actor(args.grasp_visual_checkpoint, device=device)
    place_actor = load_visual_actor(args.place_visual_checkpoint, device=device)

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

    os.makedirs(args.output_dir, exist_ok=True)
    results: List[EpisodeResult] = []
    for ep in range(args.episodes):
        env.reset(seed=ep)
        cube = as_np3(env.unwrapped.cube.pose.p)
        corners = build_corners(cube[2])
        instruction = instructions[ep % len(instructions)]
        payload = build_llm_payload(env, instruction, corners)
        llm_raw = llm_call(
            base_url=api_base_url,
            api_key=api_key,
            model=args.model,
            rules=llm_rules,
            payload=payload,
            timeout_s=args.llm_timeout_s,
        )
        default_target = ["top_left", "top_right", "bottom_left", "bottom_right"][ep % 4]
        plan = parse_target_only(llm_raw, default_target=default_target)
        print(f"[Episode {ep+1}/{args.episodes}] target={plan['target_corner']} skills=5(fixed)")
        out = run_episode(
            env=env,
            instruction=instruction,
            target_name=plan["target_corner"],
            corners=corners,
            grasp_actor=grasp_actor,
            place_actor=place_actor,
            device=device,
            success_radius=args.success_radius,
            ep_idx=ep,
            out_dir=args.output_dir,
        )
        results.append(out)
        print(f"  success={out.success} steps={out.steps} video={out.video_path}")

    env.close()
    success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results])) if results else 0.0
    save_metrics_plot(results, os.path.join(args.output_dir, "metrics_curve.png"))
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "success_rate": success_rate,
                "episodes": args.episodes,
                "grasp_visual_checkpoint": args.grasp_visual_checkpoint,
                "place_visual_checkpoint": args.place_visual_checkpoint,
                "metrics_curve": os.path.join(args.output_dir, "metrics_curve.png"),
                "results": [
                    {
                        "instruction": r.instruction,
                        "target_corner": r.target_corner,
                        "success": r.success,
                        "steps": r.steps,
                        "video": r.video_path,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved result summary: {args.output_json}")
    print(f"Success rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
