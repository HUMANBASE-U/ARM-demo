import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import imageio
import numpy as np
import requests
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io import ensure_dir  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


@dataclass
class EpisodeResult:
    success: bool
    instruction: str
    target_corner: str
    final_cube_pos: List[float]
    steps: int
    video_path: str


def as_np3(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    return x[:3].astype(np.float32)


def build_corners(center: np.ndarray, span_x: float, span_y: float, z: float) -> Dict[str, np.ndarray]:
    return {
        "top_left": np.array([center[0] - span_x, center[1] + span_y, z], dtype=np.float32),
        "top_right": np.array([center[0] + span_x, center[1] + span_y, z], dtype=np.float32),
        "bottom_left": np.array([center[0] - span_x, center[1] - span_y, z], dtype=np.float32),
        "bottom_right": np.array([center[0] + span_x, center[1] - span_y, z], dtype=np.float32),
    }


def call_llm_plan(
    *,
    base_url: str,
    api_key: str,
    model: str,
    rules_text: str,
    payload: Dict,
    timeout_s: int,
) -> Dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": rules_text},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1]
    return json.loads(text)


def normalize_instruction_to_corner(instruction: str) -> str:
    s = instruction.lower()
    if ("left" in s or "zuo" in s or "左" in s) and ("top" in s or "up" in s or "shang" in s or "上" in s):
        return "top_left"
    if ("right" in s or "you" in s or "右" in s) and ("top" in s or "up" in s or "shang" in s or "上" in s):
        return "top_right"
    if ("left" in s or "zuo" in s or "左" in s) and ("bottom" in s or "down" in s or "xia" in s or "下" in s):
        return "bottom_left"
    if ("right" in s or "you" in s or "右" in s) and ("bottom" in s or "down" in s or "xia" in s or "下" in s):
        return "bottom_right"
    # fallback
    return "top_left"


def clamp_plan(plan: Dict, workspace: Dict[str, float], corners: Dict[str, np.ndarray], cube_pos: np.ndarray) -> Dict:
    def clamp_xyz(xyz: List[float]) -> List[float]:
        x, y, z = xyz
        return [
            float(np.clip(x, workspace["x_min"], workspace["x_max"])),
            float(np.clip(y, workspace["y_min"], workspace["y_max"])),
            float(np.clip(z, workspace["z_min"], workspace["z_max"])),
        ]

    target_corner = plan.get("target_corner", "")
    if target_corner not in corners:
        target_corner = normalize_instruction_to_corner(plan.get("reasoning_brief", "") + " " + str(plan))

    pick = plan.get("pick_pos", cube_pos.tolist())
    place = plan.get("place_pos", corners[target_corner].tolist())
    plan_out = {
        "target_corner": target_corner,
        "pick_pos": clamp_xyz(pick),
        "place_pos": clamp_xyz(place),
        "approach_height": float(np.clip(plan.get("approach_height", workspace["z_min"] + 0.12), workspace["z_min"] + 0.05, workspace["z_max"])),
        "grasp_height_offset": float(np.clip(plan.get("grasp_height_offset", 0.01), -0.01, 0.03)),
        "place_height_offset": float(np.clip(plan.get("place_height_offset", 0.015), -0.01, 0.04)),
        "gripper_close": float(np.clip(plan.get("gripper_close", -0.9), -1.0, -0.6)),
        "gripper_open": float(np.clip(plan.get("gripper_open", 0.9), 0.6, 1.0)),
        "reasoning_brief": str(plan.get("reasoning_brief", "clamped plan")),
    }
    return plan_out


def action_from_delta(delta_xyz: np.ndarray, gripper_cmd: float, action_dim: int) -> np.ndarray:
    d = np.clip(delta_xyz, -0.04, 0.04) / 0.04
    if action_dim == 7:
        a = np.array([d[0], d[1], d[2], 0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32)
    elif action_dim == 8:
        a = np.array([d[0], d[1], d[2], 0.0, 0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32)
    else:
        a = np.zeros((action_dim,), dtype=np.float32)
        a[: min(3, action_dim)] = d[: min(3, action_dim)]
        a[-1] = gripper_cmd
    return np.clip(a, -1.0, 1.0)


def move_to(env, target_xyz: np.ndarray, gripper_cmd: float, max_steps: int, frames: List[np.ndarray]) -> int:
    steps = 0
    for _ in range(max_steps):
        ee = as_np3(env.unwrapped.agent.tcp.pose.p)
        delta = target_xyz - ee
        if np.linalg.norm(delta) < 0.01:
            break
        action = action_from_delta(delta, gripper_cmd=gripper_cmd, action_dim=env.action_space.shape[0])
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if frame.ndim == 4:
            frame = frame[0]
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        frames.append(frame[..., :3])
        steps += 1
        if terminated or truncated:
            break
    return steps


def run_episode(env, instruction: str, rules_text: str, base_url: str, api_key: str, model: str, success_radius: float, llm_timeout_s: int) -> EpisodeResult:
    obs, _ = env.reset()
    frames: List[np.ndarray] = []

    cube_actor = env.unwrapped.cube
    cube_pos = as_np3(cube_actor.pose.p)
    ee_pos = as_np3(env.unwrapped.agent.tcp.pose.p)
    table_center = np.array([0.0, 0.0, cube_pos[2]], dtype=np.float32)
    corners = build_corners(table_center, span_x=0.18, span_y=0.18, z=float(cube_pos[2]))
    workspace = {"x_min": -0.30, "x_max": 0.30, "y_min": -0.30, "y_max": 0.30, "z_min": float(cube_pos[2]), "z_max": 0.45}

    llm_input = {
        "instruction": instruction,
        "corners": {k: v.tolist() for k, v in corners.items()},
        "robot": {"ee_pos": ee_pos.tolist(), "gripper_open": True},
        "cube": {"pos": cube_pos.tolist()},
        "workspace": workspace,
    }
    raw_plan = call_llm_plan(
        base_url=base_url,
        api_key=api_key,
        model=model,
        rules_text=rules_text,
        payload=llm_input,
        timeout_s=llm_timeout_s,
    )
    plan = clamp_plan(raw_plan, workspace=workspace, corners=corners, cube_pos=cube_pos)

    target = corners[plan["target_corner"]].copy()
    approach_h = plan["approach_height"]
    pick = np.array(plan["pick_pos"], dtype=np.float32)
    place = np.array(plan["place_pos"], dtype=np.float32)
    pick[2] = cube_pos[2] + plan["grasp_height_offset"]
    place[2] = cube_pos[2] + plan["place_height_offset"]

    steps = 0
    steps += move_to(env, np.array([pick[0], pick[1], approach_h], dtype=np.float32), plan["gripper_open"], 80, frames)
    steps += move_to(env, pick, plan["gripper_open"], 80, frames)
    steps += move_to(env, pick, plan["gripper_close"], 20, frames)
    steps += move_to(env, np.array([pick[0], pick[1], approach_h], dtype=np.float32), plan["gripper_close"], 80, frames)
    steps += move_to(env, np.array([place[0], place[1], approach_h], dtype=np.float32), plan["gripper_close"], 120, frames)
    steps += move_to(env, place, plan["gripper_close"], 80, frames)
    steps += move_to(env, place, plan["gripper_open"], 20, frames)
    steps += move_to(env, np.array([place[0], place[1], approach_h], dtype=np.float32), plan["gripper_open"], 80, frames)

    final_cube = as_np3(cube_actor.pose.p)
    success = float(np.linalg.norm(final_cube[:2] - target[:2])) <= success_radius

    ensure_dir("outputs/llm_corner/videos")
    video_path = os.path.join("outputs/llm_corner/videos", f"episode_{instruction.replace(' ', '_')[:20]}_{np.random.randint(10000)}.mp4")
    imageio.mimsave(video_path, frames, fps=15)
    return EpisodeResult(
        success=bool(success),
        instruction=instruction,
        target_corner=plan["target_corner"],
        final_cube_pos=final_cube.tolist(),
        steps=steps,
        video_path=video_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success_radius", type=float, default=0.06)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--llm_timeout_s", type=int, default=60)
    parser.add_argument("--rules_path", type=str, default="LLM-rules")
    parser.add_argument("--output_json", type=str, default="outputs/llm_corner/result.json")
    args = parser.parse_args()

    set_seed(args.seed)
    api_base_url = args.api_base_url or os.environ.get("LLM_API_BASE_URL")
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_base_url or not api_key:
        raise ValueError("Missing LLM API config. Set --api_base_url/--api_key or env vars LLM_API_BASE_URL/LLM_API_KEY.")

    with open(args.rules_path, "r", encoding="utf-8") as f:
        rules_text = f.read()

    import mani_skill.envs  # noqa: F401

    env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    instructions_bank = [
        "Put the cube in the top left corner",
        "Put the cube in the top right corner",
        "Put the cube in the bottom left corner",
        "Put the cube in the bottom right corner",
        "把cube放到左上角",
        "把cube放到右上角",
        "把cube放到左下角",
        "把cube放到右下角",
    ]

    results: List[EpisodeResult] = []
    for ep in range(args.episodes):
        instruction = instructions_bank[ep % len(instructions_bank)]
        print(f"[Episode {ep+1}/{args.episodes}] instruction={instruction}")
        out = run_episode(
            env=env,
            instruction=instruction,
            rules_text=rules_text,
            base_url=api_base_url,
            api_key=api_key,
            model=args.model,
            success_radius=args.success_radius,
            llm_timeout_s=args.llm_timeout_s,
        )
        results.append(out)
        print(f"  success={out.success} target={out.target_corner} steps={out.steps} video={out.video_path}")

    env.close()

    success_rate = sum(r.success for r in results) / max(len(results), 1)
    ensure_dir(os.path.dirname(args.output_json))
    payload = {
        "episodes": len(results),
        "success_rate": success_rate,
        "results": [
            {
                "success": r.success,
                "instruction": r.instruction,
                "target_corner": r.target_corner,
                "final_cube_pos": r.final_cube_pos,
                "steps": r.steps,
                "video_path": r.video_path,
            }
            for r in results
        ],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved result summary: {args.output_json}")
    print(f"Success rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
