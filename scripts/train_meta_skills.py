import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SKILLS = ["move_to", "descend", "ascend", "open_gripper", "close_gripper"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class StepOut:
    obs: np.ndarray
    reward: float
    done: bool
    success: float


class MetaSkillEnv:
    def __init__(self, skill: str, max_steps: int = 60):
        self.skill = skill
        self.max_steps = max_steps
        self.t = 0
        self.ee = np.zeros(3, dtype=np.float32)
        self.grip = 1.0
        self.goal_xyz = np.zeros(3, dtype=np.float32)
        self.goal_grip = 1.0
        self.prev_pos_dist = 0.0
        self.prev_grip_dist = 0.0

    def reset(self) -> np.ndarray:
        self.t = 0
        self.ee = np.random.uniform(low=[-0.25, -0.25, 0.05], high=[0.25, 0.25, 0.35]).astype(np.float32)
        self.grip = float(np.random.uniform(0.0, 1.0))

        if self.skill == "move_to":
            delta = np.random.uniform(low=[-0.12, -0.12, -0.10], high=[0.12, 0.12, 0.10]).astype(np.float32)
            self.goal_xyz = (self.ee + delta).astype(np.float32)
            self.goal_xyz = np.clip(self.goal_xyz, [-0.22, -0.22, 0.05], [0.22, 0.22, 0.30]).astype(np.float32)
            self.goal_grip = self.grip
        elif self.skill == "descend":
            dz = np.random.uniform(0.04, 0.10)
            self.goal_xyz = self.ee.copy()
            self.goal_xyz[2] = max(0.03, self.ee[2] - dz)
            self.goal_grip = self.grip
        elif self.skill == "ascend":
            dz = np.random.uniform(0.04, 0.10)
            self.goal_xyz = self.ee.copy()
            self.goal_xyz[2] = min(0.35, self.ee[2] + dz)
            self.goal_grip = self.grip
        elif self.skill == "open_gripper":
            self.goal_xyz = self.ee.copy()
            self.goal_grip = 1.0
        elif self.skill == "close_gripper":
            self.goal_xyz = self.ee.copy()
            self.goal_grip = 0.0
        self.prev_pos_dist = float(np.linalg.norm(self.ee - self.goal_xyz))
        self.prev_grip_dist = float(abs(self.grip - self.goal_grip))
        return self._obs()

    def _obs(self) -> np.ndarray:
        tfrac = np.array([self.t / self.max_steps], dtype=np.float32)
        return np.concatenate([self.ee, np.array([self.grip], dtype=np.float32), self.goal_xyz, np.array([self.goal_grip], dtype=np.float32), tfrac], axis=0).astype(np.float32)

    def _success(self) -> bool:
        pos_ok = np.linalg.norm(self.ee - self.goal_xyz) < 0.06
        grip_ok = abs(self.grip - self.goal_grip) < 0.05
        if self.skill in ["move_to", "descend", "ascend"]:
            return pos_ok
        return grip_ok

    def step(self, action: np.ndarray) -> StepOut:
        self.t += 1
        a = np.clip(action, -1.0, 1.0)
        self.ee += 0.06 * a[:3]
        self.ee = np.clip(self.ee, [-0.30, -0.30, 0.02], [0.30, 0.30, 0.40]).astype(np.float32)
        self.grip = float(np.clip(self.grip + 0.08 * a[3], 0.0, 1.0))

        dist_pos = float(np.linalg.norm(self.ee - self.goal_xyz))
        dist_grip = float(abs(self.grip - self.goal_grip))
        prog_pos = self.prev_pos_dist - dist_pos
        prog_grip = self.prev_grip_dist - dist_grip

        if self.skill in ["move_to", "descend", "ascend"]:
            reward = -dist_pos + 2.0 * prog_pos - 0.005
        else:
            reward = -dist_grip + 2.0 * prog_grip - 0.005

        succ = self._success()
        if succ:
            reward += 1.0
        done = succ or (self.t >= self.max_steps)
        self.prev_pos_dist = dist_pos
        self.prev_grip_dist = dist_grip
        return StepOut(obs=self._obs(), reward=float(reward), done=done, success=float(succ))


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = 9, act_dim: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, act_dim)
        self.value = nn.Linear(128, 1)
        self.term = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = torch.tanh(self.mu(h))
        v = self.value(h).squeeze(-1)
        term_prob = torch.sigmoid(self.term(h)).squeeze(-1)
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)
        return mu, std, v, term_prob


def gae(rews, vals, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rews, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rews))):
        next_v = 0.0 if t == len(rews) - 1 else vals[t + 1]
        mask = 1.0 - dones[t]
        delta = rews[t] + gamma * next_v * mask - vals[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    ret = adv + vals
    return adv, ret


def train_one_skill(skill: str, steps: int, device: torch.device, save_dir: str) -> Dict:
    env = MetaSkillEnv(skill=skill, max_steps=40)
    model = PolicyNet(obs_dim=9, act_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch_size = 2048
    ppo_epochs = 6
    minibatch = 256
    clip_eps = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    term_coef = 0.2
    total_steps = 0
    success_ema = 0.0

    while total_steps < steps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf, term_tgt_buf = [], [], [], []
        o = env.reset()
        for _ in range(batch_size):
            ot = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, std, v, term = model(ot)
                dist = torch.distributions.Normal(mu, std)
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
            a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
            out = env.step(a_np)

            obs_buf.append(o)
            act_buf.append(a_np)
            logp_buf.append(float(logp.item()))
            rew_buf.append(out.reward)
            done_buf.append(float(out.done))
            val_buf.append(float(v.item()))
            term_tgt_buf.append(out.success)
            success_ema = 0.98 * success_ema + 0.02 * out.success

            total_steps += 1
            o = out.obs if not out.done else env.reset()

        adv, ret = gae(np.array(rew_buf, dtype=np.float32), np.array(val_buf, dtype=np.float32), np.array(done_buf, dtype=np.float32))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)
        old_logp_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        term_tgt_t = torch.tensor(np.array(term_tgt_buf), dtype=torch.float32, device=device)

        n = obs_t.size(0)
        idx = np.arange(n)
        for _ in range(ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, minibatch):
                j = idx[s : s + minibatch]
                b_obs = obs_t[j]
                b_act = act_t[j]
                b_oldlog = old_logp_t[j]
                b_adv = adv_t[j]
                b_ret = ret_t[j]
                b_term_tgt = term_tgt_t[j]

                mu, std, v, term_prob = model(b_obs)
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(b_act).sum(-1)
                ratio = torch.exp(logp - b_oldlog)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(v, b_ret)
                entropy = dist.entropy().sum(-1).mean()
                term_loss = F.binary_cross_entropy(term_prob, b_term_tgt)
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + term_coef * term_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if total_steps % (batch_size * 4) == 0:
            print(f"[{skill}] steps={total_steps} success_ema={success_ema:.3f}")

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{skill}.pt")
    torch.save({"state_dict": model.state_dict(), "skill": skill, "obs_dim": 9, "act_dim": 4}, path)
    return {"skill": skill, "checkpoint": path, "success_ema": float(success_ema), "steps": total_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="checkpoints/meta_skills")
    parser.add_argument("--steps_per_skill", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--summary_path", type=str, default="outputs/meta_skills/train_summary.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Training meta-skills on device={device}")

    results = []
    for skill in SKILLS:
        print(f"=== train skill: {skill} ===")
        out = train_one_skill(skill, steps=args.steps_per_skill, device=device, save_dir=args.save_dir)
        results.append(out)
        print(f"done {skill}: {out}")

    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"saved summary: {args.summary_path}")


if __name__ == "__main__":
    main()
