import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np


def make_action(env, grip):
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    a[-1] = grip
    return a


def main():
    env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    env.reset(seed=0)
    robot = env.unwrapped.agent.robot

    q0 = robot.get_qpos()
    print("qpos_init_tail=", q0[0, -2:].detach().cpu().numpy().tolist())
    for grip in [-1.0, 1.0, -1.0, 1.0]:
        for _ in range(20):
            env.step(make_action(env, grip))
        q = robot.get_qpos()
        print(f"after grip {grip} tail=", q[0, -2:].detach().cpu().numpy().tolist())
    env.close()


if __name__ == "__main__":
    main()
