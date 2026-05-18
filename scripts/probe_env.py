import gymnasium as gym
import mani_skill.envs  # noqa: F401


def main():
    env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array")
    print("action_space_shape=", env.action_space.shape)
    print("obs_space=", env.observation_space)
    u = env.unwrapped
    attrs = [a for a in dir(u) if ("cube" in a.lower() or "obj" in a.lower() or "goal" in a.lower()) and not a.startswith("_")]
    print("candidate_attrs=", attrs[:40])
    for name in attrs[:20]:
        try:
            v = getattr(u, name)
            print("attr", name, "type", type(v))
        except Exception as exc:
            print("attr", name, "error", exc)
    env.close()


if __name__ == "__main__":
    main()
