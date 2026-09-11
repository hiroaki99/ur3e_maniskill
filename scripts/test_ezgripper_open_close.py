from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import mani_skill.envs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e_ezgripper  # noqa: F401


def print_gripper_qpos(env, label: str) -> None:
    qpos = (
        env.unwrapped.agent.robot
        .get_qpos()[0]
        .detach()
        .cpu()
        .numpy()
    )

    print(
        f"{label}: "
        f"joint1={qpos[-2]:.4f}, "
        f"joint2={qpos[-1]:.4f}"
    )


def run_steps(
    env,
    action: np.ndarray,
    steps: int,
) -> None:
    for _ in range(steps):
        env.step(action)
        env.render()


def main() -> None:
    env = gym.make(
        "Empty-v1",
        robot_uids="ur3e_ezgripper",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        render_mode="human",
        num_envs=1,
    )

    env.reset(seed=0)

    print("Action space:", env.action_space)

    if env.action_space.shape[-1] != 7:
        raise RuntimeError(
            "Expected action dimension 7, "
            f"got {env.action_space.shape}"
        )

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    # UR3eの6軸はdelta=0で保持する
    action[..., :6] = 0.0

    print("\nMoving to lower joint limit...")
    action[..., 6] = -1.0
    run_steps(env, action, steps=200)
    print_gripper_qpos(env, "lower endpoint")

    print("\nMoving to upper joint limit...")
    action[..., 6] = 1.0
    run_steps(env, action, steps=200)
    print_gripper_qpos(env, "upper endpoint")

    print("\nMoving to lower joint limit again...")
    action[..., 6] = -1.0
    run_steps(env, action, steps=200)
    print_gripper_qpos(env, "lower endpoint")

    print("\nTest complete.")
    print("Close the GUI window to finish.")

    while True:
        action[..., 6] = -1.0
        env.step(action)
        env.render()


if __name__ == "__main__":
    main()