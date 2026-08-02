from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import mani_skill.envs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e_ezgripper  # noqa: F401
import envs.ezgripper_contact_test  # noqa: F401


def run_steps(
    env,
    action,
    steps: int,
    label: str,
) -> None:
    for step in range(steps):
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        env.render()

        if step % 20 == 0:
            print(
                f"{label} step={step:3d}",
                f"f1={info['finger1_force'].item():.3f}",
                f"f2={info['finger2_force'].item():.3f}",
                f"both={info['success'].item()}",
            )


def main() -> None:
    env = gym.make(
        "EZGripperContactTest-v0",
        num_envs=1,
        obs_mode="state",
        reward_mode="sparse",  
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )

    env.reset(seed=0)

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    # UR3eは動かさない
    action[:6] = 0.0

    print("Opening...")
    action[6] = -1.0
    run_steps(
        env,
        action,
        steps=100,
        label="open",
    )

    print("Closing...")
    action[6] = 1.0
    run_steps(
        env,
        action,
        steps=250,
        label="close",
    )

    print("Contact test finished.")

    while True:
        env.step(action)
        env.render()


if __name__ == "__main__":
    main()