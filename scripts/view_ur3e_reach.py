from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e  # noqa: F401
import envs.ur3e_reach  # noqa: F401


def main():
    env = gym.make(
        "UR3eReach-v0",
        num_envs=1,
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )

    env.reset(seed=0)

    while True:
        action = env.action_space.sample()
        env.step(action)
        env.render()


if __name__ == "__main__":
    main()