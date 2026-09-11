#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import envs.ur3e_reach  # noqa: F401

from scripts_sim2real.maniskill_backend import ManiSkillBackend


def main():

    env = gym.make(
        "UR3eReach-v0",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )

    env.reset()

    backend = ManiSkillBackend(env)

    print("=== Joint Positions ===")
    print(
        backend.get_joint_positions()
    )

    print("\n=== TCP Pose ===")
    print(
        backend.get_tcp_pose()
    )

    backend.close()


if __name__ == "__main__":
    main()