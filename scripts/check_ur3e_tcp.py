from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e  # noqa: F401


def main():
    env = gym.make(
        "Empty-v1",
        robot_uids="ur3e_custom",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=1,
    )

    env.reset(seed=0)

    agent = env.unwrapped.agent

    print("TCP link:", agent.tcp.name)
    print("TCP position:", agent.tcp.pose.p)
    print("TCP quaternion:", agent.tcp.pose.q)
    print("qpos:", agent.robot.get_qpos())

    env.close()


if __name__ == "__main__":
    main()