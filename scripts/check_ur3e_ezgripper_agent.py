from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e_ezgripper  # noqa: F401


def main() -> None:
    env = gym.make(
        "Empty-v1",
        robot_uids="ur3e_ezgripper",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=1,
    )

    obs, info = env.reset(seed=0)

    agent = env.unwrapped.agent

    print("=" * 70)
    print("UR3e + EZGripper Agent Test")
    print("=" * 70)

    print("Control mode:", agent.control_mode)
    print("Action space:", env.action_space)
    print("Observation shape:", obs.shape)
    print("Active joint count:", len(agent.robot.active_joints_map))

    print("\n[Active joints]")
    for index, name in enumerate(
        agent.robot.active_joints_map.keys()
    ):
        print(f"{index}: {name}")

    print("\n[TCP]")
    print("name:", agent.tcp.name)
    print("position:", agent.tcp.pose.p)

    print("\n[Finger links]")
    print("finger1:", agent.finger1_link.name)
    print("finger2:", agent.finger2_link.name)

    assert env.action_space.shape[-1] == 7, (
        "Expected 7-dimensional action space, "
        f"but got {env.action_space.shape}"
    )

    print("\nSUCCESS: action dimension is 7.")

    env.close()


if __name__ == "__main__":
    main()