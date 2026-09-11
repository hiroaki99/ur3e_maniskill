from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e_ezgripper  # noqa: F401


def quaternion_wxyz_to_matrix(
    quaternion: torch.Tensor,
) -> torch.Tensor:
    """wxyz順のQuaternionを回転行列へ変換する。"""
    quaternion = quaternion / torch.linalg.norm(
        quaternion,
        dim=-1,
        keepdim=True,
    )

    w, x, y, z = quaternion.unbind(dim=-1)

    matrix = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),

            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),

            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )

    return matrix.reshape(
        quaternion.shape[:-1] + (3, 3)
    )


def settle(
    env,
    action: np.ndarray,
    steps: int = 200,
) -> None:
    for _ in range(steps):
        env.step(action)


def measure(env, label: str) -> dict:
    agent = env.unwrapped.agent

    pad1_position = agent.finger1_link.pose.p
    pad2_position = agent.finger2_link.pose.p

    midpoint = (
        pad1_position + pad2_position
    ) / 2.0

    tcp_position = agent.tcp.pose.p

    palm = agent.robot.links_map[
        "gripper_ezgripper_palm_link"
    ]

    palm_position = palm.pose.p
    palm_quaternion = palm.pose.q

    palm_rotation = quaternion_wxyz_to_matrix(
        palm_quaternion
    )

    midpoint_in_palm = torch.matmul(
        palm_rotation.transpose(-1, -2),
        (midpoint - palm_position).unsqueeze(-1),
    ).squeeze(-1)

    pad_distance = torch.linalg.norm(
        pad1_position - pad2_position,
        dim=-1,
    )

    tcp_error = torch.linalg.norm(
        midpoint - tcp_position,
        dim=-1,
    )

    qpos = agent.robot.get_qpos()

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    print("gripper qpos:")
    print(qpos[:, -2:])

    print("finger pad 1 position:")
    print(pad1_position)

    print("finger pad 2 position:")
    print(pad2_position)

    print("finger-pad separation [m]:")
    print(pad_distance)

    print("finger midpoint, world frame:")
    print(midpoint)

    print("finger midpoint, palm frame:")
    print(midpoint_in_palm)

    print("grasp_tcp position:")
    print(tcp_position)

    print("TCP-to-midpoint error [m]:")
    print(tcp_error)

    return {
        "separation": pad_distance.item(),
        "midpoint_in_palm": (
            midpoint_in_palm[0]
            .detach()
            .cpu()
            .numpy()
        ),
        "tcp_error": tcp_error.item(),
    }


def main() -> None:
    env = gym.make(
        "Empty-v1",
        robot_uids="ur3e_ezgripper",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=1,
    )

    env.reset(seed=0)

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    # アームは停止
    action[:6] = 0.0

    action[6] = -1.0
    settle(env, action)
    lower_result = measure(
        env,
        "action[6] = -1.0 / lower endpoint",
    )

    action[6] = 1.0
    settle(env, action)
    upper_result = measure(
        env,
        "action[6] = +1.0 / upper endpoint",
    )

    print()
    print("=" * 70)
    print("Result")
    print("=" * 70)

    if (
        lower_result["separation"]
        > upper_result["separation"]
    ):
        print("OPEN action : -1.0")
        print("CLOSE action: +1.0")
    else:
        print("OPEN action : +1.0")
        print("CLOSE action: -1.0")

    print(
        "lower separation:",
        lower_result["separation"],
    )
    print(
        "upper separation:",
        upper_result["separation"],
    )

    env.close()


if __name__ == "__main__":
    main()