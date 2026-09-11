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
    quaternion = quaternion / torch.linalg.norm(
        quaternion,
        dim=-1,
        keepdim=True,
    )

    w, x, y, z = quaternion.unbind(dim=-1)

    return torch.stack(
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
    ).reshape(-1, 3, 3)


def main() -> None:
    env = gym.make(
        "Empty-v1",
        robot_uids="ur3e_ezgripper",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=1,
    )

    env.reset(seed=0)
    agent = env.unwrapped.agent

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    action[:6] = 0.0

    print(
        f"{'action':>8} "
        f"{'qpos':>10} "
        f"{'separation[m]':>15} "
        f"{'midpoint_x[m]':>15} "
        f"{'tcp_error[m]':>13}"
    )

    for gripper_action in np.linspace(-1.0, 1.0, 17):
        action[6] = gripper_action

        for _ in range(150):
            env.step(action)

        pad1 = agent.finger1_link.pose.p
        pad2 = agent.finger2_link.pose.p
        midpoint = (pad1 + pad2) / 2.0

        palm = agent.robot.links_map[
            "gripper_ezgripper_palm_link"
        ]

        rotation = quaternion_wxyz_to_matrix(
            palm.pose.q
        )

        midpoint_in_palm = torch.matmul(
            rotation.transpose(-1, -2),
            (midpoint - palm.pose.p).unsqueeze(-1),
        ).squeeze(-1)

        separation = torch.linalg.norm(
            pad1 - pad2,
            dim=1,
        )

        tcp_error = torch.linalg.norm(
            agent.tcp.pose.p - midpoint,
            dim=1,
        )

        qpos = agent.robot.get_qpos()[0, -1]

        print(
            f"{gripper_action:8.3f} "
            f"{qpos.item():10.4f} "
            f"{separation.item():15.5f} "
            f"{midpoint_in_palm[0, 0].item():15.5f} "
            f"{tcp_error.item():13.5f}"
        )

    env.close()


if __name__ == "__main__":
    main()