#!/usr/bin/env python3

import numpy as np

from scripts_sim2real.robot_backend import RobotBackend, TCPPose


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


TCP_OFFSET_TOOL0 = np.array(
    [0.0, 0.0, 0.150],
    dtype=np.float64,
)


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def quaternion_wxyz_to_rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


class ManiSkillBackend(RobotBackend):

    def __init__(self, env):
        self.env = env
        self.base_env = env.unwrapped
        self.robot = self.base_env.agent.robot

        links = self.robot.get_links()
        self.link_map = {
            link.name: link
            for link in links
        }

        if "base" not in self.link_map:
            raise RuntimeError("base link was not found.")

        if "tool0" not in self.link_map:
            raise RuntimeError("tool0 link was not found.")

        self.base_link = self.link_map["base"]
        self.tool0_link = self.link_map["tool0"]

    def get_joint_positions(self):
        qpos = to_numpy(
            self.robot.get_qpos()
        )

        if qpos.ndim == 2:
            qpos = qpos[0]

        active_joints = self.robot.get_active_joints()

        return {
            joint.name: float(qpos[i])
            for i, joint in enumerate(active_joints)
            if joint.name in JOINT_NAMES
        }

    def get_tcp_pose(self):
        T_world_base = self.base_link.pose
        T_world_tool0 = self.tool0_link.pose

        T_base_tool0 = (
            T_world_base.inv()
            * T_world_tool0
        )

        p_tool0 = np.asarray(
            T_base_tool0.p[0].detach().cpu().numpy()
            if T_base_tool0.p.ndim == 2
            else T_base_tool0.p.detach().cpu().numpy(),
            dtype=np.float64,
        )

        q_wxyz = np.asarray(
            T_base_tool0.q[0].detach().cpu().numpy()
            if T_base_tool0.q.ndim == 2
            else T_base_tool0.q.detach().cpu().numpy(),
            dtype=np.float64,
        )

        R = quaternion_wxyz_to_rotation_matrix(
            q_wxyz
        )

        p_tcp = (
            p_tool0
            + R @ TCP_OFFSET_TOOL0
        )

        q_xyzw = [
            float(q_wxyz[1]),
            float(q_wxyz[2]),
            float(q_wxyz[3]),
            float(q_wxyz[0]),
        ]

        return TCPPose(
            frame_id="base",
            position=[
                float(p_tcp[0]),
                float(p_tcp[1]),
                float(p_tcp[2]),
            ],
            orientation_xyzw=q_xyzw,
        )

    def command_joint_positions(
        self,
        joint_positions,
        duration_sec=2.0,
    ):
        qpos = self.robot.get_qpos()

        if hasattr(qpos, "clone"):
            target_qpos = qpos.clone()
        else:
            target_qpos = np.asarray(qpos).copy()

        batched = target_qpos.ndim == 2

        active_joints = self.robot.get_active_joints()

        for i, joint in enumerate(active_joints):
            name = joint.name

            if name not in joint_positions:
                continue

            value = float(joint_positions[name])

            if batched:
                target_qpos[0, i] = value
            else:
                target_qpos[i] = value

        self.robot.set_qpos(target_qpos)

        return {
            "success": True,
            "target_qpos": (
                target_qpos.detach().cpu().numpy().tolist()
                if hasattr(target_qpos, "detach")
                else np.asarray(target_qpos).tolist()
            ),
        }

    def close(self):
        self.env.close()