#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import gymnasium as gym
import mani_skill.envs  # noqa: F401

import envs.ur3e_reach  # noqa: F401

from scripts_sim2real.maniskill_backend import (
    ManiSkillBackend,
)

from scripts_sim2real.ur3e_ros2_backend import (
    UR3eROS2Backend,
)


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def quaternion_angle_error_deg(
    q1_xyzw,
    q2_xyzw,
):
    q1 = np.asarray(
        q1_xyzw,
        dtype=np.float64,
    )

    q2 = np.asarray(
        q2_xyzw,
        dtype=np.float64,
    )

    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    dot = abs(
        np.dot(q1, q2)
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0,
    )

    angle_rad = (
        2.0
        * np.arccos(dot)
    )

    return float(
        np.degrees(angle_rad)
    )


def set_maniskill_joint_positions(
    robot,
    joint_positions,
):
    active_joints = (
        robot.get_active_joints()
    )

    qpos = robot.get_qpos()

    if hasattr(qpos, "clone"):
        target_qpos = qpos.clone()
    else:
        target_qpos = (
            np.asarray(qpos).copy()
        )

    batched = (
        target_qpos.ndim == 2
    )

    for i, joint in enumerate(
        active_joints
    ):
        name = joint.name

        if name not in joint_positions:
            continue

        value = joint_positions[name]

        if batched:
            target_qpos[0, i] = value
        else:
            target_qpos[i] = value

    robot.set_qpos(
        target_qpos
    )

    return target_qpos


def main():

    # ========================================================
    # 1. 実機 Backend
    # ========================================================

    real_backend = UR3eROS2Backend()

    print(
        "=== ROS2 Bridge ==="
    )

    print(
        real_backend.ping()
    )

    real_joint_positions = (
        real_backend.get_joint_positions()
    )

    real_tcp_pose = (
        real_backend.get_tcp_pose()
    )


    # ========================================================
    # 2. ManiSkill 環境
    # ========================================================

    env = gym.make(
        "UR3eReach-v0",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )

    env.reset()

    sim_backend = (
        ManiSkillBackend(env)
    )

    robot = (
        env.unwrapped.agent.robot
    )


    # ========================================================
    # 3. 実機と同じ関節姿勢を ManiSkill に設定
    # ========================================================

    target_qpos = (
        set_maniskill_joint_positions(
            robot,
            real_joint_positions,
        )
    )

    print(
        "\n=== Target ManiSkill qpos ==="
    )

    print(target_qpos)


    # ========================================================
    # 4. Sim / Real TCP取得
    # ========================================================

    sim_tcp_pose = (
        sim_backend.get_tcp_pose()
    )

    # 実機は最初の取得から少し時間が経っているので
    # TCPをもう一度取得する
    real_tcp_pose = (
        real_backend.get_tcp_pose()
    )


    # ========================================================
    # 5. 誤差計算
    # ========================================================

    p_sim = np.asarray(
        sim_tcp_pose.position,
        dtype=np.float64,
    )

    p_real = np.asarray(
        real_tcp_pose.position,
        dtype=np.float64,
    )

    position_error_vector = (
        p_sim - p_real
    )

    position_error_m = (
        np.linalg.norm(
            position_error_vector
        )
    )

    orientation_error_deg = (
        quaternion_angle_error_deg(
            sim_tcp_pose.orientation_xyzw,
            real_tcp_pose.orientation_xyzw,
        )
    )


    # ========================================================
    # 6. 出力
    # ========================================================

    print(
        "\n=== Real Joint Positions ==="
    )

    for name in JOINT_NAMES:
        print(
            f"{name}: "
            f"{real_joint_positions[name]:.9f}"
        )


    print(
        "\n=== ManiSkill TCP ==="
    )

    print(sim_tcp_pose)


    print(
        "\n=== Real UR3e TCP ==="
    )

    print(real_tcp_pose)


    print(
        "\n=== Sim2Real Error ==="
    )

    print(
        "position error vector [m]:",
        position_error_vector.tolist(),
    )

    print(
        "position error [mm]:",
        position_error_m * 1000.0,
    )

    print(
        "orientation error [deg]:",
        orientation_error_deg,
    )


    # ========================================================
    # 7. PASS / FAIL
    # ========================================================

    position_pass = (
        position_error_m
        < 0.005
    )

    orientation_pass = (
        orientation_error_deg
        < 1.0
    )

    passed = (
        position_pass
        and orientation_pass
    )

    print(
        "\n=== Validation Result ==="
    )

    print(
        "position:",
        "PASS"
        if position_pass
        else "FAIL",
    )

    print(
        "orientation:",
        "PASS"
        if orientation_pass
        else "FAIL",
    )

    print(
        "overall:",
        "PASS"
        if passed
        else "FAIL",
    )


    sim_backend.close()
    real_backend.close()


if __name__ == "__main__":
    main()