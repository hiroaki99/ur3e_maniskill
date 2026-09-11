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


# ============================================================
# Dry-run 設定
# ============================================================

# 最初は base X方向へ +1 mm
TARGET_DELTA_POSITION = np.array(
    [0.001, 0.0, 0.0],
    dtype=np.float64,
)

# 数値Jacobianの差分幅
JACOBIAN_EPS_RAD = 1.0e-4

# Damped Least Squares
DAMPING = 1.0e-3

# dry-run時の関節変位上限
MAX_DQ_RAD = 0.02


# ============================================================
# Quaternion
# ============================================================

def quaternion_xyzw_to_rotation_matrix(q):
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    q = q / np.linalg.norm(q)

    x, y, z, w = q

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_rotvec(R):
    cos_theta = (
        np.trace(R) - 1.0
    ) / 2.0

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    theta = np.arccos(
        cos_theta
    )

    if theta < 1.0e-9:
        return np.zeros(
            3,
            dtype=np.float64,
        )

    sin_theta = np.sin(
        theta
    )

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )

    axis /= (
        2.0 * sin_theta
    )

    return axis * theta


def quaternion_angle_error_deg(
    q1,
    q2,
):
    q1 = np.asarray(
        q1,
        dtype=np.float64,
    )

    q2 = np.asarray(
        q2,
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

    return float(
        np.degrees(
            2.0 * np.arccos(dot)
        )
    )


# ============================================================
# Pose取得
# ============================================================

def get_pose_arrays(backend):
    pose = backend.get_tcp_pose()

    p = np.asarray(
        pose.position,
        dtype=np.float64,
    )

    q = np.asarray(
        pose.orientation_xyzw,
        dtype=np.float64,
    )

    R = (
        quaternion_xyzw_to_rotation_matrix(
            q
        )
    )

    return p, q, R


# ============================================================
# 数値6D Jacobian
# ============================================================

def compute_numerical_jacobian(
    sim_backend,
    q_original,
):
    J = np.zeros(
        (6, 6),
        dtype=np.float64,
    )

    for i, name in enumerate(
        JOINT_NAMES
    ):
        q_plus = (
            q_original.copy()
        )

        q_minus = (
            q_original.copy()
        )

        q_plus[name] += (
            JACOBIAN_EPS_RAD
        )

        q_minus[name] -= (
            JACOBIAN_EPS_RAD
        )


        # --------------------------------------------
        # +eps
        # --------------------------------------------

        sim_backend.command_joint_positions(
            q_plus
        )

        p_plus, _, R_plus = (
            get_pose_arrays(
                sim_backend
            )
        )


        # --------------------------------------------
        # -eps
        # --------------------------------------------

        sim_backend.command_joint_positions(
            q_minus
        )

        p_minus, _, R_minus = (
            get_pose_arrays(
                sim_backend
            )
        )


        # --------------------------------------------
        # Translation Jacobian
        # --------------------------------------------

        J[0:3, i] = (
            p_plus - p_minus
        ) / (
            2.0 * JACOBIAN_EPS_RAD
        )


        # --------------------------------------------
        # Rotation Jacobian
        #
        # minus -> plus の微小回転
        # --------------------------------------------

        R_relative = (
            R_plus
            @ R_minus.T
        )

        rotvec = (
            rotation_matrix_to_rotvec(
                R_relative
            )
        )

        J[3:6, i] = (
            rotvec
            / (
                2.0
                * JACOBIAN_EPS_RAD
            )
        )


    # 元姿勢へ復帰
    sim_backend.command_joint_positions(
        q_original
    )

    return J


# ============================================================
# main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. 実機状態取得
    #
    # READ ONLY
    # 実機へのcommandは送らない
    # --------------------------------------------------------

    real_backend = (
        UR3eROS2Backend()
    )

    print("=== Bridge ===")
    print(real_backend.ping())

    q_real = (
        real_backend.get_joint_positions()
    )

    real_tcp = (
        real_backend.get_tcp_pose()
    )


    print(
        "\n=== Real Joint Positions ==="
    )

    for name in JOINT_NAMES:
        print(
            f"{name}: "
            f"{q_real[name]:.9f} rad"
        )


    print(
        "\n=== Real TCP ==="
    )

    print(real_tcp)


    # --------------------------------------------------------
    # 2. ManiSkill
    # --------------------------------------------------------

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


    # 実機と同じqへ設定
    sim_backend.command_joint_positions(
        q_real
    )

    p0, q0, _ = (
        get_pose_arrays(
            sim_backend
        )
    )


    print(
        "\n=== ManiSkill TCP Before ==="
    )

    print(
        sim_backend.get_tcp_pose()
    )


    # --------------------------------------------------------
    # 3. 数値Jacobian
    # --------------------------------------------------------

    print(
        "\n=== Computing Numerical Jacobian ==="
    )

    J = (
        compute_numerical_jacobian(
            sim_backend,
            q_real,
        )
    )


    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print(
        "\nJacobian:"
    )

    print(J)

    print(
        "\nJacobian condition number:"
    )

    print(
        np.linalg.cond(J)
    )


    # --------------------------------------------------------
    # 4. Desired Cartesian delta
    #
    # positionだけ変更
    # orientation delta = 0
    # --------------------------------------------------------

    desired_twist = np.zeros(
        6,
        dtype=np.float64,
    )

    desired_twist[0:3] = (
        TARGET_DELTA_POSITION
    )


    # --------------------------------------------------------
    # 5. Damped Least Squares
    #
    # dq = J^T (J J^T + lambda^2 I)^-1 dx
    # --------------------------------------------------------

    JJt = (
        J @ J.T
    )

    regularized = (
        JJt
        + (
            DAMPING ** 2
        )
        * np.eye(6)
    )

    dq = (
        J.T
        @ np.linalg.solve(
            regularized,
            desired_twist,
        )
    )


    # --------------------------------------------------------
    # 6. 関節変位上限
    # --------------------------------------------------------

    max_abs_dq = float(
        np.max(
            np.abs(dq)
        )
    )

    scale = 1.0

    if max_abs_dq > MAX_DQ_RAD:
        scale = (
            MAX_DQ_RAD
            / max_abs_dq
        )

        dq *= scale


    print(
        "\n=== IK Joint Delta ==="
    )

    for name, value in zip(
        JOINT_NAMES,
        dq,
    ):
        print(
            f"{name}: "
            f"{value:+.9f} rad"
        )


    print(
        f"\nscale factor: "
        f"{scale:.6f}"
    )

    print(
        "max |dq| [rad]:",
        float(
            np.max(
                np.abs(dq)
            )
        ),
    )


    # --------------------------------------------------------
    # 7. ManiSkillだけにIK結果を適用
    #
    # 実機は動かさない
    # --------------------------------------------------------

    q_target = (
        q_real.copy()
    )

    for name, delta in zip(
        JOINT_NAMES,
        dq,
    ):
        q_target[name] += (
            float(delta)
        )


    sim_backend.command_joint_positions(
        q_target
    )

    p1, q1, _ = (
        get_pose_arrays(
            sim_backend
        )
    )


    actual_delta = (
        p1 - p0
    )

    position_error = (
        actual_delta
        - TARGET_DELTA_POSITION
    )

    position_error_norm = (
        np.linalg.norm(
            position_error
        )
    )

    orientation_error_deg = (
        quaternion_angle_error_deg(
            q0,
            q1,
        )
    )


    # --------------------------------------------------------
    # 8. 結果
    # --------------------------------------------------------

    print(
        "\n=== Cartesian Dry Run Result ==="
    )

    print(
        "requested delta [m]:",
        TARGET_DELTA_POSITION.tolist(),
    )

    print(
        "actual sim delta [m]:",
        actual_delta.tolist(),
    )

    print(
        "position error [mm]:",
        float(
            position_error_norm
            * 1000.0
        ),
    )

    print(
        "orientation change [deg]:",
        orientation_error_deg,
    )


    position_pass = (
        position_error_norm
        < 0.00025
    )

    orientation_pass = (
        orientation_error_deg
        < 0.2
    )

    joint_pass = (
        np.max(
            np.abs(dq)
        )
        <= MAX_DQ_RAD
        + 1.0e-12
    )


    passed = (
        position_pass
        and orientation_pass
        and joint_pass
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
        "joint delta:",
        "PASS"
        if joint_pass
        else "FAIL",
    )

    print(
        "overall:",
        "PASS"
        if passed
        else "FAIL",
    )


    # --------------------------------------------------------
    # 9. Cleanup
    # --------------------------------------------------------

    sim_backend.close()
    real_backend.close()


if __name__ == "__main__":
    main()