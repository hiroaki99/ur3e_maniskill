#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict

import numpy as np


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@dataclass
class CartesianDeltaPlan:
    q_current: Dict[str, float]
    q_target: Dict[str, float]

    dq: np.ndarray

    requested_delta_m: np.ndarray
    predicted_delta_m: np.ndarray

    predicted_position_error_m: float
    predicted_orientation_change_deg: float

    jacobian: np.ndarray
    condition_number: float


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

    return (
        axis * theta
    )


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


class CartesianDeltaController:
    """
    Cartesian TCP delta -> joint target converter.

    kinematics_backend:
        ManiSkillBackend を想定。
        FK / numerical Jacobian計算に使用する。

    robot_backend:
        実際に状態取得・joint commandを行うBackend。
        ManiSkillBackend または UR3eROS2Backend。
    """

    def __init__(
        self,
        kinematics_backend,
        robot_backend,
        jacobian_eps_rad=1.0e-3,
        damping=1.0e-3,
        max_joint_delta_rad=0.01,
        max_cartesian_delta_m=0.005,
        max_condition_number=200.0,
        command_duration_sec=0.5,
    ):
        self.kinematics_backend = (
            kinematics_backend
        )

        self.robot_backend = (
            robot_backend
        )

        self.jacobian_eps_rad = float(
            jacobian_eps_rad
        )

        self.damping = float(
            damping
        )

        self.max_joint_delta_rad = float(
            max_joint_delta_rad
        )

        self.max_cartesian_delta_m = float(
            max_cartesian_delta_m
        )

        self.max_condition_number = float(
            max_condition_number
        )

        self.command_duration_sec = float(
            command_duration_sec
        )


    def _get_pose_arrays(
        self,
        backend,
    ):
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


    def _compute_numerical_jacobian(
        self,
        q_original,
    ):
        J = np.zeros(
            (6, 6),
            dtype=np.float64,
        )

        eps = (
            self.jacobian_eps_rad
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

            q_plus[name] += eps
            q_minus[name] -= eps


            self.kinematics_backend.command_joint_positions(
                q_plus
            )

            p_plus, _, R_plus = (
                self._get_pose_arrays(
                    self.kinematics_backend
                )
            )


            self.kinematics_backend.command_joint_positions(
                q_minus
            )

            p_minus, _, R_minus = (
                self._get_pose_arrays(
                    self.kinematics_backend
                )
            )


            J[0:3, i] = (
                p_plus - p_minus
            ) / (
                2.0 * eps
            )


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
                    2.0 * eps
                )
            )


        self.kinematics_backend.command_joint_positions(
            q_original
        )

        return J


    def plan_tcp_delta(
        self,
        delta_position_m,
    ):
        """
        実機は動かさず、
        Cartesian deltaに必要なjoint targetを計算する。

        delta_position_m:
            [dx, dy, dz] in base frame [m]
        """

        delta_position_m = np.asarray(
            delta_position_m,
            dtype=np.float64,
        )

        if delta_position_m.shape != (3,):
            raise ValueError(
                "delta_position_m must have shape (3,)."
            )


        delta_norm = float(
            np.linalg.norm(
                delta_position_m
            )
        )

        if (
            delta_norm
            > self.max_cartesian_delta_m
        ):
            raise RuntimeError(
                "Cartesian safety limit exceeded: "
                f"{delta_norm:.6f} m > "
                f"{self.max_cartesian_delta_m:.6f} m"
            )


        # ----------------------------------------------------
        # 現在状態
        # ----------------------------------------------------

        q_current = (
            self.robot_backend
            .get_joint_positions()
        )


        # shadow ManiSkill modelを
        # 実機と同じqへ合わせる
        self.kinematics_backend.command_joint_positions(
            q_current
        )


        p_before, quat_before, _ = (
            self._get_pose_arrays(
                self.kinematics_backend
            )
        )


        # ----------------------------------------------------
        # Jacobian
        # ----------------------------------------------------

        J = (
            self._compute_numerical_jacobian(
                q_current
            )
        )


        condition_number = float(
            np.linalg.cond(J)
        )


        # ----------------------------------------------------
        # Jacobian diagnostics
        # ----------------------------------------------------

        singular_values = np.linalg.svd(
            J,
            compute_uv=False,
        )

        print(
            "\n=== CartesianDeltaController Jacobian ==="
        )

        np.set_printoptions(
            precision=6,
            suppress=True,
        )

        print(J)

        print(
            "singular values:",
            singular_values,
        )

        print(
            "condition number:",
            condition_number,
        )

        print(
            "jacobian eps [rad]:",
            self.jacobian_eps_rad,
        )


        if (
            not np.isfinite(
                condition_number
            )
            or condition_number
            > self.max_condition_number
        ):
            raise RuntimeError(
                "Jacobian condition number "
                "exceeded safety limit: "
                f"{condition_number:.3f} > "
                f"{self.max_condition_number:.3f}"
            )


        # ----------------------------------------------------
        # Desired Cartesian change
        #
        # TCP orientationは維持する。
        # ----------------------------------------------------

        desired_twist = np.zeros(
            6,
            dtype=np.float64,
        )

        desired_twist[0:3] = (
            delta_position_m
        )


        # ----------------------------------------------------
        # Damped Least Squares
        #
        # dq =
        # J^T (J J^T + lambda^2 I)^-1 dx
        # ----------------------------------------------------

        regularized = (
            J @ J.T
            + (
                self.damping ** 2
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


        max_abs_dq = float(
            np.max(
                np.abs(dq)
            )
        )


        if (
            max_abs_dq
            > self.max_joint_delta_rad
        ):
            raise RuntimeError(
                "IK joint delta exceeded "
                "safety limit: "
                f"{max_abs_dq:.6f} rad > "
                f"{self.max_joint_delta_rad:.6f} rad"
            )


        # ----------------------------------------------------
        # target q
        # ----------------------------------------------------

        q_target = (
            q_current.copy()
        )

        for name, delta in zip(
            JOINT_NAMES,
            dq,
        ):
            q_target[name] += float(
                delta
            )


        # ----------------------------------------------------
        # ManiSkill上で予測
        # ----------------------------------------------------

        self.kinematics_backend.command_joint_positions(
            q_target
        )


        p_after, quat_after, _ = (
            self._get_pose_arrays(
                self.kinematics_backend
            )
        )


        predicted_delta = (
            p_after - p_before
        )


        predicted_error_m = float(
            np.linalg.norm(
                predicted_delta
                - delta_position_m
            )
        )


        orientation_change_deg = (
            quaternion_angle_error_deg(
                quat_before,
                quat_after,
            )
        )


        # kinematics modelを元姿勢へ戻す
        self.kinematics_backend.command_joint_positions(
            q_current
        )


        return CartesianDeltaPlan(
            q_current=q_current,
            q_target=q_target,
            dq=dq,
            requested_delta_m=(
                delta_position_m.copy()
            ),
            predicted_delta_m=(
                predicted_delta
            ),
            predicted_position_error_m=(
                predicted_error_m
            ),
            predicted_orientation_change_deg=(
                orientation_change_deg
            ),
            jacobian=J,
            condition_number=(
                condition_number
            ),
        )


    def command_tcp_delta(
        self,
        delta_position_m,
        execute=True,
    ):
        """
        Cartesian deltaをjoint targetへ変換し、
        robot backendへ送る。

        execute=False:
            planのみ作成しロボットを動かさない。
        """

        plan = self.plan_tcp_delta(
            delta_position_m
        )


        result = {
            "plan": plan,
            "command_result": None,
        }


        if not execute:
            return result


        command_result = (
            self.robot_backend
            .command_joint_positions(
                plan.q_target,
                duration_sec=(
                    self.command_duration_sec
                ),
            )
        )


        result[
            "command_result"
        ] = command_result


        return result