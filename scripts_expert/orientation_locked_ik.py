#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical Cartesian pose IK for the UR3e shadow-planning environment.

Important:
- This helper is for the SHADOW planning environment only.
- It uses robot.set_qpos() only to evaluate FK/Jacobians numerically.
- Physical/ManiSkill tested motion must still be executed through env.step().
- Quaternion convention is ManiSkill/SAPIEN wxyz.
"""

from __future__ import annotations

import numpy as np

from day6_common import (
    clamp_arm_qpos,
    first_env,
    full_qpos_from_arm,
    get_grasp_center,
    set_robot_qpos,
)


def _normalize_quaternion(q):
    q = np.asarray(
        q,
        dtype=np.float64,
    )
    norm = float(
        np.linalg.norm(q)
    )
    if norm <= 1.0e-12:
        raise ValueError(
            "Zero-norm quaternion"
        )
    return q / norm


def _quat_conjugate(q):
    q = _normalize_quaternion(q)
    return np.array(
        [
            q[0],
            -q[1],
            -q[2],
            -q[3],
        ],
        dtype=np.float64,
    )


def _quat_multiply(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )
    b = np.asarray(
        b,
        dtype=np.float64,
    )

    aw, ax, ay, az = a
    bw, bx, by, bz = b

    return np.array(
        [
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ],
        dtype=np.float64,
    )


def quaternion_rotation_vector(
    target_q,
    current_q,
):
    """
    World-frame small rotation taking current orientation to target.
    """
    target = _normalize_quaternion(
        target_q
    )
    current = _normalize_quaternion(
        current_q
    )

    error_q = _quat_multiply(
        target,
        _quat_conjugate(
            current
        ),
    )

    # q and -q represent the same rotation.
    if error_q[0] < 0.0:
        error_q = -error_q

    error_q = _normalize_quaternion(
        error_q
    )

    w = float(
        np.clip(
            error_q[0],
            -1.0,
            1.0,
        )
    )
    v = error_q[1:]
    v_norm = float(
        np.linalg.norm(v)
    )

    if v_norm < 1.0e-12:
        return np.zeros(
            3,
            dtype=np.float64,
        )

    angle = 2.0 * np.arctan2(
        v_norm,
        w,
    )

    return (
        angle
        * v
        / v_norm
    )


def get_tcp_quaternion(
    tcp_link,
):
    return _normalize_quaternion(
        first_env(
            tcp_link.pose.q
        )
    )


def numerical_pose_jacobian(
    robot,
    device,
    left_links,
    right_links,
    tcp_link,
    template_qpos,
    arm_indices,
    arm_qpos,
    epsilon,
):
    base_full = (
        full_qpos_from_arm(
            template_qpos,
            arm_indices,
            arm_qpos,
        )
    )

    set_robot_qpos(
        robot,
        device,
        base_full,
    )

    base_position = (
        get_grasp_center(
            left_links,
            right_links,
        )
    )
    base_quaternion = (
        get_tcp_quaternion(
            tcp_link
        )
    )

    position_jacobian = np.zeros(
        (
            3,
            len(arm_indices),
        ),
        dtype=np.float64,
    )
    rotation_jacobian = np.zeros(
        (
            3,
            len(arm_indices),
        ),
        dtype=np.float64,
    )

    for column in range(
        len(arm_indices)
    ):
        q_perturbed = (
            arm_qpos.copy()
        )
        q_perturbed[
            column
        ] += epsilon

        full_perturbed = (
            full_qpos_from_arm(
                template_qpos,
                arm_indices,
                q_perturbed,
            )
        )

        set_robot_qpos(
            robot,
            device,
            full_perturbed,
        )

        position = (
            get_grasp_center(
                left_links,
                right_links,
            )
        )
        quaternion = (
            get_tcp_quaternion(
                tcp_link
            )
        )

        position_jacobian[
            :,
            column,
        ] = (
            position
            - base_position
        ) / epsilon

        # Current orientation change caused by +epsilon.
        rotation_jacobian[
            :,
            column,
        ] = (
            quaternion_rotation_vector(
                quaternion,
                base_quaternion,
            )
            / epsilon
        )

    set_robot_qpos(
        robot,
        device,
        base_full,
    )

    return (
        position_jacobian,
        rotation_jacobian,
        base_position,
        base_quaternion,
    )


def solve_pose_ik(
    robot,
    device,
    left_links,
    right_links,
    tcp_link,
    template_qpos,
    arm_indices,
    qlimits_arm,
    start_arm_qpos,
    target_position,
    target_quaternion,
    epsilon,
    damping,
    position_tolerance,
    orientation_tolerance_rad,
    orientation_weight_m_per_rad,
    max_iterations,
    max_step,
):
    q_arm = (
        start_arm_qpos.copy()
    )

    final_position_error = float(
        "inf"
    )
    final_orientation_error = float(
        "inf"
    )

    for iteration in range(
        max_iterations
    ):
        (
            J_pos,
            J_rot,
            current_position,
            current_quaternion,
        ) = numerical_pose_jacobian(
            robot=robot,
            device=device,
            left_links=left_links,
            right_links=right_links,
            tcp_link=tcp_link,
            template_qpos=template_qpos,
            arm_indices=arm_indices,
            arm_qpos=q_arm,
            epsilon=epsilon,
        )

        position_error = (
            np.asarray(
                target_position,
                dtype=np.float64,
            )
            - current_position
        )

        rotation_error = (
            quaternion_rotation_vector(
                target_quaternion,
                current_quaternion,
            )
        )

        final_position_error = float(
            np.linalg.norm(
                position_error
            )
        )
        final_orientation_error = float(
            np.linalg.norm(
                rotation_error
            )
        )

        if (
            final_position_error
            <= position_tolerance
            and final_orientation_error
            <= orientation_tolerance_rad
        ):
            return (
                q_arm,
                final_position_error,
                final_orientation_error,
                iteration + 1,
            )

        weight = float(
            orientation_weight_m_per_rad
        )

        J = np.vstack(
            [
                J_pos,
                weight * J_rot,
            ]
        )
        error = np.concatenate(
            [
                position_error,
                weight * rotation_error,
            ]
        )

        matrix = (
            J @ J.T
            + (
                damping ** 2
            )
            * np.eye(6)
        )

        dq = (
            J.T
            @ np.linalg.solve(
                matrix,
                error,
            )
        )

        dq = np.clip(
            dq,
            -max_step,
            max_step,
        )

        q_arm = clamp_arm_qpos(
            q_arm + dq,
            qlimits_arm,
        )

    return (
        q_arm,
        final_position_error,
        final_orientation_error,
        max_iterations,
    )


def plan_cartesian_pose_segment(
    robot,
    device,
    left_links,
    right_links,
    tcp_link,
    template_qpos,
    arm_indices,
    qlimits_arm,
    start_arm_qpos,
    target_position,
    target_quaternion,
    waypoint_count,
    epsilon,
    damping,
    position_tolerance,
    orientation_tolerance_rad,
    orientation_weight_m_per_rad,
    max_iterations,
    max_step,
):
    start_full = (
        full_qpos_from_arm(
            template_qpos,
            arm_indices,
            start_arm_qpos,
        )
    )

    set_robot_qpos(
        robot,
        device,
        start_full,
    )

    start_position = (
        get_grasp_center(
            left_links,
            right_links,
        )
    )

    q_current = (
        start_arm_qpos.copy()
    )

    q_path = []
    position_path = []
    position_errors = []
    orientation_errors_rad = []

    for index in range(
        1,
        waypoint_count + 1,
    ):
        ratio = (
            index
            / waypoint_count
        )

        target = (
            start_position
            + ratio
            * (
                np.asarray(
                    target_position,
                    dtype=np.float64,
                )
                - start_position
            )
        )

        (
            q_solution,
            position_error,
            orientation_error,
            _iterations,
        ) = solve_pose_ik(
            robot=robot,
            device=device,
            left_links=left_links,
            right_links=right_links,
            tcp_link=tcp_link,
            template_qpos=template_qpos,
            arm_indices=arm_indices,
            qlimits_arm=qlimits_arm,
            start_arm_qpos=q_current,
            target_position=target,
            target_quaternion=target_quaternion,
            epsilon=epsilon,
            damping=damping,
            position_tolerance=position_tolerance,
            orientation_tolerance_rad=(
                orientation_tolerance_rad
            ),
            orientation_weight_m_per_rad=(
                orientation_weight_m_per_rad
            ),
            max_iterations=max_iterations,
            max_step=max_step,
        )

        if (
            position_error
            > position_tolerance * 3.0
            or orientation_error
            > orientation_tolerance_rad * 3.0
        ):
            raise RuntimeError(
                "Pose IK did not converge. "
                f"waypoint={index}/{waypoint_count}, "
                f"position_error={position_error:.6f} m, "
                f"orientation_error="
                f"{np.degrees(orientation_error):.3f} deg"
            )

        q_current = (
            q_solution.copy()
        )

        full = (
            full_qpos_from_arm(
                template_qpos,
                arm_indices,
                q_current,
            )
        )

        set_robot_qpos(
            robot,
            device,
            full,
        )

        actual_position = (
            get_grasp_center(
                left_links,
                right_links,
            )
        )

        q_path.append(
            q_current.tolist()
        )
        position_path.append(
            actual_position.tolist()
        )
        position_errors.append(
            float(
                position_error
            )
        )
        orientation_errors_rad.append(
            float(
                orientation_error
            )
        )

    return {
        "arm_qpos":
            q_path,
        "positions":
            position_path,
        "errors_m":
            position_errors,
        "orientation_errors_rad":
            orientation_errors_rad,
        "final_arm_qpos":
            q_current.tolist(),
    }
