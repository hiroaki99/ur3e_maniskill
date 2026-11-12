#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yaw-aware orientation utilities for the UR3e Alignment grasp test.

This file is additive: it does not modify the already validated
orientation_locked_ik.py implementation.

Purpose
-------
1. Measure the gripper closing axis from the actual left/right finger geometry.
2. Select the nearest face normal of the square object.
3. Rotate the canonical TCP orientation only about world Z.
4. Interpolate the orientation during home -> pregrasp planning.

Quaternion convention: wxyz.
"""

from __future__ import annotations

import numpy as np

from day6_common import (
    full_qpos_from_arm,
    get_collision_center,
    get_grasp_center,
    set_robot_qpos,
)
from orientation_locked_ik import (
    get_tcp_quaternion,
    solve_pose_ik,
)


def normalize_quaternion(q):
    q = np.asarray(
        q,
        dtype=np.float64,
    ).reshape(4)

    norm = float(
        np.linalg.norm(q)
    )

    if norm <= 1.0e-12:
        raise ValueError(
            "Zero-norm quaternion."
        )

    return q / norm


def quaternion_multiply(a, b):
    """Hamilton product for wxyz quaternions."""
    a = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(4)
    b = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(4)

    aw, ax, ay, az = a
    bw, bx, by, bz = b

    result = np.array(
        [
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ],
        dtype=np.float64,
    )

    return normalize_quaternion(
        result
    )


def world_z_yaw_quaternion(yaw_rad):
    """World-frame rotation about +Z, wxyz."""
    half = 0.5 * float(
        yaw_rad
    )

    return np.array(
        [
            np.cos(half),
            0.0,
            0.0,
            np.sin(half),
        ],
        dtype=np.float64,
    )


def quaternion_slerp(q0, q1, ratio):
    """Shortest-path SLERP for wxyz quaternions."""
    q0 = normalize_quaternion(
        q0
    )
    q1 = normalize_quaternion(
        q1
    )

    t = float(
        np.clip(
            ratio,
            0.0,
            1.0,
        )
    )

    dot = float(
        np.dot(
            q0,
            q1,
        )
    )

    # q and -q are the same orientation; use the shorter arc.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )

    if dot > 0.9995:
        return normalize_quaternion(
            (1.0 - t) * q0
            + t * q1
        )

    theta_0 = float(
        np.arccos(dot)
    )
    sin_theta_0 = float(
        np.sin(theta_0)
    )

    theta = theta_0 * t

    s0 = (
        np.sin(
            theta_0 - theta
        )
        / sin_theta_0
    )
    s1 = (
        np.sin(theta)
        / sin_theta_0
    )

    return normalize_quaternion(
        s0 * q0
        + s1 * q1
    )


def wrap_to_pi(angle_rad):
    return float(
        (
            float(angle_rad)
            + np.pi
        )
        % (
            2.0 * np.pi
        )
        - np.pi
    )


def yaw_from_wxyz(q):
    """Extract world Z yaw from a wxyz quaternion."""
    q = normalize_quaternion(
        q
    )

    w, x, y, z = q

    siny_cosp = (
        2.0
        * (
            w*z
            + x*y
        )
    )
    cosy_cosp = (
        1.0
        - 2.0
        * (
            y*y
            + z*z
        )
    )

    return float(
        np.arctan2(
            siny_cosp,
            cosy_cosp,
        )
    )


def mean_link_center(links):
    centers = np.stack(
        [
            get_collision_center(
                link
            )
            for link in links
        ],
        axis=0,
    )

    return centers.mean(
        axis=0
    )


def finger_closing_axis_yaw(
    left_links,
    right_links,
):
    """
    Return the world-XY direction from the left finger group to the right
    finger group.  This is the gripper closing-axis direction.

    The direction sign is not important for a square because both +normal
    and -normal are included when selecting the nearest face.
    """
    left = mean_link_center(
        left_links
    )
    right = mean_link_center(
        right_links
    )

    axis_xy = (
        right[:2]
        - left[:2]
    )

    norm_xy = float(
        np.linalg.norm(
            axis_xy
        )
    )

    if norm_xy <= 1.0e-6:
        raise RuntimeError(
            "Finger closing axis is degenerate in world XY: "
            f"left={left}, right={right}"
        )

    yaw = float(
        np.arctan2(
            axis_xy[1],
            axis_xy[0],
        )
    )

    return (
        yaw,
        left,
        right,
    )


def nearest_square_face_normal(
    object_yaw_rad,
    closing_axis_yaw_rad,
):
    """
    A square has four equivalent face normals:
        yaw + k*pi/2, k=0..3.

    Choose the face normal requiring the smallest world-Z rotation of the
    current closing axis.
    """
    candidates = np.asarray(
        [
            float(object_yaw_rad)
            + k * np.pi / 2.0
            for k in range(4)
        ],
        dtype=np.float64,
    )

    deltas = np.asarray(
        [
            wrap_to_pi(
                candidate
                - float(
                    closing_axis_yaw_rad
                )
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    index = int(
        np.argmin(
            np.abs(
                deltas
            )
        )
    )

    return (
        float(
            candidates[index]
        ),
        float(
            deltas[index]
        ),
        index,
    )


def build_object_yaw_aware_tcp_orientation(
    *,
    left_links,
    right_links,
    tcp_link,
    object_yaw_rad,
):
    """
    Build the target TCP quaternion by rotating the canonical/home TCP
    orientation about WORLD Z so that the finger closing axis is parallel to
    the nearest face normal of object A.

    q_target = q_world_z(delta_yaw) * q_home
    """
    home_tcp_quaternion = (
        get_tcp_quaternion(
            tcp_link
        )
    )

    (
        closing_axis_yaw,
        left_center,
        right_center,
    ) = finger_closing_axis_yaw(
        left_links,
        right_links,
    )

    (
        face_normal_yaw,
        delta_yaw,
        face_index,
    ) = nearest_square_face_normal(
        object_yaw_rad,
        closing_axis_yaw,
    )

    delta_quaternion = (
        world_z_yaw_quaternion(
            delta_yaw
        )
    )

    target_tcp_quaternion = (
        quaternion_multiply(
            delta_quaternion,
            home_tcp_quaternion,
        )
    )

    return {
        "home_tcp_quaternion":
            home_tcp_quaternion,
        "target_tcp_quaternion":
            target_tcp_quaternion,
        "object_yaw_rad":
            float(
                object_yaw_rad
            ),
        "closing_axis_yaw_rad":
            float(
                closing_axis_yaw
            ),
        "selected_face_normal_yaw_rad":
            float(
                face_normal_yaw
            ),
        "yaw_correction_rad":
            float(
                delta_yaw
            ),
        "selected_face_index":
            int(
                face_index
            ),
        "left_finger_center_m":
            np.asarray(
                left_center,
                dtype=np.float64,
            ),
        "right_finger_center_m":
            np.asarray(
                right_center,
                dtype=np.float64,
            ),
    }


def plan_cartesian_pose_segment_interpolated(
    *,
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
    start_quaternion,
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
    """
    Position interpolation + quaternion SLERP.

    This is used only for home -> pregrasp so the yaw correction is spread
    across the whole approach rather than demanded at the first waypoint.
    """
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

    start_quaternion = (
        normalize_quaternion(
            start_quaternion
        )
    )
    target_quaternion = (
        normalize_quaternion(
            target_quaternion
        )
    )

    q_current = (
        start_arm_qpos.copy()
    )

    q_path = []
    position_path = []
    orientation_target_path = []
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

        waypoint_quaternion = (
            quaternion_slerp(
                start_quaternion,
                target_quaternion,
                ratio,
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
            target_quaternion=(
                waypoint_quaternion
            ),
            epsilon=epsilon,
            damping=damping,
            position_tolerance=(
                position_tolerance
            ),
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
                "Interpolated pose IK did not converge. "
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
        orientation_target_path.append(
            waypoint_quaternion.tolist()
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
        "target_orientations_wxyz":
            orientation_target_path,
        "errors_m":
            position_errors,
        "orientation_errors_rad":
            orientation_errors_rad,
        "final_arm_qpos":
            q_current.tolist(),
    }
