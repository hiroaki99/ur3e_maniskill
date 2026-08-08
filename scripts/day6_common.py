#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from mani_skill.utils.geometry.trimesh_utils import (
    get_component_mesh,
)


def to_numpy(value: Any):

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def first_env(value: Any):

    array = to_numpy(value)

    if (
        array.ndim >= 2
        and array.shape[0] == 1
    ):
        return array[0]

    return array


def scalar_first(value) -> float:

    array = np.asarray(
        first_env(value)
    )

    if array.size != 1:
        raise ValueError(
            f"scalarを想定: shape={array.shape}"
        )

    return float(
        array.reshape(-1)[0]
    )


def get_controller_space(controller):

    if hasattr(
        controller,
        "single_action_space",
    ):
        return controller.single_action_space

    return controller.action_space


def build_controller_slices(
    controllers: Mapping,
):

    result = {}

    offset = 0

    for name, controller in controllers.items():

        space = get_controller_space(
            controller
        )

        dim = int(
            np.prod(space.shape)
        )

        result[str(name)] = slice(
            offset,
            offset + dim,
        )

        offset += dim

    return result


def find_controller(
    controllers,
    keyword: str,
):

    for name in controllers:

        if keyword in str(name).lower():
            return str(name)

    raise RuntimeError(
        f"{keyword} controllerがありません: "
        f"{list(controllers.keys())}"
    )


def get_controller_joint_names(
    controller,
):

    names = getattr(
        controller.config,
        "joint_names",
        None,
    )

    if names is None:
        return []

    return [
        str(name)
        for name in names
    ]


def get_collision_center(link):

    mesh = get_component_mesh(
        link._objs[0],
        to_world_frame=True,
    )

    if mesh is None:
        raise RuntimeError(
            f"{link.name}: collision meshがありません"
        )

    bounds = np.asarray(
        mesh.bounds,
        dtype=np.float64,
    )

    return (
        bounds[0]
        + bounds[1]
    ) / 2.0


def get_grasp_center(
    left_links,
    right_links,
):

    left_centers = np.stack(
        [
            get_collision_center(link)
            for link in left_links
        ],
        axis=0,
    )

    right_centers = np.stack(
        [
            get_collision_center(link)
            for link in right_links
        ],
        axis=0,
    )

    left = left_centers.mean(
        axis=0
    )

    right = right_centers.mean(
        axis=0
    )

    center = (
        left + right
    ) / 2.0

    return center


def set_robot_qpos(
    robot,
    device,
    qpos,
):

    q = torch.tensor(
        qpos,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    robot.set_qpos(q)

    robot.set_qvel(
        torch.zeros_like(q)
    )


def clamp_arm_qpos(
    q_arm,
    qlimits_arm,
):

    q = q_arm.copy()

    for i in range(
        len(q)
    ):

        lower = qlimits_arm[
            i, 0
        ]

        upper = qlimits_arm[
            i, 1
        ]

        if np.isfinite(lower):
            q[i] = max(
                q[i],
                lower,
            )

        if np.isfinite(upper):
            q[i] = min(
                q[i],
                upper,
            )

    return q


def full_qpos_from_arm(
    template_qpos,
    arm_indices,
    arm_qpos,
):

    q = template_qpos.copy()

    q[
        arm_indices
    ] = arm_qpos

    return q


def numerical_position_jacobian(
    robot,
    device,
    left_links,
    right_links,
    template_qpos,
    arm_indices,
    arm_qpos,
    epsilon,
):

    base_full = full_qpos_from_arm(
        template_qpos,
        arm_indices,
        arm_qpos,
    )

    set_robot_qpos(
        robot,
        device,
        base_full,
    )

    base_position = get_grasp_center(
        left_links,
        right_links,
    )

    jacobian = np.zeros(
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

        position = get_grasp_center(
            left_links,
            right_links,
        )

        jacobian[
            :,
            column,
        ] = (
            position
            - base_position
        ) / epsilon

    # restore
    set_robot_qpos(
        robot,
        device,
        base_full,
    )

    return (
        jacobian,
        base_position,
    )


def solve_position_ik(
    robot,
    device,
    left_links,
    right_links,
    template_qpos,
    arm_indices,
    qlimits_arm,
    start_arm_qpos,
    target_position,
    epsilon,
    damping,
    tolerance,
    max_iterations,
    max_step,
):

    q_arm = (
        start_arm_qpos.copy()
    )

    final_error = None

    for iteration in range(
        max_iterations
    ):

        (
            jacobian,
            current_position,
        ) = numerical_position_jacobian(
            robot=robot,
            device=device,
            left_links=left_links,
            right_links=right_links,
            template_qpos=template_qpos,
            arm_indices=arm_indices,
            arm_qpos=q_arm,
            epsilon=epsilon,
        )

        error = (
            target_position
            - current_position
        )

        error_norm = float(
            np.linalg.norm(
                error
            )
        )

        final_error = error_norm

        if error_norm <= tolerance:

            return (
                q_arm,
                error_norm,
                iteration + 1,
            )

        # Damped Least Squares
        matrix = (
            jacobian
            @ jacobian.T
            + (
                damping ** 2
            )
            * np.eye(3)
        )

        dq = (
            jacobian.T
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

        q_arm = (
            q_arm
            + dq
        )

        q_arm = clamp_arm_qpos(
            q_arm,
            qlimits_arm,
        )

    return (
        q_arm,
        float(final_error),
        max_iterations,
    )


def plan_cartesian_segment(
    robot,
    device,
    left_links,
    right_links,
    template_qpos,
    arm_indices,
    qlimits_arm,
    start_arm_qpos,
    target_position,
    waypoint_count,
    epsilon,
    damping,
    tolerance,
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

    start_position = get_grasp_center(
        left_links,
        right_links,
    )

    q_current = (
        start_arm_qpos.copy()
    )

    q_path = []

    position_path = []

    errors = []

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
                target_position
                - start_position
            )
        )

        (
            q_solution,
            error,
            iterations,
        ) = solve_position_ik(
            robot=robot,
            device=device,
            left_links=left_links,
            right_links=right_links,
            template_qpos=template_qpos,
            arm_indices=arm_indices,
            qlimits_arm=qlimits_arm,
            start_arm_qpos=q_current,
            target_position=target,
            epsilon=epsilon,
            damping=damping,
            tolerance=tolerance,
            max_iterations=max_iterations,
            max_step=max_step,
        )

        if error > tolerance * 3.0:

            raise RuntimeError(
                "IKが収束しませんでした。 "
                f"waypoint={index}/"
                f"{waypoint_count}, "
                f"error={error:.6f} m"
            )

        q_current = (
            q_solution.copy()
        )

        full = full_qpos_from_arm(
            template_qpos,
            arm_indices,
            q_current,
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

        errors.append(
            float(error)
        )

    return {
        "arm_qpos": q_path,
        "positions": position_path,
        "errors_m": errors,
        "final_arm_qpos": (
            q_current.tolist()
        ),
    }