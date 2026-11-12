#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 5D: yaw-aware grasp + strict Yaginuma fixed Expert sequence [6,4,3] with zero residual.

Purpose
-------
Validate that the existing UR3e+EZGripper grasp stack can grasp a randomized
alignment object A before GP Expert execution is connected.

Architecture
------------
planning_env (shadow)
    -> Day6 numerical position IK
    -> home -> pregrasp -> grasp -> 10 mm micro-lift q paths

physical_env
    -> same seed / same A,B,C reset
    -> env.step() only
    -> pre-contact
    -> execute approach/descent
    -> gradual gripper close
    -> bilateral contact
    -> stable grasp
    -> 10 mm micro-lift
    -> verify A follows the gripper

Important
---------
- Direct robot.set_qpos is used only inside the separate planning shadow env
  by the existing Day6 numerical IK utilities.
- The physical env never uses set_qpos for tested motion.
- Strict Yaginuma begin/end correction is applied to Experts [6,4,3].
- After every completed Expert, the measured A logical [x,z] is fed to the next begin correction.
- ManiSkill success observed inside an Expert is recorded but does not stop the segment; source-style done is checked at the Expert endpoint.
- TD3 residual is fixed to [0, 0] for this phase.
- Expert motion is executed only through ManiSkill env.step().
- No ROS2 / real robot backend is imported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "scripts_pickplace",
    PROJECT_ROOT / "scripts_expert",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import mani_skill.envs  # noqa: F401,E402
import envs.ur3e_alignment  # noqa: F401,E402

from day6_common import (  # noqa: E402
    first_env,
    get_grasp_center,
    set_robot_qpos,
)
from orientation_locked_ik import (  # noqa: E402
    get_tcp_quaternion,
    plan_cartesian_pose_segment,
)
from yaw_aware_orientation import (  # noqa: E402
    build_object_yaw_aware_tcp_orientation,
    plan_cartesian_pose_segment_interpolated,
    yaw_from_wxyz,
)
from expert_gp.expert_correction import (  # noqa: E402
    correct_expert_begin,
    correct_expert_end,
    load_expert_mean_sigma,
)
from expert_gp.expert_coordinate_mapper import (  # noqa: E402
    ExpertCoordinateMapper,
)

from pickplace_week1_common import (  # noqa: E402
    build_context,
    step_robot,
    run_path,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "ur3e_alignment.yaml"
)

EXPERT_MODEL_DIR = (
    PROJECT_ROOT
    / "reports"
    / "expert_gp"
)

PHASE5D_EXPERT_SEQUENCE = (6, 4, 3)
PHASE5C_ZERO_RESIDUAL = np.asarray(
    [0.0, 0.0],
    dtype=np.float64,
)
YAGINUMA_EXECUTION_STRIDE = 100
EXPERT_EXECUTION_INTERP_STEP_M = 1.0e-3
EXPERT_ENDPOINT_ERROR_LIMIT_M = 5.0e-3


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help=(
            "Gym TimeLimit override for the long Phase-5D "
            "[6,4,3] sequence. This changes only the test "
            "harness TimeLimit, not the task or RRL algorithm."
        ),
    )

    parser.add_argument(
        "--orientation-tolerance-deg",
        type=float,
        default=0.25,
        help=(
            "Pose-IK orientation convergence tolerance."
        ),
    )

    parser.add_argument(
        "--orientation-weight-m-per-rad",
        type=float,
        default=0.05,
        help=(
            "DLS task-space scaling for rotational error."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. "
            "If omitted, a seed-specific filename is used."
        ),
    )

    return parser.parse_args()


def scalar(value):
    array = np.asarray(
        first_env(value)
    )

    if array.size != 1:
        raise ValueError(
            f"Expected scalar, got shape={array.shape}"
        )

    return float(
        array.reshape(-1)[0]
    )


def as_bool(value):
    return bool(
        scalar(value)
    )


def object_position(actor):
    return (
        first_env(
            actor.pose.p
        )
        .astype(np.float64)
        .copy()
    )


def pose9_numpy(base_env):
    return (
        first_env(
            base_env
            .get_alignment_pose9()
        )
        .astype(np.float64)
        .copy()
    )


def quaternion_angle_deg(q0, q1):
    """Smallest angle between wxyz quaternions."""
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(
        np.clip(
            abs(np.dot(q0, q1)),
            0.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            2.0 * np.arccos(dot)
        )
    )


def hold_precontact(
    env,
    ctx,
    *,
    pre_contact,
    steps,
    render,
    sleep,
):
    current_arm = (
        first_env(
            ctx["robot"]
            .get_qpos()
        )
        .astype(float)[
            ctx["arm_indices"]
        ]
        .copy()
    )

    last_info = None

    for _ in range(
        int(steps)
    ):
        (
            _obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = step_robot(
            env,
            ctx,
            current_arm,
            pre_contact,
            render=render,
            sleep=sleep,
        )

        last_info = info

        if as_bool(terminated):
            raise RuntimeError(
                "Unexpected termination during pre-contact hold."
            )

        if as_bool(truncated):
            raise RuntimeError(
                "Unexpected truncation during pre-contact hold."
            )

    return last_info


def make_env(
    max_episode_steps: int,
):
    max_episode_steps = int(
        max_episode_steps
    )

    if max_episode_steps < 1000:
        raise ValueError(
            "--max-episode-steps must be >= 1000 "
            "for the long Phase-5D diagnostic."
        )

    return gym.make(
        "UR3eAlignment-v0",
        num_envs=1,
        obs_mode="state",
        reward_mode="dense",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        max_episode_steps=max_episode_steps,
    )


def plan_paths(
    env,
    config,
    *,
    seed,
    orientation_tolerance_rad,
    orientation_weight_m_per_rad,
):
    env.reset(
        seed=int(seed)
    )

    base = env.unwrapped
    ctx = build_context(
        env,
        config,
    )

    gripper_cfg = config[
        "gripper"
    ]
    grasp_cfg = config[
        "grasp"
    ]
    ik_cfg = config[
        "ik"
    ]

    pre_contact = float(
        gripper_cfg[
            "pre_contact_action"
        ]
    )

    hold_precontact(
        env,
        ctx,
        pre_contact=pre_contact,
        steps=int(
            grasp_cfg[
                "reset_precontact_steps"
            ]
        ),
        render=False,
        sleep=0.0,
    )

    robot = ctx["robot"]

    template_qpos = (
        first_env(
            robot.get_qpos()
        )
        .astype(float)
        .copy()
    )

    home_arm = (
        template_qpos[
            ctx["arm_indices"]
        ]
        .copy()
    )

    qlimits = (
        first_env(
            robot.get_qlimits()
        )
        .astype(float)
    )
    qlimits_arm = (
        qlimits[
            ctx["arm_indices"]
        ]
    )

    A_position = (
        object_position(
            base.object_A
        )
    )

    A_quaternion = (
        first_env(
            base.object_A.pose.q
        )
        .astype(np.float64)
        .copy()
    )

    A_yaw_rad = (
        yaw_from_wxyz(
            A_quaternion
        )
    )

    grasp_offset = np.asarray(
        grasp_cfg[
            "grasp_center_offset_m"
        ],
        dtype=np.float64,
    )

    grasp_target = (
        A_position
        + grasp_offset
    )

    pregrasp_target = (
        grasp_target.copy()
    )
    pregrasp_target[2] += float(
        grasp_cfg[
            "pregrasp_height_m"
        ]
    )

    lift_target = (
        grasp_target.copy()
    )
    lift_target[2] += float(
        grasp_cfg[
            "micro_lift_height_m"
        ]
    )

    common = {
        "robot":
            robot,
        "device":
            base.device,
        "left_links":
            base.left_contact_links,
        "right_links":
            base.right_contact_links,
        "template_qpos":
            template_qpos,
        "arm_indices":
            ctx["arm_indices"],
        "qlimits_arm":
            qlimits_arm,
        "epsilon":
            float(
                ik_cfg[
                    "epsilon_rad"
                ]
            ),
        "damping":
            float(
                ik_cfg[
                    "damping"
                ]
            ),
        "tolerance":
            float(
                ik_cfg[
                    "tolerance_m"
                ]
            ),
        "max_iterations":
            int(
                ik_cfg[
                    "max_iterations"
                ]
            ),
        "max_step":
            float(
                ik_cfg[
                    "max_step_rad"
                ]
            ),
    }

    home_center = get_grasp_center(
        base.left_contact_links,
        base.right_contact_links,
    )

    yaw_alignment = (
        build_object_yaw_aware_tcp_orientation(
            left_links=base.left_contact_links,
            right_links=base.right_contact_links,
            tcp_link=base.agent.tcp,
            object_yaw_rad=A_yaw_rad,
        )
    )

    home_tcp_quaternion = np.asarray(
        yaw_alignment[
            "home_tcp_quaternion"
        ],
        dtype=np.float64,
    )

    grasp_tcp_quaternion = np.asarray(
        yaw_alignment[
            "target_tcp_quaternion"
        ],
        dtype=np.float64,
    )

    print(
        "home TCP quaternion [wxyz]:",
        home_tcp_quaternion,
    )
    print(
        "planning A world yaw [deg]:",
        np.degrees(
            A_yaw_rad
        ),
    )
    print(
        "home finger closing-axis yaw [deg]:",
        np.degrees(
            yaw_alignment[
                "closing_axis_yaw_rad"
            ]
        ),
    )
    print(
        "selected cube-face normal yaw [deg]:",
        np.degrees(
            yaw_alignment[
                "selected_face_normal_yaw_rad"
            ]
        ),
    )
    print(
        "TCP yaw correction [deg]:",
        np.degrees(
            yaw_alignment[
                "yaw_correction_rad"
            ]
        ),
    )
    print(
        "yaw-aware TCP quaternion [wxyz]:",
        grasp_tcp_quaternion,
    )

    print(
        "planning A world position:",
        A_position,
    )
    print(
        "planning home grasp center:",
        home_center,
    )
    print(
        "pregrasp target:",
        pregrasp_target,
    )
    print(
        "grasp target:",
        grasp_target,
    )
    print(
        "micro-lift target:",
        lift_target,
    )

    to_pregrasp = (
        plan_cartesian_pose_segment_interpolated(
            robot=robot,
            device=base.device,
            left_links=base.left_contact_links,
            right_links=base.right_contact_links,
            tcp_link=base.agent.tcp,
            template_qpos=template_qpos,
            arm_indices=ctx["arm_indices"],
            qlimits_arm=qlimits_arm,
            start_arm_qpos=home_arm,
            target_position=pregrasp_target,
            start_quaternion=home_tcp_quaternion,
            target_quaternion=grasp_tcp_quaternion,
            waypoint_count=int(
                grasp_cfg[
                    "to_pregrasp_waypoints"
                ]
            ),
            epsilon=float(
                ik_cfg[
                    "epsilon_rad"
                ]
            ),
            damping=float(
                ik_cfg[
                    "damping"
                ]
            ),
            position_tolerance=float(
                ik_cfg[
                    "tolerance_m"
                ]
            ),
            orientation_tolerance_rad=(
                orientation_tolerance_rad
            ),
            orientation_weight_m_per_rad=(
                orientation_weight_m_per_rad
            ),
            max_iterations=int(
                ik_cfg[
                    "max_iterations"
                ]
            ),
            max_step=float(
                ik_cfg[
                    "max_step_rad"
                ]
            ),
        )
    )

    q_pregrasp = np.asarray(
        to_pregrasp[
            "final_arm_qpos"
        ],
        dtype=np.float64,
    )

    descend = (
        plan_cartesian_pose_segment(
            robot=robot,
            device=base.device,
            left_links=base.left_contact_links,
            right_links=base.right_contact_links,
            tcp_link=base.agent.tcp,
            template_qpos=template_qpos,
            arm_indices=ctx["arm_indices"],
            qlimits_arm=qlimits_arm,
            start_arm_qpos=q_pregrasp,
            target_position=grasp_target,
            target_quaternion=grasp_tcp_quaternion,
            waypoint_count=int(
                grasp_cfg[
                    "descend_waypoints"
                ]
            ),
            epsilon=float(
                ik_cfg[
                    "epsilon_rad"
                ]
            ),
            damping=float(
                ik_cfg[
                    "damping"
                ]
            ),
            position_tolerance=float(
                ik_cfg[
                    "tolerance_m"
                ]
            ),
            orientation_tolerance_rad=(
                orientation_tolerance_rad
            ),
            orientation_weight_m_per_rad=(
                orientation_weight_m_per_rad
            ),
            max_iterations=int(
                ik_cfg[
                    "max_iterations"
                ]
            ),
            max_step=float(
                ik_cfg[
                    "max_step_rad"
                ]
            ),
        )
    )

    q_grasp = np.asarray(
        descend[
            "final_arm_qpos"
        ],
        dtype=np.float64,
    )

    lift_common = dict(
        common
    )
    lift_common["tolerance"] = float(
        grasp_cfg[
            "micro_lift_ik_tolerance_m"
        ]
    )

    lift = (
        plan_cartesian_pose_segment(
            robot=robot,
            device=base.device,
            left_links=base.left_contact_links,
            right_links=base.right_contact_links,
            tcp_link=base.agent.tcp,
            template_qpos=template_qpos,
            arm_indices=ctx["arm_indices"],
            qlimits_arm=qlimits_arm,
            start_arm_qpos=q_grasp,
            target_position=lift_target,
            target_quaternion=grasp_tcp_quaternion,
            waypoint_count=int(
                grasp_cfg[
                    "lift_waypoints"
                ]
            ),
            epsilon=float(
                ik_cfg[
                    "epsilon_rad"
                ]
            ),
            damping=float(
                ik_cfg[
                    "damping"
                ]
            ),
            position_tolerance=float(
                grasp_cfg[
                    "micro_lift_ik_tolerance_m"
                ]
            ),
            orientation_tolerance_rad=(
                orientation_tolerance_rad
            ),
            orientation_weight_m_per_rad=(
                orientation_weight_m_per_rad
            ),
            max_iterations=int(
                ik_cfg[
                    "max_iterations"
                ]
            ),
            max_step=float(
                ik_cfg[
                    "max_step_rad"
                ]
            ),
        )
    )

    lift_q = np.asarray(
        lift[
            "arm_qpos"
        ],
        dtype=np.float64,
    )

    if lift_q.shape[0] >= 2:
        lift_dq = np.diff(
            np.vstack(
                [
                    q_grasp[None, :],
                    lift_q,
                ]
            ),
            axis=0,
        )
        max_lift_transition_dq = float(
            np.max(
                np.abs(
                    lift_dq
                )
            )
        )
        zero_like_transitions = int(
            np.sum(
                np.max(
                    np.abs(
                        lift_dq
                    ),
                    axis=1,
                )
                < 1.0e-8
            )
        )
    else:
        max_lift_transition_dq = 0.0
        zero_like_transitions = 0

    print(
        "micro-lift IK tolerance [mm]:",
        lift_common["tolerance"] * 1000.0,
    )
    print(
        "micro-lift zero-like q transitions:",
        zero_like_transitions,
        "/",
        len(lift["arm_qpos"]),
    )
    print(
        "micro-lift planned max |dq| [rad]:",
        max_lift_transition_dq,
    )

    return {
        "A_world":
            A_position,
        "pregrasp_target":
            pregrasp_target,
        "grasp_target":
            grasp_target,
        "lift_target":
            lift_target,
        "to_pregrasp":
            to_pregrasp,
        "descend":
            descend,
        "lift":
            lift,
        "zero_like_transitions":
            int(
                zero_like_transitions
            ),
        "max_lift_transition_dq":
            float(
                max_lift_transition_dq
            ),
        "home_tcp_quaternion":
            home_tcp_quaternion.tolist(),
        "grasp_tcp_quaternion":
            grasp_tcp_quaternion.tolist(),
        "object_yaw_rad":
            float(
                A_yaw_rad
            ),
        "closing_axis_yaw_rad":
            float(
                yaw_alignment[
                    "closing_axis_yaw_rad"
                ]
            ),
        "selected_face_normal_yaw_rad":
            float(
                yaw_alignment[
                    "selected_face_normal_yaw_rad"
                ]
            ),
        "yaw_correction_rad":
            float(
                yaw_alignment[
                    "yaw_correction_rad"
                ]
            ),
        "selected_face_index":
            int(
                yaw_alignment[
                    "selected_face_index"
                ]
            ),
    }


def replan_micro_lift_from_actual_state(
    planning_env,
    physical_env,
    physical_ctx,
    config,
    *,
    target_tcp_quaternion,
    orientation_tolerance_rad,
    orientation_weight_m_per_rad,
):
    """
    Replan the lift from the ACTUAL post-grasp robot state.

    This is the generalized fix for the previous absolute-target mismatch:
    the nominal grasp target is only used for approach/descent.  Once a
    physical grasp is established, the lift target becomes

        current_actual_grasp_center + [0, 0, micro_lift_height]

    so contact-induced offsets do not shorten the intended relative lift.

    Direct set_qpos is used only in the shadow planning environment.
    """
    planning_base = planning_env.unwrapped
    physical_base = physical_env.unwrapped

    planning_ctx = build_context(
        planning_env,
        config,
    )

    physical_full_qpos = (
        first_env(
            physical_ctx["robot"].get_qpos()
        )
        .astype(float)
        .copy()
    )

    planning_robot = planning_ctx[
        "robot"
    ]

    # Shadow-only state synchronization.
    set_robot_qpos(
        planning_robot,
        planning_base.device,
        physical_full_qpos,
    )

    physical_center = np.asarray(
        get_grasp_center(
            physical_base.left_contact_links,
            physical_base.right_contact_links,
        ),
        dtype=np.float64,
    )

    shadow_center = np.asarray(
        get_grasp_center(
            planning_base.left_contact_links,
            planning_base.right_contact_links,
        ),
        dtype=np.float64,
    )

    center_sync_error_m = float(
        np.linalg.norm(
            shadow_center
            - physical_center
        )
    )

    if center_sync_error_m > 5.0e-4:
        raise RuntimeError(
            "Shadow/physical grasp-center mismatch is too large: "
            f"{center_sync_error_m*1000.0:.3f} mm"
        )

    grasp_cfg = config[
        "grasp"
    ]
    ik_cfg = config[
        "ik"
    ]

    relative_lift_m = float(
        grasp_cfg[
            "micro_lift_height_m"
        ]
    )

    lift_target = (
        shadow_center.copy()
    )
    lift_target[2] += (
        relative_lift_m
    )

    qlimits = (
        first_env(
            planning_robot.get_qlimits()
        )
        .astype(float)
    )

    qlimits_arm = (
        qlimits[
            planning_ctx[
                "arm_indices"
            ]
        ]
    )

    current_arm = (
        physical_full_qpos[
            planning_ctx[
                "arm_indices"
            ]
        ]
        .copy()
    )

    lift = plan_cartesian_pose_segment(
        robot=planning_robot,
        device=planning_base.device,
        left_links=planning_base.left_contact_links,
        right_links=planning_base.right_contact_links,
        tcp_link=planning_base.agent.tcp,
        template_qpos=physical_full_qpos,
        arm_indices=planning_ctx[
            "arm_indices"
        ],
        qlimits_arm=qlimits_arm,
        start_arm_qpos=current_arm,
        target_position=lift_target,
        target_quaternion=np.asarray(
            target_tcp_quaternion,
            dtype=np.float64,
        ),
        waypoint_count=int(
            grasp_cfg[
                "lift_waypoints"
            ]
        ),
        epsilon=float(
            ik_cfg[
                "epsilon_rad"
            ]
        ),
        damping=float(
            ik_cfg[
                "damping"
            ]
        ),
        position_tolerance=float(
            grasp_cfg[
                "micro_lift_ik_tolerance_m"
            ]
        ),
        orientation_tolerance_rad=(
            orientation_tolerance_rad
        ),
        orientation_weight_m_per_rad=(
            orientation_weight_m_per_rad
        ),
        max_iterations=int(
            ik_cfg[
                "max_iterations"
            ]
        ),
        max_step=float(
            ik_cfg[
                "max_step_rad"
            ]
        ),
    )

    lift_q = np.asarray(
        lift[
            "arm_qpos"
        ],
        dtype=np.float64,
    )

    if lift_q.shape[0] > 0:
        transitions = np.diff(
            np.vstack(
                [
                    current_arm[None, :],
                    lift_q,
                ]
            ),
            axis=0,
        )

        max_abs_dq = float(
            np.max(
                np.abs(
                    transitions
                )
            )
        )

        zero_like = int(
            np.sum(
                np.max(
                    np.abs(
                        transitions
                    ),
                    axis=1,
                )
                < 1.0e-8
            )
        )
    else:
        max_abs_dq = 0.0
        zero_like = 0

    return {
        "lift":
            lift,
        "physical_start_center_m":
            physical_center,
        "shadow_start_center_m":
            shadow_center,
        "target_center_m":
            lift_target,
        "center_sync_error_m":
            center_sync_error_m,
        "relative_lift_m":
            relative_lift_m,
        "max_abs_dq_rad":
            max_abs_dq,
        "zero_like_transitions":
            zero_like,
    }


def update_adaptive_gripper_command(
    command,
    left_force,
    right_force,
    *,
    low_force_n,
    high_force_n,
    command_step,
    min_command,
    max_command,
):
    """
    One-step hysteretic gripper update.

    Priority:
    1) If either finger is above the high-force band, open slightly.
    2) Else if either finger is below the stable-force band, close slightly.
    3) Otherwise hold.

    This intentionally depends on measured contact force, not object pose,
    seed, world position, or yaw.
    """
    command = float(command)

    if max(left_force, right_force) > high_force_n:
        command -= command_step
        mode = "open_reduce_force"

    elif min(left_force, right_force) < low_force_n:
        command += command_step
        mode = "close_recover_force"

    else:
        mode = "hold_force_band"

    command = float(
        np.clip(
            command,
            min_command,
            max_command,
        )
    )

    return command, mode



def yaginuma_execution_anchor_indices(
    trajectory_length: int,
) -> np.ndarray:
    """
    Reproduce gazebo_alignment.simulate() execution-point semantics:

        tmp[::100] + final point

    The strict corrected Expert 06 has 997 points, therefore the expected
    indices are:

        0, 100, 200, ..., 900, 996

    i.e. 11 anchors / 10 physical anchor-to-anchor transitions.
    """
    trajectory_length = int(
        trajectory_length
    )

    if trajectory_length < 2:
        raise ValueError(
            "trajectory_length must be >= 2."
        )

    indices = list(
        range(
            0,
            trajectory_length,
            YAGINUMA_EXECUTION_STRIDE,
        )
    )

    final_index = (
        trajectory_length - 1
    )

    if indices[-1] != final_index:
        indices.append(
            final_index
        )

    return np.asarray(
        indices,
        dtype=np.int64,
    )


def build_strict_expert_segment(
    expert_id: int,
    current_logical_xz,
):
    """
    Build the exact Phase-5C logical Expert trajectory:

        Expert 06 raw GP
        -> strict begin correction from measured A [x,z]
        -> strict end correction with residual [0,0]
        -> Yaginuma execution anchors
    """
    mean, sigma = (
        load_expert_mean_sigma(
            EXPERT_MODEL_DIR,
            expert_id,
        )
    )

    begin = correct_expert_begin(
        expert_id=expert_id,
        mean=mean,
        sigma=sigma,
        current_logical_xz=(
            current_logical_xz
        ),
    )

    end = correct_expert_end(
        mean=begin.corrected_mean,
        sigma=begin.corrected_sigma,
        ddpg_action=(
            PHASE5C_ZERO_RESIDUAL
        ),
    )

    if begin.corrected_length != 998:
        raise RuntimeError(
            "Strict Expert begin length mismatch: "
            f"{begin.corrected_length} != 998"
        )

    if end.corrected_length != 997:
        raise RuntimeError(
            "Strict Expert end length mismatch: "
            f"{end.corrected_length} != 997"
        )

    anchor_indices = (
        yaginuma_execution_anchor_indices(
            end.corrected_length
        )
    )

    logical_anchors = (
        end.corrected_mean[
            anchor_indices
        ]
        .copy()
    )

    if logical_anchors.shape != (11, 2):
        raise RuntimeError(
            "Unexpected Yaginuma anchor shape: "
            f"{logical_anchors.shape}; expected (11,2)."
        )

    return {
        "begin":
            begin,
        "end":
            end,
        "anchor_indices":
            anchor_indices,
        "logical_anchors":
            logical_anchors,
    }


def plan_expert_anchor_from_actual_state(
    *,
    planning_env,
    physical_env,
    physical_ctx,
    config,
    target_grasp_center,
    target_tcp_quaternion,
    orientation_tolerance_rad,
    orientation_weight_m_per_rad,
):
    """
    Plan one absolute grasp-center target from the ACTUAL physical arm state.

    Important:
    - set_qpos is used only in planning_env.
    - target_grasp_center is an absolute world target derived from the
      corrected object-A logical trajectory.
    - the physical TCP/grasp center is NOT used to re-anchor the Expert.
    """
    planning_base = (
        planning_env.unwrapped
    )
    physical_base = (
        physical_env.unwrapped
    )

    planning_ctx = build_context(
        planning_env,
        config,
    )

    physical_full_qpos = (
        first_env(
            physical_ctx[
                "robot"
            ].get_qpos()
        )
        .astype(float)
        .copy()
    )

    planning_robot = (
        planning_ctx[
            "robot"
        ]
    )

    # Shadow-only synchronization.
    set_robot_qpos(
        planning_robot,
        planning_base.device,
        physical_full_qpos,
    )

    physical_center = np.asarray(
        get_grasp_center(
            physical_base.left_contact_links,
            physical_base.right_contact_links,
        ),
        dtype=np.float64,
    )

    shadow_center = np.asarray(
        get_grasp_center(
            planning_base.left_contact_links,
            planning_base.right_contact_links,
        ),
        dtype=np.float64,
    )

    center_sync_error_m = float(
        np.linalg.norm(
            shadow_center
            - physical_center
        )
    )

    if center_sync_error_m > 5.0e-4:
        raise RuntimeError(
            "Expert shadow/physical grasp-center mismatch: "
            f"{center_sync_error_m*1000.0:.3f} mm"
        )

    target_grasp_center = np.asarray(
        target_grasp_center,
        dtype=np.float64,
    )

    distance_m = float(
        np.linalg.norm(
            target_grasp_center
            - physical_center
        )
    )

    waypoint_count = max(
        1,
        int(
            np.ceil(
                distance_m
                / EXPERT_EXECUTION_INTERP_STEP_M
            )
        ),
    )

    qlimits = (
        first_env(
            planning_robot.get_qlimits()
        )
        .astype(float)
    )

    qlimits_arm = qlimits[
        planning_ctx[
            "arm_indices"
        ]
    ]

    current_arm = physical_full_qpos[
        planning_ctx[
            "arm_indices"
        ]
    ].copy()

    ik_cfg = config[
        "ik"
    ]

    path = plan_cartesian_pose_segment(
        robot=planning_robot,
        device=planning_base.device,
        left_links=planning_base.left_contact_links,
        right_links=planning_base.right_contact_links,
        tcp_link=planning_base.agent.tcp,
        template_qpos=physical_full_qpos,
        arm_indices=planning_ctx[
            "arm_indices"
        ],
        qlimits_arm=qlimits_arm,
        start_arm_qpos=current_arm,
        target_position=(
            target_grasp_center
        ),
        target_quaternion=np.asarray(
            target_tcp_quaternion,
            dtype=np.float64,
        ),
        waypoint_count=waypoint_count,
        epsilon=float(
            ik_cfg[
                "epsilon_rad"
            ]
        ),
        damping=float(
            ik_cfg[
                "damping"
            ]
        ),
        position_tolerance=float(
            ik_cfg[
                "tolerance_m"
            ]
        ),
        orientation_tolerance_rad=(
            orientation_tolerance_rad
        ),
        orientation_weight_m_per_rad=(
            orientation_weight_m_per_rad
        ),
        max_iterations=int(
            ik_cfg[
                "max_iterations"
            ]
        ),
        max_step=float(
            ik_cfg[
                "max_step_rad"
            ]
        ),
    )

    return {
        "path":
            path,
        "physical_start_center_m":
            physical_center,
        "shadow_start_center_m":
            shadow_center,
        "center_sync_error_m":
            center_sync_error_m,
        "target_center_m":
            target_grasp_center.copy(),
        "distance_m":
            distance_m,
        "waypoint_count":
            waypoint_count,
    }


def execute_strict_expert_segment(
    *,
    expert_id: int,
    planning_env,
    physical_env,
    physical_ctx,
    config,
    grasp_command,
    pre_contact,
    close_action,
    contact_threshold,
    adaptive_low_force,
    adaptive_high_force,
    adaptive_command_step,
    target_tcp_quaternion,
    orientation_tolerance_rad,
    orientation_weight_m_per_rad,
    render,
    sleep,
):
    """
    Execute strict Expert 06 after the successful yaw-aware micro-lift.

    Logical trajectory semantics are Yaginuma-original.
    UR3e execution adapter differences:
    - Expert sparse anchors are preserved.
    - each anchor-to-anchor move is interpolated at <= 1 mm for UR3e IK.
    - yaw-aware grasp orientation is preserved.
    - gripper force-band adaptation remains active.
    """
    physical_base = (
        physical_env.unwrapped
    )

    pose9_start = pose9_numpy(
        physical_base
    )

    current_logical_xz = np.asarray(
        pose9_start[:2],
        dtype=np.float64,
    )

    A_start_world = object_position(
        physical_base.object_A
    )

    grasp_center_start = np.asarray(
        get_grasp_center(
            physical_base.left_contact_links,
            physical_base.right_contact_links,
        ),
        dtype=np.float64,
    )

    # Preserve the measured object-A -> grasp-center transform.
    grasp_center_minus_A = (
        grasp_center_start
        - A_start_world
    )

    strict = build_strict_expert_segment(
        expert_id,
        current_logical_xz,
    )

    logical_anchors = strict[
        "logical_anchors"
    ]
    anchor_indices = strict[
        "anchor_indices"
    ]

    # Use the measured A center as the object-map reference, not the TCP.
    # This maps logical +x -> world +X and logical +z -> world -Y using the
    # already validated ExpertCoordinateMapper scale/direction contract.
    object_mapper = ExpertCoordinateMapper(
        logical_reference=(
            current_logical_xz
        ),
        tcp_reference=(
            A_start_world
        ),
    )

    object_world_targets = np.stack(
        [
            object_mapper.logical_to_tcp(
                logical
            )
            for logical in logical_anchors
        ],
        axis=0,
    )

    # Keep the lifted object at its current world-Z height.
    object_world_targets[:, 2] = (
        A_start_world[2]
    )

    grasp_center_targets = (
        object_world_targets
        + grasp_center_minus_A[
            None,
            :
        ]
    )

    grasp_cfg = config[
        "grasp"
    ]

    expert_control_steps = int(
        grasp_cfg[
            "micro_lift_control_steps_per_waypoint"
        ]
    )

    max_force_limit = float(
        grasp_cfg[
            "max_contact_force_n"
        ]
    )

    rows = []
    total_planned_q_waypoints = 0
    total_env_steps = 0
    max_left_force = 0.0
    max_right_force = 0.0
    first_contact_loss = None
    first_task_success_event = None
    task_success_event_count = 0
    final_left_force = 0.0
    final_right_force = 0.0

    print(
        f"\n--- Phase 5D: strict Expert {expert_id:02d} execution ---"
    )
    print(
        "measured A logical start [x,z]:",
        current_logical_xz,
    )
    print(
        "strict begin length:",
        strict[
            "begin"
        ].corrected_length,
    )
    print(
        "strict end length:",
        strict[
            "end"
        ].corrected_length,
    )
    print(
        "zero residual:",
        PHASE5C_ZERO_RESIDUAL,
    )
    print(
        "Yaginuma execution anchor indices:",
        anchor_indices,
    )
    print(
        f"Expert {expert_id:02d} physical anchor transitions:",
        len(
            logical_anchors
        ) - 1,
    )
    print(
        "A->grasp-center offset [mm]:",
        grasp_center_minus_A
        * 1000.0,
    )

    # Anchor 0 represents the measured current state.
    # Execute anchor 1 ... final.
    for anchor_number in range(
        1,
        len(logical_anchors),
    ):
        target_logical = (
            logical_anchors[
                anchor_number
            ]
        )

        target_A_world = (
            object_world_targets[
                anchor_number
            ]
        )

        target_center = (
            grasp_center_targets[
                anchor_number
            ]
        )

        segment_plan = (
            plan_expert_anchor_from_actual_state(
                planning_env=planning_env,
                physical_env=physical_env,
                physical_ctx=physical_ctx,
                config=config,
                target_grasp_center=(
                    target_center
                ),
                target_tcp_quaternion=(
                    target_tcp_quaternion
                ),
                orientation_tolerance_rad=(
                    orientation_tolerance_rad
                ),
                orientation_weight_m_per_rad=(
                    orientation_weight_m_per_rad
                ),
            )
        )

        q_path = (
            segment_plan[
                "path"
            ][
                "arm_qpos"
            ]
        )

        total_planned_q_waypoints += (
            len(
                q_path
            )
        )

        segment_first_loss = None

        for q_index, q_target in enumerate(
            q_path,
            start=1,
        ):
            last_info = None

            for _ in range(
                expert_control_steps
            ):
                (
                    _obs,
                    _reward,
                    terminated,
                    truncated,
                    last_info,
                ) = step_robot(
                    physical_env,
                    physical_ctx,
                    q_target,
                    grasp_command,
                    render=render,
                    sleep=sleep,
                )

                total_env_steps += 1

                left = scalar(
                    last_info[
                        "left_contact_force_A"
                    ]
                )
                right = scalar(
                    last_info[
                        "right_contact_force_A"
                    ]
                )

                final_left_force = (
                    left
                )
                final_right_force = (
                    right
                )

                max_left_force = max(
                    max_left_force,
                    left,
                )
                max_right_force = max(
                    max_right_force,
                    right,
                )

                contact_ok = (
                    left
                    >= contact_threshold
                    and right
                    >= contact_threshold
                )

                if (
                    not contact_ok
                    and first_contact_loss
                    is None
                ):
                    first_contact_loss = {
                        "anchor_number":
                            int(
                                anchor_number
                            ),
                        "q_waypoint":
                            int(
                                q_index
                            ),
                    }

                if (
                    not contact_ok
                    and segment_first_loss
                    is None
                ):
                    segment_first_loss = int(
                        q_index
                    )

                grasp_command, _mode = (
                    update_adaptive_gripper_command(
                        grasp_command,
                        left,
                        right,
                        low_force_n=(
                            adaptive_low_force
                        ),
                        high_force_n=(
                            adaptive_high_force
                        ),
                        command_step=(
                            adaptive_command_step
                        ),
                        min_command=(
                            pre_contact
                        ),
                        max_command=(
                            close_action
                        ),
                    )
                )

                if as_bool(truncated):
                    raise RuntimeError(
                        f"Unexpected truncation during Expert {expert_id:02d}."
                    )

                if as_bool(terminated):
                    # UR3eAlignment-v0 evaluate() defines only "success" as
                    # a termination criterion.  During one Yaginuma Expert
                    # segment, success must be treated as an observed event,
                    # not as an immediate low-level stop: the original code
                    # evaluates done only after simulating the whole Expert.
                    success_now = (
                        "success" in last_info
                        and as_bool(
                            last_info[
                                "success"
                            ]
                        )
                    )

                    if not success_now:
                        raise RuntimeError(
                            "Environment terminated without "
                            f"success during Expert {expert_id:02d}."
                        )

                    task_success_event_count += 1

                    if first_task_success_event is None:
                        first_task_success_event = {
                            "anchor_number":
                                int(
                                    anchor_number
                                ),
                            "source_index":
                                int(
                                    anchor_indices[
                                        anchor_number
                                    ]
                                ),
                            "q_waypoint":
                                int(
                                    q_index
                                ),
                            "total_env_steps":
                                int(
                                    total_env_steps
                                ),
                        }

                        print(
                            f"Expert{expert_id:02d} task-success "
                            "observed inside segment; "
                            "continuing to the Expert endpoint "
                            "for Yaginuma source fidelity."
                        )

        measured_A_world = object_position(
            physical_base.object_A
        )

        measured_pose9 = pose9_numpy(
            physical_base
        )

        measured_logical = np.asarray(
            measured_pose9[:2],
            dtype=np.float64,
        )

        logical_error_m = float(
            np.linalg.norm(
                measured_logical
                - target_logical
            )
        )

        measured_center = np.asarray(
            get_grasp_center(
                physical_base.left_contact_links,
                physical_base.right_contact_links,
            ),
            dtype=np.float64,
        )

        rows.append(
            {
                "anchor_number":
                    int(
                        anchor_number
                    ),
                "source_index":
                    int(
                        anchor_indices[
                            anchor_number
                        ]
                    ),
                "target_logical_xz":
                    target_logical.tolist(),
                "measured_logical_xz":
                    measured_logical.tolist(),
                "logical_error_mm":
                    logical_error_m
                    * 1000.0,
                "target_A_world_m":
                    target_A_world.tolist(),
                "measured_A_world_m":
                    measured_A_world.tolist(),
                "target_grasp_center_m":
                    target_center.tolist(),
                "measured_grasp_center_m":
                    measured_center.tolist(),
                "planned_q_waypoints":
                    int(
                        len(
                            q_path
                        )
                    ),
                "shadow_sync_error_mm":
                    float(
                        segment_plan[
                            "center_sync_error_m"
                        ]
                        * 1000.0
                    ),
                "first_contact_loss_q_waypoint":
                    segment_first_loss,
                "left_force_N":
                    float(
                        final_left_force
                    ),
                "right_force_N":
                    float(
                        final_right_force
                    ),
                "grasp_command":
                    float(
                        grasp_command
                    ),
            }
        )

        print(
            f"Expert{expert_id:02d} anchor "
            f"{anchor_number:02d}/"
            f"{len(logical_anchors)-1:02d} | "
            f"src={anchor_indices[anchor_number]:03d} | "
            f"q_wp={len(q_path):03d} | "
            f"A_err={logical_error_m*1000.0:.3f} mm | "
            f"L={final_left_force:.3f} N "
            f"R={final_right_force:.3f} N | "
            f"grip={grasp_command:.4f}"
        )

    pose9_end = pose9_numpy(
        physical_base
    )

    endpoint_eval = (
        physical_base.evaluate()
    )

    task_success_at_end = as_bool(
        endpoint_eval[
            "success"
        ]
    )

    final_logical = np.asarray(
        pose9_end[:2],
        dtype=np.float64,
    )

    target_final_logical = (
        logical_anchors[-1]
    )

    endpoint_error_m = float(
        np.linalg.norm(
            final_logical
            - target_final_logical
        )
    )

    force_gate = (
        max_left_force
        <= max_force_limit
        and max_right_force
        <= max_force_limit
    )

    contact_gate = (
        first_contact_loss is None
    )

    endpoint_gate = (
        endpoint_error_m
        <= EXPERT_ENDPOINT_ERROR_LIMIT_M
    )

    success = bool(
        force_gate
        and contact_gate
        and endpoint_gate
    )

    failure_reasons = []

    if not force_gate:
        failure_reasons.append(
            "expert_contact_force"
        )

    if not contact_gate:
        failure_reasons.append(
            "expert_bilateral_contact_loss"
        )

    if not endpoint_gate:
        failure_reasons.append(
            "expert_endpoint_error"
        )

    print(
        f"\n=== Expert {expert_id:02d} strict result ==="
    )
    print(
        "target final logical [x,z]:",
        target_final_logical,
    )
    print(
        "measured final logical [x,z]:",
        final_logical,
    )
    print(
        "logical endpoint error [mm]:",
        endpoint_error_m * 1000.0,
    )
    print(
        "max Expert left force [N]:",
        max_left_force,
    )
    print(
        "max Expert right force [N]:",
        max_right_force,
    )
    print(
        "first Expert contact loss:",
        first_contact_loss,
    )
    print(
        "first task-success event inside segment:",
        first_task_success_event,
    )
    print(
        "task-success event count inside segment:",
        task_success_event_count,
    )
    print(
        "task success at Expert endpoint:",
        task_success_at_end,
    )
    print(
        f"Expert {expert_id:02d} strict:",
        "PASS" if success else "FAIL",
    )

    return {
        "success":
            success,
        "failure_reasons":
            failure_reasons,
        "expert_id":
            int(expert_id),
        "ddpg_action":
            PHASE5C_ZERO_RESIDUAL.tolist(),
        "measured_start_logical_xz":
            current_logical_xz.tolist(),
        "begin_corrected_length":
            int(
                strict[
                    "begin"
                ].corrected_length
            ),
        "end_corrected_length":
            int(
                strict[
                    "end"
                ].corrected_length
            ),
        "end_sigma_scale":
            float(
                strict[
                    "end"
                ].sigma_scale
            ),
        "anchor_indices":
            anchor_indices.tolist(),
        "num_execution_anchors":
            int(
                len(
                    anchor_indices
                )
            ),
        "num_anchor_transitions":
            int(
                len(
                    anchor_indices
                ) - 1
            ),
        "grasp_center_minus_A_start_mm":
            (
                grasp_center_minus_A
                * 1000.0
            ).tolist(),
        "target_final_logical_xz":
            target_final_logical.tolist(),
        "measured_final_logical_xz":
            final_logical.tolist(),
        "endpoint_error_mm":
            endpoint_error_m
            * 1000.0,
        "endpoint_error_limit_mm":
            EXPERT_ENDPOINT_ERROR_LIMIT_M
            * 1000.0,
        "max_left_contact_force_N":
            float(
                max_left_force
            ),
        "max_right_contact_force_N":
            float(
                max_right_force
            ),
        "final_left_contact_force_N":
            float(
                final_left_force
            ),
        "final_right_contact_force_N":
            float(
                final_right_force
            ),
        "first_bilateral_contact_loss":
            first_contact_loss,
        "first_task_success_event":
            first_task_success_event,
        "task_success_event_count":
            int(
                task_success_event_count
            ),
        "task_success_at_end":
            bool(
                task_success_at_end
            ),
        "total_planned_q_waypoints":
            int(
                total_planned_q_waypoints
            ),
        "total_env_steps":
            int(
                total_env_steps
            ),
        "final_grasp_command":
            float(
                grasp_command
            ),
        "rows":
            rows,
    }



def execute_fixed_expert_sequence(
    *,
    planning_env,
    physical_env,
    physical_ctx,
    config,
    grasp_command,
    pre_contact,
    close_action,
    contact_threshold,
    adaptive_low_force,
    adaptive_high_force,
    adaptive_command_step,
    target_tcp_quaternion,
    orientation_tolerance_rad,
    orientation_weight_m_per_rad,
    render,
    sleep,
):
    """
    Execute Yaginuma's fixed nominal sequence [6,4,3].

    Critical fidelity rule:
    each segment begins from the ACTUAL measured A logical [x,z] after the
    previous segment, matching residual_ddpg.py's self.initial_pos update.
    """
    segment_results = []
    current_grasp_command = float(
        grasp_command
    )
    sequence_execution_ok = True
    task_success = False
    source_done_reason = None
    failure_reasons = []

    print(
        "\n=== Phase 5D fixed nominal sequence ==="
    )
    print(
        "Expert sequence:",
        list(
            PHASE5D_EXPERT_SEQUENCE
        ),
    )
    print(
        "Residual for every segment:",
        PHASE5C_ZERO_RESIDUAL,
    )

    for sequence_index, expert_id in enumerate(
        PHASE5D_EXPERT_SEQUENCE,
        start=1,
    ):
        base = physical_env.unwrapped

        measured_before_pose9 = pose9_numpy(
            base
        )
        measured_before_logical = np.asarray(
            measured_before_pose9[:2],
            dtype=np.float64,
        )

        print(
            f"\nSequence {sequence_index}/"
            f"{len(PHASE5D_EXPERT_SEQUENCE)}: "
            f"Expert {expert_id:02d}"
        )
        print(
            "measured A logical before segment:",
            measured_before_logical,
        )

        segment = (
            execute_strict_expert_segment(
                expert_id=expert_id,
                planning_env=planning_env,
                physical_env=physical_env,
                physical_ctx=physical_ctx,
                config=config,
                grasp_command=(
                    current_grasp_command
                ),
                pre_contact=pre_contact,
                close_action=close_action,
                contact_threshold=(
                    contact_threshold
                ),
                adaptive_low_force=(
                    adaptive_low_force
                ),
                adaptive_high_force=(
                    adaptive_high_force
                ),
                adaptive_command_step=(
                    adaptive_command_step
                ),
                target_tcp_quaternion=(
                    target_tcp_quaternion
                ),
                orientation_tolerance_rad=(
                    orientation_tolerance_rad
                ),
                orientation_weight_m_per_rad=(
                    orientation_weight_m_per_rad
                ),
                render=render,
                sleep=sleep,
            )
        )

        measured_after_pose9 = pose9_numpy(
            base
        )
        measured_after_logical = np.asarray(
            measured_after_pose9[:2],
            dtype=np.float64,
        )

        segment[
            "sequence_index"
        ] = int(
            sequence_index
        )
        segment[
            "measured_before_logical_xz"
        ] = (
            measured_before_logical.tolist()
        )
        segment[
            "measured_after_logical_xz"
        ] = (
            measured_after_logical.tolist()
        )

        segment_results.append(
            segment
        )

        current_grasp_command = float(
            segment[
                "final_grasp_command"
            ]
        )

        if not segment[
            "success"
        ]:
            sequence_execution_ok = False
            source_done_reason = (
                f"expert_{expert_id:02d}_execution_failure"
            )
            failure_reasons.extend(
                [
                    f"expert_{expert_id:02d}:"
                    f"{reason}"
                    for reason in segment[
                        "failure_reasons"
                    ]
                ]
            )
            print(
                f"Stopping sequence because "
                f"Expert {expert_id:02d} failed its "
                "physical execution gates."
            )
            break

        # Yaginuma source semantics:
        # reward/done is evaluated only after a whole Expert segment.
        if segment[
            "task_success_at_end"
        ]:
            task_success = True
            source_done_reason = (
                f"success_after_expert_{expert_id:02d}"
            )
            print(
                "Yaginuma-style segment-end success: "
                f"stop after Expert {expert_id:02d}."
            )
            break

        # If the final fixed Expert completed without task success, the
        # source would end because the maximum segment count was reached.
        if (
            sequence_index
            == len(
                PHASE5D_EXPERT_SEQUENCE
            )
        ):
            source_done_reason = (
                "max_segments_without_task_success"
            )
            break

        # Explicitly show the measured feedback state that seeds the next
        # Expert begin correction.
        print(
            "feedback A logical for next Expert:",
            measured_after_logical,
        )

    final_pose9 = pose9_numpy(
        physical_env.unwrapped
    )

    print(
        "\n=== Fixed sequence result ==="
    )
    print(
        "segments completed:",
        len(
            segment_results
        ),
        "/",
        len(
            PHASE5D_EXPERT_SEQUENCE
        ),
    )
    print(
        "final pose9:",
        final_pose9,
    )
    phase5d_pass = bool(
        sequence_execution_ok
        and (
            task_success
            or len(
                segment_results
            )
            == len(
                PHASE5D_EXPERT_SEQUENCE
            )
        )
    )

    print(
        "task success at segment boundary:",
        task_success,
    )
    print(
        "source done reason:",
        source_done_reason,
    )
    print(
        "fixed [6,4,3] nominal execution:",
        "PASS"
        if phase5d_pass
        else "FAIL",
    )

    return {
        "success":
            phase5d_pass,
        "execution_success":
            bool(
                sequence_execution_ok
            ),
        "task_success":
            bool(
                task_success
            ),
        "source_done_reason":
            source_done_reason,
        "failure_reasons":
            failure_reasons,
        "expert_sequence":
            list(
                PHASE5D_EXPERT_SEQUENCE
            ),
        "ddpg_action_each_segment":
            PHASE5C_ZERO_RESIDUAL.tolist(),
        "segments_completed":
            int(
                len(
                    segment_results
                )
            ),
        "segments":
            segment_results,
        "final_pose9":
            final_pose9.tolist(),
        "final_grasp_command":
            float(
                current_grasp_command
            ),
    }


def main():
    args = parse_args()

    if args.output is None:
        args.output = (
            PROJECT_ROOT
            / "reports"
            / "expert_gp"
            / "alignment"
            / (
                "grasp_A_fixed_643_strict_seed"
                f"{int(args.seed)}.json"
            )
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(
            f
        )

    if args.max_episode_steps < 1000:
        raise ValueError(
            "--max-episode-steps must be >= 1000."
        )

    planning_env = make_env(
        args.max_episode_steps
    )
    physical_env = make_env(
        args.max_episode_steps
    )

    try:
        print(
            "=== Phase 5B-2 Alignment A Grasp Smoke Test ==="
        )
        print(
            "EXPERT [6,4,3] / ZERO RESIDUAL / NO ROS2 / NO REAL ROBOT COMMAND"
        )
        print(
            "Gym max episode steps:",
            int(
                args.max_episode_steps
            ),
        )

        orientation_tolerance_rad = (
            np.deg2rad(
                float(
                    args.orientation_tolerance_deg
                )
            )
        )

        plan = plan_paths(
            planning_env,
            config,
            seed=args.seed,
            orientation_tolerance_rad=(
                orientation_tolerance_rad
            ),
            orientation_weight_m_per_rad=float(
                args.orientation_weight_m_per_rad
            ),
        )

        physical_env.reset(
            seed=int(
                args.seed
            )
        )

        base = (
            physical_env.unwrapped
        )
        ctx = build_context(
            physical_env,
            config,
        )

        gripper_cfg = config[
            "gripper"
        ]
        grasp_cfg = config[
            "grasp"
        ]
        adaptive_cfg = config[
            "adaptive_gripper"
        ]

        adaptive_low_force = float(
            adaptive_cfg[
                "low_force_n"
            ]
        )
        adaptive_high_force = float(
            adaptive_cfg[
                "high_force_n"
            ]
        )
        adaptive_command_step = float(
            adaptive_cfg[
                "command_step"
            ]
        )
        adaptive_settle_steps = int(
            adaptive_cfg[
                "settle_steps"
            ]
        )

        pre_contact = float(
            gripper_cfg[
                "pre_contact_action"
            ]
        )
        close_action = float(
            gripper_cfg[
                "close_action"
            ]
        )
        contact_threshold = float(
            gripper_cfg[
                "contact_force_threshold_n"
            ]
        )
        stable_force = float(
            grasp_cfg[
                "stable_grasp_force_n"
            ]
        )

        hold_precontact(
            physical_env,
            ctx,
            pre_contact=pre_contact,
            steps=int(
                grasp_cfg[
                    "reset_precontact_steps"
                ]
            ),
            render=args.render,
            sleep=args.sleep,
        )

        A_initial = object_position(
            base.object_A
        )
        pose9_initial = pose9_numpy(
            base
        )

        plan_alignment_error_mm = float(
            np.linalg.norm(
                A_initial
                - plan["A_world"]
            )
            * 1000.0
        )

        print(
            "\nphysical A world position:",
            A_initial,
        )
        print(
            "planning/physical A mismatch [mm]:",
            plan_alignment_error_mm,
        )

        if (
            plan_alignment_error_mm
            > 0.1
        ):
            raise RuntimeError(
                "Planning and physical env reset do not match."
            )

        control_steps = int(
            grasp_cfg[
                "control_steps_per_waypoint"
            ]
        )

        print(
            "\n--- Physical approach: home -> pregrasp ---"
        )

        run_path(
            physical_env,
            ctx,
            plan[
                "to_pregrasp"
            ][
                "arm_qpos"
            ],
            pre_contact,
            control_steps=control_steps,
            render=args.render,
            sleep=args.sleep,
            verbose=False,
            phase_name="to_pregrasp",
        )

        q_pregrasp = np.asarray(
            plan[
                "to_pregrasp"
            ][
                "final_arm_qpos"
            ],
            dtype=float,
        )

        for _ in range(
            int(
                grasp_cfg[
                    "pregrasp_hold_steps"
                ]
            )
        ):
            step_robot(
                physical_env,
                ctx,
                q_pregrasp,
                pre_contact,
                render=args.render,
                sleep=args.sleep,
            )

        print(
            "--- Physical descend: pregrasp -> grasp ---"
        )

        run_path(
            physical_env,
            ctx,
            plan[
                "descend"
            ][
                "arm_qpos"
            ],
            pre_contact,
            control_steps=control_steps,
            render=args.render,
            sleep=args.sleep,
            verbose=False,
            phase_name="descend",
        )

        q_grasp = np.asarray(
            plan[
                "descend"
            ][
                "final_arm_qpos"
            ],
            dtype=float,
        )

        print(
            "--- Gripper ramp ---"
        )

        bilateral_streak = 0
        detected_command = None
        max_left_force = 0.0
        max_right_force = 0.0

        for command in np.linspace(
            pre_contact,
            close_action,
            int(
                grasp_cfg[
                    "grasp_ramp_steps"
                ]
            ),
        ):
            (
                _obs,
                _reward,
                terminated,
                truncated,
                info,
            ) = step_robot(
                physical_env,
                ctx,
                q_grasp,
                float(command),
                render=args.render,
                sleep=args.sleep,
            )

            if as_bool(terminated):
                raise RuntimeError(
                    "Unexpected termination during grasp ramp."
                )
            if as_bool(truncated):
                raise RuntimeError(
                    "Unexpected truncation during grasp ramp."
                )

            left = scalar(
                info[
                    "left_contact_force_A"
                ]
            )
            right = scalar(
                info[
                    "right_contact_force_A"
                ]
            )

            max_left_force = max(
                max_left_force,
                left,
            )
            max_right_force = max(
                max_right_force,
                right,
            )

            if (
                left
                >= contact_threshold
                and right
                >= contact_threshold
            ):
                bilateral_streak += 1
            else:
                bilateral_streak = 0

            if (
                bilateral_streak
                >= int(
                    grasp_cfg[
                        "bilateral_contact_streak"
                    ]
                )
            ):
                detected_command = float(
                    command
                )
                break

        if detected_command is None:
            raise RuntimeError(
                "Bilateral contact was not detected."
            )

        print(
            "bilateral contact command:",
            detected_command,
        )

        print(
            "adaptive force band [N]:",
            f"[{adaptive_low_force:.3f}, "
            f"{adaptive_high_force:.3f}]",
        )
        print(
            "adaptive command step:",
            adaptive_command_step,
        )

        # Start from the first reliably detected bilateral-contact command.
        # Do not add the old pose-independent fixed +0.05 squeeze.
        grasp_command = float(
            detected_command
        )

        adaptive_mode_counts = {
            "open_reduce_force": 0,
            "close_recover_force": 0,
            "hold_force_band": 0,
        }

        # --------------------------------------------------
        # Closed-loop settle before formal stability scoring.
        # --------------------------------------------------
        print(
            "adaptive settle steps:",
            adaptive_settle_steps,
        )

        for _ in range(
            adaptive_settle_steps
        ):
            (
                _obs,
                _reward,
                _terminated,
                _truncated,
                info,
            ) = step_robot(
                physical_env,
                ctx,
                q_grasp,
                grasp_command,
                render=args.render,
                sleep=args.sleep,
            )

            left = scalar(
                info[
                    "left_contact_force_A"
                ]
            )
            right = scalar(
                info[
                    "right_contact_force_A"
                ]
            )

            max_left_force = max(
                max_left_force,
                left,
            )
            max_right_force = max(
                max_right_force,
                right,
            )

            grasp_command, adaptive_mode = (
                update_adaptive_gripper_command(
                    grasp_command,
                    left,
                    right,
                    low_force_n=adaptive_low_force,
                    high_force_n=adaptive_high_force,
                    command_step=adaptive_command_step,
                    min_command=pre_contact,
                    max_command=close_action,
                )
            )

            adaptive_mode_counts[
                adaptive_mode
            ] += 1

        print(
            "adaptive settled grasp command:",
            grasp_command,
        )

        # --------------------------------------------------
        # Formal pre-lift stability gate, still adaptive.
        # --------------------------------------------------
        stable_count = 0
        hold_steps = int(
            grasp_cfg[
                "pre_lift_hold_steps"
            ]
        )

        for _ in range(
            hold_steps
        ):
            (
                _obs,
                _reward,
                _terminated,
                _truncated,
                info,
            ) = step_robot(
                physical_env,
                ctx,
                q_grasp,
                grasp_command,
                render=args.render,
                sleep=args.sleep,
            )

            left = scalar(
                info[
                    "left_contact_force_A"
                ]
            )
            right = scalar(
                info[
                    "right_contact_force_A"
                ]
            )

            max_left_force = max(
                max_left_force,
                left,
            )
            max_right_force = max(
                max_right_force,
                right,
            )

            if (
                left >= stable_force
                and right >= stable_force
            ):
                stable_count += 1

            grasp_command, adaptive_mode = (
                update_adaptive_gripper_command(
                    grasp_command,
                    left,
                    right,
                    low_force_n=adaptive_low_force,
                    high_force_n=adaptive_high_force,
                    command_step=adaptive_command_step,
                    min_command=pre_contact,
                    max_command=close_action,
                )
            )

            adaptive_mode_counts[
                adaptive_mode
            ] += 1

        required_stable = int(
            np.ceil(
                hold_steps
                * float(
                    grasp_cfg[
                        "pre_lift_required_contact_ratio"
                    ]
                )
            )
        )

        print(
            "stable bilateral steps:",
            f"{stable_count}/{hold_steps}",
            "(required:",
            f"{required_stable})",
        )

        if (
            stable_count
            < required_stable
        ):
            raise RuntimeError(
                "Stable grasp force gate failed."
            )

        A_before_lift = object_position(
            base.object_A
        )

        print(
            "--- Relative micro-lift replanning from actual grasp state ---"
        )

        actual_lift_plan = (
            replan_micro_lift_from_actual_state(
                planning_env=planning_env,
                physical_env=physical_env,
                physical_ctx=ctx,
                config=config,
                target_tcp_quaternion=np.asarray(
                    plan[
                        "grasp_tcp_quaternion"
                    ],
                    dtype=np.float64,
                ),
                orientation_tolerance_rad=(
                    orientation_tolerance_rad
                ),
                orientation_weight_m_per_rad=float(
                    args.orientation_weight_m_per_rad
                ),
            )
        )

        print(
            "physical start grasp center [m]:",
            actual_lift_plan[
                "physical_start_center_m"
            ],
        )
        print(
            "shadow start grasp center [m]:",
            actual_lift_plan[
                "shadow_start_center_m"
            ],
        )
        print(
            "shadow/physical sync error [mm]:",
            actual_lift_plan[
                "center_sync_error_m"
            ] * 1000.0,
        )
        print(
            "relative lift request [mm]:",
            actual_lift_plan[
                "relative_lift_m"
            ] * 1000.0,
        )
        print(
            "actual-state lift target [m]:",
            actual_lift_plan[
                "target_center_m"
            ],
        )
        print(
            "replanned zero-like q transitions:",
            actual_lift_plan[
                "zero_like_transitions"
            ],
            "/",
            len(
                actual_lift_plan[
                    "lift"
                ][
                    "arm_qpos"
                ]
            ),
        )
        print(
            "replanned max |dq| [rad]:",
            actual_lift_plan[
                "max_abs_dq_rad"
            ],
        )

        micro_lift_control_steps = int(
            grasp_cfg[
                "micro_lift_control_steps_per_waypoint"
            ]
        )

        print(
            "micro-lift physical control steps/waypoint:",
            micro_lift_control_steps,
        )

        grasp_center_before_lift = np.asarray(
            get_grasp_center(
                base.left_contact_links,
                base.right_contact_links,
            ),
            dtype=np.float64,
        )

        tcp_q_before_lift = (
            first_env(
                base.agent.tcp.pose.q
            )
            .astype(np.float64)
            .copy()
        )

        lift_trace = []
        first_contact_loss_waypoint = None
        max_A_z = float(A_before_lift[2])
        max_grasp_center_z = float(
            grasp_center_before_lift[2]
        )

        q_path = actual_lift_plan["lift"]["arm_qpos"]

        for waypoint_index, q_target in enumerate(
            q_path,
            start=1,
        ):
            last_lift_info = None

            for _ in range(
                micro_lift_control_steps
            ):
                (
                    _obs,
                    _reward,
                    terminated,
                    truncated,
                    last_lift_info,
                ) = step_robot(
                    physical_env,
                    ctx,
                    q_target,
                    grasp_command,
                    render=args.render,
                    sleep=args.sleep,
                )

                lift_left = scalar(
                    last_lift_info[
                        "left_contact_force_A"
                    ]
                )
                lift_right = scalar(
                    last_lift_info[
                        "right_contact_force_A"
                    ]
                )

                max_left_force = max(
                    max_left_force,
                    lift_left,
                )
                max_right_force = max(
                    max_right_force,
                    lift_right,
                )

                grasp_command, adaptive_mode = (
                    update_adaptive_gripper_command(
                        grasp_command,
                        lift_left,
                        lift_right,
                        low_force_n=adaptive_low_force,
                        high_force_n=adaptive_high_force,
                        command_step=adaptive_command_step,
                        min_command=pre_contact,
                        max_command=close_action,
                    )
                )

                adaptive_mode_counts[
                    adaptive_mode
                ] += 1

                if as_bool(terminated):
                    raise RuntimeError(
                        "Unexpected termination during micro-lift."
                    )

                if as_bool(truncated):
                    raise RuntimeError(
                        "Unexpected truncation during micro-lift."
                    )

            center = np.asarray(
                get_grasp_center(
                    base.left_contact_links,
                    base.right_contact_links,
                ),
                dtype=np.float64,
            )

            A_now = object_position(
                base.object_A
            )

            tcp_q_now = (
                first_env(
                    base.agent.tcp.pose.q
                )
                .astype(np.float64)
                .copy()
            )

            current_arm = (
                first_env(
                    ctx["robot"]
                    .get_qpos()
                )
                .astype(float)[
                    ctx["arm_indices"]
                ]
            )

            target_arm = np.asarray(
                q_target,
                dtype=np.float64,
            )

            arm_tracking_error_max = float(
                np.max(
                    np.abs(
                        target_arm
                        - current_arm
                    )
                )
            )

            left = scalar(
                last_lift_info[
                    "left_contact_force_A"
                ]
            )
            right = scalar(
                last_lift_info[
                    "right_contact_force_A"
                ]
            )

            contact_ok = (
                left >= contact_threshold
                and right >= contact_threshold
            )

            if (
                not contact_ok
                and first_contact_loss_waypoint is None
            ):
                first_contact_loss_waypoint = waypoint_index

            relative = (
                A_now
                - center
            )

            orientation_drift = (
                quaternion_angle_deg(
                    tcp_q_before_lift,
                    tcp_q_now,
                )
            )

            max_A_z = max(
                max_A_z,
                float(A_now[2]),
            )
            max_grasp_center_z = max(
                max_grasp_center_z,
                float(center[2]),
            )

            row = {
                "waypoint":
                    waypoint_index,
                "grasp_center_m":
                    center.tolist(),
                "A_world_m":
                    A_now.tolist(),
                "A_minus_grasp_center_m":
                    relative.tolist(),
                "left_force_N":
                    float(left),
                "right_force_N":
                    float(right),
                "bilateral_contact":
                    bool(contact_ok),
                "tcp_orientation_drift_deg":
                    orientation_drift,
                "arm_tracking_error_max_rad":
                    arm_tracking_error_max,
                "adaptive_gripper_command":
                    float(
                        grasp_command
                    ),
            }

            lift_trace.append(
                row
            )

            print(
                f"lift wp {waypoint_index:02d}/{len(q_path):02d} | "
                f"center_dz="
                f"{(center[2]-grasp_center_before_lift[2])*1000.0:+.3f} mm | "
                f"A_dz="
                f"{(A_now[2]-A_before_lift[2])*1000.0:+.3f} mm | "
                f"rel_norm="
                f"{np.linalg.norm(relative)*1000.0:.3f} mm | "
                f"L={left:.3f} N R={right:.3f} N | "
                f"grip_cmd={grasp_command:.4f} | "
                f"arm_err={arm_tracking_error_max:.6f} rad | "
                f"rot={orientation_drift:.3f} deg"
            )

        # Hold a few steps at the top, but preserve the diagnostic trace.
        q_lift = np.asarray(
            actual_lift_plan[
                "lift"
            ][
                "final_arm_qpos"
            ],
            dtype=float,
        )

        final_info = None

        for _ in range(10):
            (
                _obs,
                _reward,
                _terminated,
                _truncated,
                final_info,
            ) = step_robot(
                physical_env,
                ctx,
                q_lift,
                grasp_command,
                render=args.render,
                sleep=args.sleep,
            )

            final_hold_left = scalar(
                final_info[
                    "left_contact_force_A"
                ]
            )
            final_hold_right = scalar(
                final_info[
                    "right_contact_force_A"
                ]
            )

            max_left_force = max(
                max_left_force,
                final_hold_left,
            )
            max_right_force = max(
                max_right_force,
                final_hold_right,
            )

            grasp_command, adaptive_mode = (
                update_adaptive_gripper_command(
                    grasp_command,
                    final_hold_left,
                    final_hold_right,
                    low_force_n=adaptive_low_force,
                    high_force_n=adaptive_high_force,
                    command_step=adaptive_command_step,
                    min_command=pre_contact,
                    max_command=close_action,
                )
            )

            adaptive_mode_counts[
                adaptive_mode
            ] += 1

        grasp_center_final = np.asarray(
            get_grasp_center(
                base.left_contact_links,
                base.right_contact_links,
            ),
            dtype=np.float64,
        )

        tcp_q_final = (
            first_env(
                base.agent.tcp.pose.q
            )
            .astype(np.float64)
            .copy()
        )

        A_final = object_position(
            base.object_A
        )
        pose9_final = pose9_numpy(
            base
        )

        # Micro-lift success must be measured from the state immediately
        # before lift, not from episode reset.  Grasp/settling can move A
        # slightly before the lift begins.
        lift_m = float(
            A_final[2]
            - A_before_lift[2]
        )

        episode_lift_m = float(
            A_final[2]
            - A_initial[2]
        )

        xy_shift_m = float(
            np.linalg.norm(
                A_final[:2]
                - A_before_lift[:2]
            )
        )

        episode_xy_shift_m = float(
            np.linalg.norm(
                A_final[:2]
                - A_initial[:2]
            )
        )

        final_left = scalar(
            final_info[
                "left_contact_force_A"
            ]
        )
        final_right = scalar(
            final_info[
                "right_contact_force_A"
            ]
        )

        print(
            "\n=== Result ==="
        )
        print(
            "A initial world [m]:",
            A_initial,
        )
        print(
            "A before lift [m]:",
            A_before_lift,
        )
        print(
            "A final world [m]:",
            A_final,
        )
        print(
            "A micro-lift from pre-lift state [mm]:",
            lift_m * 1000.0,
        )
        print(
            "A episode-total lift [mm]:",
            episode_lift_m * 1000.0,
        )
        print(
            "A XY shift during lift [mm]:",
            xy_shift_m * 1000.0,
        )
        print(
            "A episode-total XY shift [mm]:",
            episode_xy_shift_m * 1000.0,
        )
        print(
            "max left force [N]:",
            max_left_force,
        )
        print(
            "max right force [N]:",
            max_right_force,
        )
        print(
            "final left force [N]:",
            final_left,
        )
        print(
            "final right force [N]:",
            final_right,
        )
        print(
            "grasp center lift [mm]:",
            (
                grasp_center_final[2]
                - grasp_center_before_lift[2]
            ) * 1000.0,
        )
        print(
            "max grasp center lift [mm]:",
            (
                max_grasp_center_z
                - grasp_center_before_lift[2]
            ) * 1000.0,
        )
        print(
            "max A lift during motion [mm]:",
            (
                max_A_z
                - A_before_lift[2]
            ) * 1000.0,
        )
        print(
            "first bilateral-contact loss waypoint:",
            first_contact_loss_waypoint,
        )
        print(
            "final TCP orientation drift [deg]:",
            quaternion_angle_deg(
                tcp_q_before_lift,
                tcp_q_final,
            ),
        )
        print(
            "final A-center relative [mm]:",
            (
                A_final
                - grasp_center_final
            ) * 1000.0,
        )
        print(
            "pose9 initial:",
            pose9_initial,
        )
        print(
            "pose9 final  :",
            pose9_final,
        )

        required_lift = float(
            grasp_cfg[
                "micro_lift_success_threshold_m"
            ]
        )

        max_force_limit = float(
            grasp_cfg[
                "max_contact_force_n"
            ]
        )

        lift_gate = (
            lift_m >= required_lift
        )

        force_gate = (
            max_left_force
            <= max_force_limit
            and max_right_force
            <= max_force_limit
        )

        success = bool(
            lift_gate
            and force_gate
            and first_contact_loss_waypoint is None
        )

        failure_reasons = []

        if not lift_gate:
            failure_reasons.append(
                "micro_lift"
            )

        if not force_gate:
            failure_reasons.append(
                "contact_force"
            )

        if first_contact_loss_waypoint is not None:
            failure_reasons.append(
                "bilateral_contact_loss"
            )

        # ----------------------------------------------
        # Phase 5C: execute strict Expert 06 only after
        # the validated Phase-5B-2 grasp gate passes.
        # ----------------------------------------------
        grasp_success = bool(
            success
        )

        fixed_sequence_result = None

        if grasp_success:
            fixed_sequence_result = (
                execute_fixed_expert_sequence(
                    planning_env=planning_env,
                    physical_env=physical_env,
                    physical_ctx=ctx,
                    config=config,
                    grasp_command=(
                        grasp_command
                    ),
                    pre_contact=(
                        pre_contact
                    ),
                    close_action=(
                        close_action
                    ),
                    contact_threshold=(
                        contact_threshold
                    ),
                    adaptive_low_force=(
                        adaptive_low_force
                    ),
                    adaptive_high_force=(
                        adaptive_high_force
                    ),
                    adaptive_command_step=(
                        adaptive_command_step
                    ),
                    target_tcp_quaternion=np.asarray(
                        plan[
                            "grasp_tcp_quaternion"
                        ],
                        dtype=np.float64,
                    ),
                    orientation_tolerance_rad=(
                        orientation_tolerance_rad
                    ),
                    orientation_weight_m_per_rad=float(
                        args.orientation_weight_m_per_rad
                    ),
                    render=args.render,
                    sleep=args.sleep,
                )
            )

            for segment_result in fixed_sequence_result[
                "segments"
            ]:
                max_left_force = max(
                    max_left_force,
                    float(
                        segment_result[
                            "max_left_contact_force_N"
                        ]
                    ),
                )
                max_right_force = max(
                    max_right_force,
                    float(
                        segment_result[
                            "max_right_contact_force_N"
                        ]
                    ),
                )

            if fixed_sequence_result[
                "segments"
            ]:
                final_segment = (
                    fixed_sequence_result[
                        "segments"
                    ][
                        -1
                    ]
                )
                final_left = float(
                    final_segment[
                        "final_left_contact_force_N"
                    ]
                )
                final_right = float(
                    final_segment[
                        "final_right_contact_force_N"
                    ]
                )

            if not fixed_sequence_result[
                "success"
            ]:
                failure_reasons.extend(
                    fixed_sequence_result[
                        "failure_reasons"
                    ]
                )

        else:
            print(
                "\nPhase 5D fixed sequence skipped because "
                "the grasp gate failed."
            )

        success = bool(
            grasp_success
            and fixed_sequence_result is not None
            and fixed_sequence_result[
                "success"
            ]
        )

        result = {
            "seed":
                int(args.seed),
            "yaw_aware_orientation": {
                "home_tcp_quaternion_wxyz":
                    plan[
                        "home_tcp_quaternion"
                    ],
                "target_tcp_quaternion_wxyz":
                    plan[
                        "grasp_tcp_quaternion"
                    ],
                "object_yaw_deg":
                    float(
                        np.degrees(
                            plan[
                                "object_yaw_rad"
                            ]
                        )
                    ),
                "closing_axis_yaw_deg":
                    float(
                        np.degrees(
                            plan[
                                "closing_axis_yaw_rad"
                            ]
                        )
                    ),
                "selected_face_normal_yaw_deg":
                    float(
                        np.degrees(
                            plan[
                                "selected_face_normal_yaw_rad"
                            ]
                        )
                    ),
                "yaw_correction_deg":
                    float(
                        np.degrees(
                            plan[
                                "yaw_correction_rad"
                            ]
                        )
                    ),
                "selected_face_index":
                    int(
                        plan[
                            "selected_face_index"
                        ]
                    ),
                "orientation_tolerance_deg":
                    float(
                        args.orientation_tolerance_deg
                    ),
                "orientation_weight_m_per_rad":
                    float(
                        args.orientation_weight_m_per_rad
                    ),
            },
            "success":
                success,
            "failure_reasons":
                failure_reasons,
            "A_initial_world_m":
                A_initial.tolist(),
            "A_before_lift_world_m":
                A_before_lift.tolist(),
            "A_final_world_m":
                A_final.tolist(),
            "A_lift_mm":
                lift_m * 1000.0,
            "A_micro_lift_from_pre_lift_mm":
                lift_m * 1000.0,
            "A_episode_total_lift_mm":
                episode_lift_m * 1000.0,
            "A_xy_shift_mm":
                xy_shift_m * 1000.0,
            "A_xy_shift_during_lift_mm":
                xy_shift_m * 1000.0,
            "A_episode_total_xy_shift_mm":
                episode_xy_shift_m * 1000.0,
            "pose9_initial":
                pose9_initial.tolist(),
            "pose9_final":
                pose9_final.tolist(),
            "planning_physical_A_mismatch_mm":
                plan_alignment_error_mm,
            "detected_gripper_command":
                detected_command,
            "grasp_command":
                grasp_command,
            "adaptive_gripper": {
                "low_force_n":
                    adaptive_low_force,
                "high_force_n":
                    adaptive_high_force,
                "command_step":
                    adaptive_command_step,
                "settle_steps":
                    adaptive_settle_steps,
                "mode_counts":
                    adaptive_mode_counts,
                "final_command":
                    float(
                        grasp_command
                    ),
            },
            "stable_bilateral_steps":
                stable_count,
            "stable_required_steps":
                required_stable,
            "max_left_contact_force_N":
                max_left_force,
            "max_right_contact_force_N":
                max_right_force,
            "final_left_contact_force_N":
                final_left,
            "final_right_contact_force_N":
                final_right,
            "grasp_center_before_lift_m":
                grasp_center_before_lift.tolist(),
            "grasp_center_final_m":
                grasp_center_final.tolist(),
            "grasp_center_lift_mm":
                float(
                    (
                        grasp_center_final[2]
                        - grasp_center_before_lift[2]
                    )
                    * 1000.0
                ),
            "max_grasp_center_lift_mm":
                float(
                    (
                        max_grasp_center_z
                        - grasp_center_before_lift[2]
                    )
                    * 1000.0
                ),
            "max_A_lift_during_motion_mm":
                float(
                    (
                        max_A_z
                        - A_before_lift[2]
                    )
                    * 1000.0
                ),
            "first_bilateral_contact_loss_waypoint":
                first_contact_loss_waypoint,
            "final_tcp_orientation_drift_deg":
                quaternion_angle_deg(
                    tcp_q_before_lift,
                    tcp_q_final,
                ),
            "final_A_minus_grasp_center_mm":
                (
                    (
                        A_final
                        - grasp_center_final
                    )
                    * 1000.0
                ).tolist(),
            "lift_trace":
                lift_trace,
            "micro_lift_ik_tolerance_m":
                float(
                    grasp_cfg[
                        "micro_lift_ik_tolerance_m"
                    ]
                ),
            "micro_lift_zero_like_q_transitions":
                int(
                    plan[
                        "zero_like_transitions"
                    ]
                ),
            "micro_lift_planned_max_abs_dq_rad":
                float(
                    plan[
                        "max_lift_transition_dq"
                    ]
                ),
            "micro_lift_control_steps_per_waypoint":
                int(
                    micro_lift_control_steps
                ),
            "actual_state_replan": {
                "physical_start_grasp_center_m":
                    actual_lift_plan[
                        "physical_start_center_m"
                    ].tolist(),
                "shadow_start_grasp_center_m":
                    actual_lift_plan[
                        "shadow_start_center_m"
                    ].tolist(),
                "shadow_physical_sync_error_mm":
                    float(
                        actual_lift_plan[
                            "center_sync_error_m"
                        ]
                        * 1000.0
                    ),
                "requested_relative_lift_mm":
                    float(
                        actual_lift_plan[
                            "relative_lift_m"
                        ]
                        * 1000.0
                    ),
                "target_grasp_center_m":
                    actual_lift_plan[
                        "target_center_m"
                    ].tolist(),
                "zero_like_q_transitions":
                    int(
                        actual_lift_plan[
                            "zero_like_transitions"
                        ]
                    ),
                "planned_max_abs_dq_rad":
                    float(
                        actual_lift_plan[
                            "max_abs_dq_rad"
                        ]
                    ),
                "final_ik_error_m":
                    float(
                        actual_lift_plan[
                            "lift"
                        ][
                            "errors_m"
                        ][-1]
                    ),
            },
            "ik_final_errors_m": {
                "to_pregrasp":
                    float(
                        plan[
                            "to_pregrasp"
                        ][
                            "errors_m"
                        ][-1]
                    ),
                "descend":
                    float(
                        plan[
                            "descend"
                        ][
                            "errors_m"
                        ][-1]
                    ),
                "micro_lift":
                    float(
                        actual_lift_plan[
                            "lift"
                        ][
                            "errors_m"
                        ][-1]
                    ),
            },
        }

        result[
            "phase5d_fixed_expert_sequence"
        ] = fixed_sequence_result

        output = (
            args.output
            if args.output.is_absolute()
            else PROJECT_ROOT
            / args.output
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "JSON:",
            output,
        )

        print(
            "\n========================================"
        )
        print(
            "ALIGNMENT A + STRICT FIXED [6,4,3] ZERO-RESIDUAL:",
            "PASS" if success else "FAIL",
        )
        print(
            "NO ROS2 / NO REAL ROBOT COMMAND"
        )
        print(
            "========================================"
        )

        if not success:
            raise RuntimeError(
                "Alignment A + fixed [6,4,3] diagnostic failed after report save: "
                + ", ".join(failure_reasons)
            )

    finally:
        planning_env.close()
        physical_env.close()


if __name__ == "__main__":
    main()
