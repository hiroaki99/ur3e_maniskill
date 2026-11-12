#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 5B-2 yaw-aware controller: object-face-aligned orientation + adaptive force-band squeeze + actual-state lift replan.

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
- No Expert trajectory is executed.
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


def make_env():
    return gym.make(
        "UR3eAlignment-v0",
        num_envs=1,
        obs_mode="state",
        reward_mode="dense",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        max_episode_steps=2000,
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


def main():
    args = parse_args()

    if args.output is None:
        args.output = (
            PROJECT_ROOT
            / "reports"
            / "expert_gp"
            / "alignment"
            / (
                "grasp_A_yaw_aware_seed"
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

    planning_env = make_env()
    physical_env = make_env()

    try:
        print(
            "=== Phase 5B-2 Alignment A Grasp Smoke Test ==="
        )
        print(
            "NO EXPERT / NO ROS2 / NO REAL ROBOT COMMAND"
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
            "ALIGNMENT OBJECT-A YAW-AWARE + ADAPTIVE:",
            "PASS" if success else "FAIL",
        )
        print(
            "NO EXPERT / NO REAL ROBOT COMMAND"
        )
        print(
            "========================================"
        )

        if not success:
            raise RuntimeError(
                "Alignment A grasp diagnostic failed after report save: "
                + ", ".join(failure_reasons)
            )

    finally:
        planning_env.close()
        physical_env.close()


if __name__ == "__main__":
    main()
