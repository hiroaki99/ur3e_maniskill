#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch kinematic validation of Yaginuma Experts 00..11 in ManiSkill UR3e.

IMPORTANT:
- NO UR3eROS2Backend is imported.
- NO real robot command is sent.
- ManiSkillBackend is used for both kinematics_backend and robot_backend.
- execute=True therefore updates only the ManiSkill shadow qpos through
  ManiSkillBackend.command_joint_positions().

This script is the Phase-3B gate:
    12 GP experts
        -> relative TCP mapping
        -> 1 mm arc-length resampling
        -> sequential Cartesian IK
        -> one summary CSV + per-expert detail CSV

The same validated expert-library / expert-executor interface is intended
to be reused later with:
    ManiSkill physical control backend (env.step)
and then, only after safety validation:
    UR3eROS2Backend
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


PROJECT_ROOT = (
    Path(__file__).resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import mani_skill.envs  # noqa: F401,E402
import envs.ur3e_reach  # noqa: F401,E402


from expert_gp.expert_library import (  # noqa: E402
    ExpertLibrary,
)
from expert_gp.expert_executor import (  # noqa: E402
    ExpertExecutor,
)
from scripts_sim2real.maniskill_backend import (  # noqa: E402
    ManiSkillBackend,
)
from scripts_sim2real.cartesian_delta_controller import (  # noqa: E402
    CartesianDeltaController,
    JOINT_NAMES,
)


# Previously validated non-singular pose.
# Expert 06 passed 125/125 transitions from this pose at 1 mm spacing.
DEFAULT_INITIAL_Q_VALUES = [
    1.548456430,
    -1.609772821,
    1.540878121,
    -3.047041079,
    0.949946761,
    -0.000787560,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch ManiSkill kinematic validation "
            "for Yaginuma Experts 00..11."
        )
    )

    parser.add_argument(
        "--expert-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "expert_gp"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "expert_gp"
            / "all_experts_kinematic"
        ),
    )

    parser.add_argument(
        "--env-id",
        default="UR3eReach-v0",
    )

    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        default=list(
            range(12)
        ),
        help=(
            "Expert ids to evaluate. "
            "Default: 0 1 ... 11"
        ),
    )

    parser.add_argument(
        "--max-step-mm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--initial-q",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_Q_VALUES,
        metavar=(
            "SHOULDER_PAN",
            "SHOULDER_LIFT",
            "ELBOW",
            "WRIST1",
            "WRIST2",
            "WRIST3",
        ),
    )

    parser.add_argument(
        "--jacobian-eps-rad",
        type=float,
        default=1.0e-3,
    )

    parser.add_argument(
        "--damping",
        type=float,
        default=1.0e-3,
    )

    parser.add_argument(
        "--max-joint-delta-rad",
        type=float,
        default=1.0e-2,
    )

    parser.add_argument(
        "--max-cartesian-delta-m",
        type=float,
        default=5.0e-3,
    )

    parser.add_argument(
        "--max-condition-number",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--command-duration-sec",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--verbose-waypoints",
        action="store_true",
        help=(
            "Print every waypoint. "
            "Default prints one line per expert."
        ),
    )

    return parser.parse_args()


def q_values_to_dict(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.shape != (6,):
        raise ValueError(
            "initial q must contain "
            "exactly six values."
        )

    return {
        name: float(value)
        for name, value in zip(
            JOINT_NAMES,
            values,
        )
    }


def write_detail_csv(
    path: Path,
    result,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "waypoint",
        "delta_x_m",
        "delta_y_m",
        "delta_z_m",
        "delta_norm_mm",
        "condition_number",
        "max_abs_dq_rad",
        "predicted_position_error_mm",
        "predicted_orientation_change_deg",
        "actual_target_error_mm",
        "status",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for step in result.steps:
            writer.writerow(
                {
                    "waypoint":
                        step.waypoint,

                    "delta_x_m":
                        step.delta_x_m,

                    "delta_y_m":
                        step.delta_y_m,

                    "delta_z_m":
                        step.delta_z_m,

                    "delta_norm_mm":
                        step.delta_norm_m
                        * 1000.0,

                    "condition_number":
                        (
                            ""
                            if step.condition_number
                            is None
                            else step.condition_number
                        ),

                    "max_abs_dq_rad":
                        (
                            ""
                            if step.max_abs_dq_rad
                            is None
                            else step.max_abs_dq_rad
                        ),

                    "predicted_position_error_mm":
                        (
                            ""
                            if step.predicted_position_error_m
                            is None
                            else (
                                step.predicted_position_error_m
                                * 1000.0
                            )
                        ),

                    "predicted_orientation_change_deg":
                        (
                            ""
                            if step.predicted_orientation_change_deg
                            is None
                            else (
                                step.predicted_orientation_change_deg
                            )
                        ),

                    "actual_target_error_mm":
                        (
                            ""
                            if step.actual_target_error_m
                            is None
                            else (
                                step.actual_target_error_m
                                * 1000.0
                            )
                        ),

                    "status":
                        step.status,
                }
            )


def main():
    args = parse_args()

    if (
        not np.isfinite(
            args.max_step_mm
        )
        or args.max_step_mm <= 0.0
    ):
        raise ValueError(
            "--max-step-mm must be > 0."
        )

    expert_ids = [
        int(x)
        for x in args.experts
    ]

    if (
        len(expert_ids)
        != len(set(expert_ids))
    ):
        raise ValueError(
            "--experts contains duplicates."
        )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    expert_library = (
        ExpertLibrary(
            args.expert_dir
        )
    )

    initial_q_dict = (
        q_values_to_dict(
            args.initial_q
        )
    )

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        control_mode=(
            "pd_joint_delta_pos"
        ),
    )

    sim_backend = None

    try:
        env.reset()

        sim_backend = (
            ManiSkillBackend(
                env
            )
        )

        controller = (
            CartesianDeltaController(
                kinematics_backend=(
                    sim_backend
                ),
                robot_backend=(
                    sim_backend
                ),
                jacobian_eps_rad=(
                    args.jacobian_eps_rad
                ),
                damping=(
                    args.damping
                ),
                max_joint_delta_rad=(
                    args.max_joint_delta_rad
                ),
                max_cartesian_delta_m=(
                    args.max_cartesian_delta_m
                ),
                max_condition_number=(
                    args.max_condition_number
                ),
                command_duration_sec=(
                    args.command_duration_sec
                ),
            )
        )

        executor = (
            ExpertExecutor(
                controller=controller,
                robot_backend=(
                    sim_backend
                ),
            )
        )

        summary_rows = []

        print(
            "=== All Experts ManiSkill "
            "Kinematic Validation ==="
        )
        print(
            "experts       :",
            expert_ids,
        )
        print(
            "max step [mm] :",
            args.max_step_mm,
        )
        print(
            "NO REAL ROBOT BACKEND "
            "IS CREATED."
        )
        print()

        for expert_id in expert_ids:
            # ------------------------------------------------
            # Every expert starts from exactly the same q.
            # ------------------------------------------------
            sim_backend.command_joint_positions(
                initial_q_dict,
                duration_sec=0.0,
            )

            trajectory = (
                expert_library
                .prepare_relative(
                    expert_id=expert_id,
                    max_step_m=(
                        args.max_step_mm
                        / 1000.0
                    ),
                )
            )

            start_pose = (
                sim_backend
                .get_tcp_pose()
            )

            start_tcp = np.asarray(
                start_pose.position,
                dtype=np.float64,
            )

            result = (
                executor
                .execute_relative_waypoints(
                    trajectory
                    .tcp_sampled_relative,
                    expert_id=(
                        expert_id
                    ),
                    execute=True,
                    verbose=(
                        args.verbose_waypoints
                    ),
                )
            )

            detail_csv = (
                output_dir
                / f"expert_{expert_id:02d}_steps.csv"
            )

            write_detail_csv(
                detail_csv,
                result,
            )

            summary_rows.append(
                {
                    "expert_id":
                        expert_id,

                    "logical_points":
                        int(
                            trajectory
                            .logical.shape[0]
                        ),

                    "sampled_waypoints":
                        int(
                            trajectory
                            .tcp_sampled_relative
                            .shape[0]
                        ),

                    "planned_transitions":
                        result
                        .planned_transitions,

                    "passed_transitions":
                        result
                        .passed_transitions,

                    "logical_path_length_mm":
                        trajectory
                        .logical_path_length_m
                        * 1000.0,

                    "tcp_path_length_mm":
                        trajectory
                        .tcp_path_length_m
                        * 1000.0,

                    "sampled_path_length_mm":
                        trajectory
                        .sampled_path_length_m
                        * 1000.0,

                    "max_nominal_sample_step_mm":
                        trajectory
                        .max_sampled_step_m
                        * 1000.0,

                    "anchor_x_m":
                        float(
                            result
                            .anchor_tcp_m[0]
                        ),

                    "anchor_y_m":
                        float(
                            result
                            .anchor_tcp_m[1]
                        ),

                    "anchor_z_m":
                        float(
                            result
                            .anchor_tcp_m[2]
                        ),

                    "max_condition_number":
                        (
                            ""
                            if result
                            .max_condition_number
                            is None
                            else result
                            .max_condition_number
                        ),

                    "max_abs_dq_rad":
                        (
                            ""
                            if result
                            .max_abs_dq_rad
                            is None
                            else result
                            .max_abs_dq_rad
                        ),

                    "max_predicted_error_mm":
                        (
                            ""
                            if result
                            .max_predicted_error_m
                            is None
                            else (
                                result
                                .max_predicted_error_m
                                * 1000.0
                            )
                        ),

                    "max_actual_error_mm":
                        (
                            ""
                            if result
                            .max_actual_error_m
                            is None
                            else (
                                result
                                .max_actual_error_m
                                * 1000.0
                            )
                        ),

                    "success":
                        result.success,

                    "failure_waypoint":
                        (
                            ""
                            if result
                            .failure_waypoint
                            is None
                            else result
                            .failure_waypoint
                        ),

                    "failure_type":
                        (
                            ""
                            if result
                            .failure_type
                            is None
                            else result
                            .failure_type
                        ),

                    "failure_reason":
                        (
                            ""
                            if result
                            .failure_reason
                            is None
                            else result
                            .failure_reason
                        ),

                    "detail_csv":
                        str(
                            detail_csv
                        ),
                }
            )

            status = (
                "PASS"
                if result.success
                else "FAIL"
            )

            max_cond = (
                "-"
                if result
                .max_condition_number
                is None
                else (
                    f"{result.max_condition_number:.2f}"
                )
            )

            max_dq = (
                "-"
                if result
                .max_abs_dq_rad
                is None
                else (
                    f"{result.max_abs_dq_rad:.6f}"
                )
            )

            print(
                f"Expert {expert_id:02d}: "
                f"{status} | "
                f"{result.passed_transitions}/"
                f"{result.planned_transitions} | "
                f"waypoints="
                f"{result.num_waypoints} | "
                f"cond_max={max_cond} | "
                f"dq_max={max_dq}"
            )

            if not result.success:
                print(
                    "  failure:",
                    result.failure_type,
                    "|",
                    result.failure_reason,
                )

        summary_csv = (
            output_dir
            / "all_experts_kinematic_summary.csv"
        )

        fieldnames = [
            "expert_id",
            "logical_points",
            "sampled_waypoints",
            "planned_transitions",
            "passed_transitions",
            "logical_path_length_mm",
            "tcp_path_length_mm",
            "sampled_path_length_mm",
            "max_nominal_sample_step_mm",
            "anchor_x_m",
            "anchor_y_m",
            "anchor_z_m",
            "max_condition_number",
            "max_abs_dq_rad",
            "max_predicted_error_mm",
            "max_actual_error_mm",
            "success",
            "failure_waypoint",
            "failure_type",
            "failure_reason",
            "detail_csv",
        ]

        with summary_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                summary_rows
            )

        num_pass = sum(
            bool(
                row["success"]
            )
            for row in summary_rows
        )

        num_total = len(
            summary_rows
        )

        print()
        print(
            "=== Batch Summary ==="
        )
        print(
            "passed experts:",
            f"{num_pass}/{num_total}",
        )
        print(
            "summary CSV:",
            summary_csv,
        )

        if num_pass == num_total:
            print()
            print(
                "========================================"
            )
            print(
                "ALL REQUESTED EXPERTS: PASS"
            )
            print(
                "Phase 3B kinematic batch gate: PASS"
            )
            print(
                "NO REAL ROBOT COMMAND WAS SENT."
            )
            print(
                "========================================"
            )
        else:
            print()
            print(
                "========================================"
            )
            print(
                "SOME EXPERTS FAILED"
            )
            print(
                "Inspect failure_type / failure_reason "
                "before changing safety limits."
            )
            print(
                "NO REAL ROBOT COMMAND WAS SENT."
            )
            print(
                "========================================"
            )

            # Results are already written for every expert.
            raise SystemExit(2)

    finally:
        if sim_backend is not None:
            sim_backend.close()

        env.close()


if __name__ == "__main__":
    main()
