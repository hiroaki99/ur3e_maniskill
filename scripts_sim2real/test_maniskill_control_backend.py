#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4A smoke test for ManiSkillControlBackend.

Tests:
1. Runtime controller contract.
2. Normalized action mapping for +0.001 rad.
3. Physical joint tracking with env.step().
4. Cartesian +X 1 mm:
       separate kinematics ManiSkill env
       +
       physical ManiSkill env
       +
       existing CartesianDeltaController

No UR3eROS2Backend is imported or created.
No real robot command is sent.

Direct set_qpos is used ONLY once for TEST INITIALIZATION through the already
validated ManiSkillBackend. All tested motion after initialization uses
ManiSkillControlBackend -> env.step(action).
"""

from __future__ import annotations

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


from scripts_sim2real.maniskill_backend import (  # noqa: E402
    ManiSkillBackend,
)
from scripts_sim2real.maniskill_control_backend import (  # noqa: E402
    ManiSkillControlBackend,
)
from scripts_sim2real.cartesian_delta_controller import (  # noqa: E402
    CartesianDeltaController,
    JOINT_NAMES,
)


INITIAL_Q = {
    "shoulder_pan_joint": 1.548456430,
    "shoulder_lift_joint": -1.609772821,
    "elbow_joint": 1.540878121,
    "wrist_1_joint": -3.047041079,
    "wrist_2_joint": 0.949946761,
    "wrist_3_joint": -0.000787560,
}


def dict_to_array(
    d,
):
    return np.asarray(
        [
            d[name]
            for name in JOINT_NAMES
        ],
        dtype=np.float64,
    )


def make_env():
    env = gym.make(
        "UR3eReach-v0",
        num_envs=1,
        obs_mode="state",
        control_mode=(
            "pd_joint_delta_pos"
        ),
    )

    env.reset(
        seed=0
    )

    return env


def initialize_q(
    env,
):
    """
    Test-only state initialization.

    This is NOT the motion method being validated.
    """
    setup_backend = (
        ManiSkillBackend(
            env
        )
    )

    setup_backend.command_joint_positions(
        INITIAL_Q,
        duration_sec=0.0,
    )

    actual = dict_to_array(
        setup_backend
        .get_joint_positions()
    )

    expected = dict_to_array(
        INITIAL_Q
    )

    if not np.allclose(
        actual,
        expected,
        atol=1.0e-5,
    ):
        raise RuntimeError(
            "Failed to initialize test q.\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )

    return setup_backend


def main():
    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "expert_gp"
        / "physical_control"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    physical_env = make_env()
    kinematics_env = make_env()

    try:
        # set_qpos is permitted here ONLY as test initialization.
        physical_setup = initialize_q(
            physical_env
        )
        kinematics_backend = initialize_q(
            kinematics_env
        )

        control_backend = (
            ManiSkillControlBackend(
                physical_env,
                joint_tolerance_rad=(
                    2.0e-4
                ),
                max_requested_joint_delta_rad=(
                    1.0e-2
                ),
                default_duration_sec=(
                    0.5
                ),
            )
        )

        rows = []

        print(
            "=== ManiSkill Physical Control Backend Test ==="
        )
        print(
            "NO REAL ROBOT BACKEND IS CREATED."
        )
        print(
            "Motion under test uses env.step(action)."
        )

        print(
            "\n=== Runtime controller contract ==="
        )
        print(
            "control_freq [Hz]:",
            control_backend.control_freq,
        )
        print(
            "lower [rad]:",
            control_backend.lower,
        )
        print(
            "upper [rad]:",
            control_backend.upper,
        )

        # ----------------------------------------------------
        # Test 1: inverse normalization
        # ----------------------------------------------------
        requested_delta = np.array(
            [
                0.001,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        mapped_action = (
            control_backend
            .joint_delta_to_normalized_action(
                requested_delta
            )
        )

        expected_action = np.array(
            [
                0.001 / 0.03,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        mapping_error = float(
            np.max(
                np.abs(
                    mapped_action
                    - expected_action
                )
            )
        )

        print(
            "\n=== Test 1: normalization ==="
        )
        print(
            "requested dq [rad]:",
            requested_delta,
        )
        print(
            "normalized action:",
            mapped_action,
        )
        print(
            "expected:",
            expected_action,
        )
        print(
            "max mapping error:",
            mapping_error,
        )

        if mapping_error > 1.0e-6:
            raise RuntimeError(
                "Normalized action mapping mismatch."
            )

        rows.append(
            {
                "test":
                    "normalization_1mrad",

                "success":
                    True,

                "requested_max_dq_rad":
                    0.001,

                "env_steps":
                    0,

                "final_max_joint_error_rad":
                    mapping_error,

                "tcp_error_mm":
                    "",

                "condition_number":
                    "",
            }
        )

        # ----------------------------------------------------
        # Test 2: physical +0.001 rad shoulder_pan
        # ----------------------------------------------------
        q_before_dict = (
            control_backend
            .get_joint_positions()
        )

        q_before = dict_to_array(
            q_before_dict
        )

        q_target = (
            q_before.copy()
        )

        q_target[0] += 0.001

        q_target_dict = {
            name: float(value)
            for name, value in zip(
                JOINT_NAMES,
                q_target,
            )
        }

        print(
            "\n=== Test 2: physical +1 mrad joint tracking ==="
        )

        joint_result = (
            control_backend
            .command_joint_positions(
                q_target_dict,
                duration_sec=0.5,
            )
        )

        q_after = dict_to_array(
            control_backend
            .get_joint_positions()
        )

        achieved_delta = (
            q_after
            - q_before
        )

        print(
            "requested dq [rad]:",
            q_target - q_before,
        )
        print(
            "achieved dq [rad]:",
            achieved_delta,
        )
        print(
            "env steps:",
            joint_result[
                "env_steps"
            ],
        )
        print(
            "final max q error [rad]:",
            joint_result[
                "final_max_joint_error_rad"
            ],
        )
        print(
            "max normalized action:",
            joint_result[
                "max_abs_normalized_action"
            ],
        )

        if not joint_result[
            "success"
        ]:
            raise RuntimeError(
                "Joint tracking test failed."
            )

        rows.append(
            {
                "test":
                    "physical_joint_1mrad",

                "success":
                    joint_result[
                        "success"
                    ],

                "requested_max_dq_rad":
                    0.001,

                "env_steps":
                    joint_result[
                        "env_steps"
                    ],

                "final_max_joint_error_rad":
                    joint_result[
                        "final_max_joint_error_rad"
                    ],

                "tcp_error_mm":
                    "",

                "condition_number":
                    "",
            }
        )

        # ----------------------------------------------------
        # Reinitialize BOTH models before Cartesian test.
        # This direct qpos assignment is still only test setup.
        # ----------------------------------------------------
        initialize_q(
            physical_env
        )
        initialize_q(
            kinematics_env
        )

        # ----------------------------------------------------
        # Test 3: physical Cartesian +X 1 mm
        # ----------------------------------------------------
        controller = (
            CartesianDeltaController(
                kinematics_backend=(
                    kinematics_backend
                ),
                robot_backend=(
                    control_backend
                ),
                jacobian_eps_rad=(
                    1.0e-3
                ),
                damping=1.0e-3,
                max_joint_delta_rad=(
                    1.0e-2
                ),
                max_cartesian_delta_m=(
                    5.0e-3
                ),
                max_condition_number=(
                    200.0
                ),
                command_duration_sec=(
                    0.5
                ),
            )
        )

        tcp_before = np.asarray(
            control_backend
            .get_tcp_pose()
            .position,
            dtype=np.float64,
        )

        requested_tcp_delta = np.array(
            [
                0.001,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        tcp_target = (
            tcp_before
            + requested_tcp_delta
        )

        print(
            "\n=== Test 3: physical Cartesian +X 1 mm ==="
        )
        print(
            "TCP before [m]:",
            tcp_before,
        )

        command_result = (
            controller
            .command_tcp_delta(
                requested_tcp_delta,
                execute=True,
            )
        )

        plan = (
            command_result["plan"]
        )

        tcp_after = np.asarray(
            control_backend
            .get_tcp_pose()
            .position,
            dtype=np.float64,
        )

        actual_tcp_delta = (
            tcp_after
            - tcp_before
        )

        tcp_target_error_m = float(
            np.linalg.norm(
                tcp_after
                - tcp_target
            )
        )

        print(
            "requested TCP delta [m]:",
            requested_tcp_delta,
        )
        print(
            "predicted TCP delta [m]:",
            plan.predicted_delta_m,
        )
        print(
            "actual TCP delta [m]:",
            actual_tcp_delta,
        )
        print(
            "condition number:",
            plan.condition_number,
        )
        print(
            "max |dq| [rad]:",
            float(
                np.max(
                    np.abs(
                        plan.dq
                    )
                )
            ),
        )
        print(
            "TCP target error [mm]:",
            tcp_target_error_m
            * 1000.0,
        )

        # Initial smoke-test gate:
        # 1 mm target should be followed to within 0.25 mm.
        if (
            tcp_target_error_m
            > 0.00025
        ):
            raise RuntimeError(
                "Cartesian physical tracking "
                "error too large: "
                f"{tcp_target_error_m*1000.0:.3f} mm "
                "> 0.250 mm"
            )

        rows.append(
            {
                "test":
                    "physical_cartesian_x_1mm",

                "success":
                    True,

                "requested_max_dq_rad":
                    float(
                        np.max(
                            np.abs(
                                plan.dq
                            )
                        )
                    ),

                "env_steps":
                    (
                        control_backend
                        .last_command_result[
                            "env_steps"
                        ]
                    ),

                "final_max_joint_error_rad":
                    (
                        control_backend
                        .last_command_result[
                            "final_max_joint_error_rad"
                        ]
                    ),

                "tcp_error_mm":
                    (
                        tcp_target_error_m
                        * 1000.0
                    ),

                "condition_number":
                    plan.condition_number,
            }
        )

        # ----------------------------------------------------
        # Save compact result
        # ----------------------------------------------------
        csv_path = (
            output_dir
            / "maniskill_control_backend_test.csv"
        )

        fieldnames = [
            "test",
            "success",
            "requested_max_dq_rad",
            "env_steps",
            "final_max_joint_error_rad",
            "tcp_error_mm",
            "condition_number",
        ]

        with csv_path.open(
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
                rows
            )

        print(
            "\nCSV:",
            csv_path,
        )

        print(
            "\n========================================"
        )
        print(
            "MANISKILL PHYSICAL CONTROL SMOKE TEST: PASS"
        )
        print(
            "NO REAL ROBOT COMMAND WAS SENT."
        )
        print(
            "========================================"
        )

    finally:
        physical_env.close()
        kinematics_env.close()


if __name__ == "__main__":
    main()
