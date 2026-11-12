#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reusable sequential expert trajectory executor / planner.

The executor itself is simulator/robot agnostic.  It only requires:
- a CartesianDeltaController
- a robot_backend exposing get_tcp_pose()

For the current kinematic validation:
    kinematics_backend = ManiSkillBackend
    robot_backend      = ManiSkillBackend
    execute            = True

In that configuration, execute=True updates ONLY the ManiSkill shadow
qpos through ManiSkillBackend.command_joint_positions(); no real robot
backend is created.

Later, the same high-level trajectory interface can be reused with a
physical ManiSkill control backend and, after safety validation, a real
UR3e backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExpertStepResult:
    waypoint: int
    delta_x_m: float
    delta_y_m: float
    delta_z_m: float
    delta_norm_m: float
    condition_number: float | None
    max_abs_dq_rad: float | None
    predicted_position_error_m: float | None
    predicted_orientation_change_deg: float | None
    actual_target_error_m: float | None
    status: str


@dataclass
class ExpertExecutionResult:
    success: bool
    expert_id: int | None
    num_waypoints: int
    planned_transitions: int
    passed_transitions: int
    anchor_tcp_m: np.ndarray
    final_tcp_m: np.ndarray
    max_condition_number: float | None
    max_abs_dq_rad: float | None
    max_predicted_error_m: float | None
    max_actual_error_m: float | None
    failure_waypoint: int | None
    failure_type: str | None
    failure_reason: str | None
    steps: list[ExpertStepResult] = field(
        default_factory=list
    )


def classify_failure(
    message: str,
) -> str:
    lower = message.lower()

    if (
        "condition number" in lower
        or "singular" in lower
    ):
        return "singularity"

    if (
        "joint delta" in lower
        or "max_joint_delta" in lower
    ):
        return "joint_delta"

    if (
        "cartesian safety limit" in lower
        or "cartesian delta" in lower
    ):
        return "cartesian_delta"

    if (
        "joint limit" in lower
        or "limit exceeded" in lower
    ):
        return "joint_limit"

    if (
        "workspace" in lower
        or "unreachable" in lower
        or "reach" in lower
    ):
        return "workspace"

    return "other"


class ExpertExecutor:
    def __init__(
        self,
        controller,
        robot_backend,
    ):
        self.controller = controller
        self.robot_backend = (
            robot_backend
        )

    def _current_tcp(
        self,
    ) -> np.ndarray:
        pose = (
            self.robot_backend
            .get_tcp_pose()
        )

        return np.asarray(
            pose.position,
            dtype=np.float64,
        )

    def execute_relative_waypoints(
        self,
        relative_waypoints,
        *,
        expert_id: int | None = None,
        execute: bool = True,
        verbose: bool = False,
    ) -> ExpertExecutionResult:
        """
        Anchor a RELATIVE TCP trajectory at the current TCP and follow it
        sequentially.

        Parameters
        ----------
        relative_waypoints:
            (N,3) trajectory whose first point should normally be [0,0,0].

        execute:
            Passed to CartesianDeltaController.command_tcp_delta().
            For current kinematic validation, the controller uses
            ManiSkillBackend for both backends, so execute=True only updates
            the ManiSkill shadow state.
        """
        relative_waypoints = np.asarray(
            relative_waypoints,
            dtype=np.float64,
        )

        if (
            relative_waypoints.ndim != 2
            or relative_waypoints.shape[1] != 3
        ):
            raise ValueError(
                "relative_waypoints must have "
                f"shape (N,3), got "
                f"{relative_waypoints.shape}."
            )

        if (
            relative_waypoints.shape[0]
            < 2
        ):
            raise ValueError(
                "At least two waypoints are required."
            )

        if not np.all(
            np.isfinite(
                relative_waypoints
            )
        ):
            raise ValueError(
                "relative_waypoints contains "
                "NaN/Inf."
            )

        anchor = self._current_tcp()

        # Remove any small non-zero first-point offset before anchoring.
        relative_zeroed = (
            relative_waypoints
            - relative_waypoints[0]
        )

        absolute_waypoints = (
            anchor[None, :]
            + relative_zeroed
        )

        total_transitions = (
            absolute_waypoints.shape[0]
            - 1
        )

        steps = []

        failure_waypoint = None
        failure_type = None
        failure_reason = None

        for waypoint_id in range(
            1,
            absolute_waypoints.shape[0],
        ):
            current_tcp = (
                self._current_tcp()
            )

            target_tcp = (
                absolute_waypoints[
                    waypoint_id
                ]
            )

            delta = (
                target_tcp
                - current_tcp
            )

            delta_norm_m = float(
                np.linalg.norm(
                    delta
                )
            )

            try:
                result = (
                    self.controller
                    .command_tcp_delta(
                        delta,
                        execute=execute,
                    )
                )

                plan = result["plan"]

                achieved_tcp = (
                    self._current_tcp()
                    if execute
                    else (
                        current_tcp
                        + np.asarray(
                            plan.predicted_delta_m,
                            dtype=np.float64,
                        )
                    )
                )

                actual_error_m = float(
                    np.linalg.norm(
                        achieved_tcp
                        - target_tcp
                    )
                )

                max_abs_dq = float(
                    np.max(
                        np.abs(
                            plan.dq
                        )
                    )
                )

                step = ExpertStepResult(
                    waypoint=waypoint_id,
                    delta_x_m=float(
                        delta[0]
                    ),
                    delta_y_m=float(
                        delta[1]
                    ),
                    delta_z_m=float(
                        delta[2]
                    ),
                    delta_norm_m=(
                        delta_norm_m
                    ),
                    condition_number=float(
                        plan.condition_number
                    ),
                    max_abs_dq_rad=(
                        max_abs_dq
                    ),
                    predicted_position_error_m=float(
                        plan.predicted_position_error_m
                    ),
                    predicted_orientation_change_deg=float(
                        plan.predicted_orientation_change_deg
                    ),
                    actual_target_error_m=(
                        actual_error_m
                    ),
                    status="PASS",
                )

                steps.append(
                    step
                )

                if verbose:
                    print(
                        f"WP "
                        f"{waypoint_id:03d}/"
                        f"{total_transitions:03d}: "
                        f"|dTCP|="
                        f"{delta_norm_m*1000.0:.3f} mm, "
                        f"cond="
                        f"{plan.condition_number:.2f}, "
                        f"max|dq|="
                        f"{max_abs_dq:.6f} rad, "
                        f"actual_err="
                        f"{actual_error_m*1000.0:.4f} mm"
                    )

            except Exception as exc:
                failure_waypoint = (
                    waypoint_id
                )

                failure_reason = str(
                    exc
                )

                failure_type = (
                    classify_failure(
                        failure_reason
                    )
                )

                steps.append(
                    ExpertStepResult(
                        waypoint=(
                            waypoint_id
                        ),
                        delta_x_m=float(
                            delta[0]
                        ),
                        delta_y_m=float(
                            delta[1]
                        ),
                        delta_z_m=float(
                            delta[2]
                        ),
                        delta_norm_m=(
                            delta_norm_m
                        ),
                        condition_number=None,
                        max_abs_dq_rad=None,
                        predicted_position_error_m=None,
                        predicted_orientation_change_deg=None,
                        actual_target_error_m=None,
                        status=(
                            f"FAIL: "
                            f"{failure_reason}"
                        ),
                    )
                )

                break

        passed_steps = [
            s
            for s in steps
            if s.status == "PASS"
        ]

        success = (
            failure_waypoint is None
            and len(
                passed_steps
            )
            == total_transitions
        )

        def _max_or_none(
            values,
        ):
            values = [
                v
                for v in values
                if v is not None
            ]

            if not values:
                return None

            return float(
                max(values)
            )

        return ExpertExecutionResult(
            success=success,
            expert_id=expert_id,
            num_waypoints=int(
                absolute_waypoints.shape[0]
            ),
            planned_transitions=int(
                total_transitions
            ),
            passed_transitions=int(
                len(
                    passed_steps
                )
            ),
            anchor_tcp_m=(
                anchor.copy()
            ),
            final_tcp_m=(
                self._current_tcp()
            ),
            max_condition_number=(
                _max_or_none(
                    [
                        s.condition_number
                        for s in passed_steps
                    ]
                )
            ),
            max_abs_dq_rad=(
                _max_or_none(
                    [
                        s.max_abs_dq_rad
                        for s in passed_steps
                    ]
                )
            ),
            max_predicted_error_m=(
                _max_or_none(
                    [
                        s.predicted_position_error_m
                        for s in passed_steps
                    ]
                )
            ),
            max_actual_error_m=(
                _max_or_none(
                    [
                        s.actual_target_error_m
                        for s in passed_steps
                    ]
                )
            ),
            failure_waypoint=(
                failure_waypoint
            ),
            failure_type=(
                failure_type
            ),
            failure_reason=(
                failure_reason
            ),
            steps=steps,
        )
