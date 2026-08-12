#!/usr/bin/env python3
"""
Reward model for UR3e Residual Pick-and-Lift.

Yaginuma RRL:
    Expert trajectory + learned residual correction
    reward = task-level geometric quality

UR3e port:
    Reference trajectory + 6-DoF residual
    reward = approach
           + grasp
           + hold
           + lift
           + success
           - residual penalty
           - excessive force
           - unsafe collision
           - terminal failure

報酬項を分離して返し、
学習失敗時に原因を解析できるようにする。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ResidualPickLiftReward:

    APPROACH_PHASES = {
        "to_pregrasp",
        "descend",
    }

    GRASP_PHASES = {
        "grasp",
        "stable_hold",
    }

    HOLD_PHASES = {
        "lift",
        "final_hold",
    }

    def __init__(
        self,
        config: dict,
        observation_layout: dict,
        alpha: float,
    ):

        self.config = dict(
            config
        )

        self.layout = dict(
            observation_layout
        )

        self.alpha = float(
            alpha
        )

        # --------------------------------------------------
        # Approach
        # --------------------------------------------------

        self.approach_scale = float(
            self.config.get(
                "approach_progress_scale_m",
                0.01,
            )
        )

        self.approach_weight = float(
            self.config.get(
                "approach_progress_weight",
                0.2,
            )
        )

        # --------------------------------------------------
        # Contact
        # --------------------------------------------------

        self.bilateral_contact_bonus = float(
            self.config.get(
                "bilateral_contact_bonus",
                0.05,
            )
        )

        self.hold_contact_bonus = float(
            self.config.get(
                "hold_contact_bonus",
                0.02,
            )
        )

        # --------------------------------------------------
        # Lift
        # --------------------------------------------------

        self.lift_weight = float(
            self.config.get(
                "lift_progress_weight",
                100.0,
            )
        )

        self.max_lift_delta = float(
            self.config.get(
                "max_lift_delta_per_step_m",
                0.01,
            )
        )

        # --------------------------------------------------
        # Terminal
        # --------------------------------------------------

        self.success_bonus = float(
            self.config.get(
                "success_bonus",
                10.0,
            )
        )

        self.failure_penalty = float(
            self.config.get(
                "failure_penalty",
                5.0,
            )
        )

        # --------------------------------------------------
        # Residual
        # --------------------------------------------------

        self.residual_penalty_weight = float(
            self.config.get(
                "residual_penalty_weight",
                0.1,
            )
        )

        # --------------------------------------------------
        # Safety
        # --------------------------------------------------

        self.max_safe_force = float(
            self.config.get(
                "max_safe_gripper_force_n",
                30.0,
            )
        )

        self.excessive_force_weight = float(
            self.config.get(
                "excessive_force_penalty_weight",
                0.2,
            )
        )

        self.unsafe_collision_penalty = float(
            self.config.get(
                "unsafe_collision_penalty",
                5.0,
            )
        )

        if self.approach_scale <= 0.0:

            raise ValueError(
                "approach_progress_scale_m "
                "must be > 0"
            )

        if self.max_safe_force <= 0.0:

            raise ValueError(
                "max_safe_gripper_force_n "
                "must be > 0"
            )

    # ======================================================
    # Observation access
    # ======================================================

    def _slice(
        self,
        observation,
        name: str,
    ):

        start, stop = (
            self.layout[
                name
            ]
        )

        observation = np.asarray(
            observation,
            dtype=np.float64,
        )

        return observation[
            start:stop
        ]

    def _scalar(
        self,
        observation,
        name: str,
    ) -> float:

        value = self._slice(
            observation,
            name,
        )

        if value.size != 1:

            raise ValueError(
                f"{name} is not scalar"
            )

        return float(
            value.reshape(-1)[0]
        )

    # ======================================================
    # Reward
    # ======================================================

    def compute(
        self,
        previous_observation,
        observation,
        residual_action,
        info: dict[str, Any],
        terminated: bool,
    ):

        previous_observation = np.asarray(
            previous_observation,
            dtype=np.float64,
        )

        observation = np.asarray(
            observation,
            dtype=np.float64,
        )

        residual_action = np.asarray(
            residual_action,
            dtype=np.float64,
        ).reshape(-1)

        phase_name = str(
            info.get(
                "phase_name_before",
                "",
            )
        )

        # ==================================================
        # 1. Approach progress
        # ==================================================

        previous_cube_to_grasp = (
            self._slice(
                previous_observation,
                "cube_to_grasp",
            )
        )

        current_cube_to_grasp = (
            self._slice(
                observation,
                "cube_to_grasp",
            )
        )

        previous_distance = float(
            np.linalg.norm(
                previous_cube_to_grasp
            )
        )

        current_distance = float(
            np.linalg.norm(
                current_cube_to_grasp
            )
        )

        approach_progress = (
            previous_distance
            - current_distance
        )

        reward_approach = 0.0

        if (
            phase_name
            in self.APPROACH_PHASES
        ):

            normalized_progress = float(
                np.clip(
                    approach_progress
                    / self.approach_scale,
                    -1.0,
                    1.0,
                )
            )

            reward_approach = (
                self.approach_weight
                * normalized_progress
            )

        # ==================================================
        # 2. Grasp / Hold
        # ==================================================

        left_contact = bool(
            info.get(
                "left_contact",
                False,
            )
        )

        right_contact = bool(
            info.get(
                "right_contact",
                False,
            )
        )

        both_contact = (
            left_contact
            and right_contact
        )

        reward_grasp = 0.0
        reward_hold = 0.0

        if (
            both_contact
            and phase_name
            in self.GRASP_PHASES
        ):

            reward_grasp = (
                self.bilateral_contact_bonus
            )

        if (
            both_contact
            and phase_name
            in self.HOLD_PHASES
        ):

            reward_hold = (
                self.hold_contact_bonus
            )

        # ==================================================
        # 3. Lift progress
        # ==================================================

        previous_lift = self._scalar(
            previous_observation,
            "cube_lift",
        )

        current_lift = self._scalar(
            observation,
            "cube_lift",
        )

        lift_delta = (
            current_lift
            - previous_lift
        )

        lift_delta_clipped = float(
            np.clip(
                lift_delta,
                -self.max_lift_delta,
                self.max_lift_delta,
            )
        )

        reward_lift = 0.0

        if phase_name in {
            "stable_hold",
            "lift",
            "final_hold",
        }:

            reward_lift = (
                self.lift_weight
                * lift_delta_clipped
            )

        # ==================================================
        # 4. Residual penalty
        # ==================================================

        residual_action = np.clip(
            residual_action,
            -1.0,
            1.0,
        )

        # 実際にReferenceへ加わる大きさは
        # alpha * residual
        scaled_residual = (
            self.alpha
            * residual_action
        )

        residual_squared_mean = float(
            np.mean(
                np.square(
                    scaled_residual
                )
            )
        )

        reward_residual = (
            -self.residual_penalty_weight
            * residual_squared_mean
        )

        # ==================================================
        # 5. Excessive contact force
        # ==================================================

        left_force = float(
            info.get(
                "left_contact_force",
                0.0,
            )
        )

        right_force = float(
            info.get(
                "right_contact_force",
                0.0,
            )
        )

        maximum_force = max(
            left_force,
            right_force,
        )

        force_excess_ratio = max(
            0.0,
            (
                maximum_force
                - self.max_safe_force
            )
            / self.max_safe_force,
        )

        reward_force = (
            -self.excessive_force_weight
            * force_excess_ratio
        )

        # ==================================================
        # 6. Unsafe collision
        # ==================================================

        unsafe_collision = bool(
            info.get(
                "unsafe_collision",
                False,
            )
        )

        reward_collision = (
            -self.unsafe_collision_penalty
            if unsafe_collision
            else 0.0
        )

        # ==================================================
        # 7. Terminal reward
        # ==================================================

        success = bool(
            info.get(
                "success",
                False,
            )
        )

        reward_success = 0.0
        reward_failure = 0.0

        if terminated:

            if success:

                reward_success = (
                    self.success_bonus
                )

            else:

                reward_failure = (
                    -self.failure_penalty
                )

        # ==================================================
        # Total
        # ==================================================

        total = float(
            reward_approach
            + reward_grasp
            + reward_hold
            + reward_lift
            + reward_success
            + reward_residual
            + reward_force
            + reward_collision
            + reward_failure
        )

        terms = {

            "approach":
                float(
                    reward_approach
                ),

            "grasp":
                float(
                    reward_grasp
                ),

            "hold":
                float(
                    reward_hold
                ),

            "lift":
                float(
                    reward_lift
                ),

            "success":
                float(
                    reward_success
                ),

            "residual":
                float(
                    reward_residual
                ),

            "excessive_force":
                float(
                    reward_force
                ),

            "unsafe_collision":
                float(
                    reward_collision
                ),

            "failure":
                float(
                    reward_failure
                ),

            "total":
                total,

            # Debug metrics
            "grasp_distance_m":
                current_distance,

            "approach_progress_m":
                approach_progress,

            "cube_lift_m":
                current_lift,

            "cube_lift_delta_m":
                lift_delta,

            "residual_squared_mean":
                residual_squared_mean,

            "max_gripper_force_n":
                maximum_force,
        }

        return (
            total,
            terms,
        )