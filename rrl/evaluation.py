#!/usr/bin/env python3
"""
Evaluation utilities for Residual Pick-and-Lift.

Training:
    random perturbation direction

Evaluation:
    fixed directions
    no exploration noise

Reference-onlyとTD3を同一条件で比較可能にする。
"""

from __future__ import annotations

import numpy as np


def evaluate_residual_policy(
    *,
    env,
    perturbation_wrapper,
    policy,
    angles_deg,
    episodes_per_direction: int,
    seed: int,
    max_steps: int,
    lift_threshold_m: float = 0.05,
    saturation_threshold: float = 0.95,
):
    """
    Parameters
    ----------
    policy:
        TD3 object.
        NoneならResidual=0、つまりReference-only。

    Returns
    -------
    summary: dict
    rows: list[dict]
    """

    rows = []

    action_dim = int(
        env.action_space.shape[0]
    )

    alpha = float(
        env.alpha
    )

    for angle_index, angle_deg in enumerate(
        angles_deg
    ):

        # Day12で追加した固定角度機能
        perturbation_wrapper.fixed_angle_rad = float(
            np.deg2rad(
                angle_deg
            )
        )

        for repeat in range(
            episodes_per_direction
        ):

            episode_seed = (
                int(seed)
                + angle_index * 1000
                + repeat
            )

            observation, reset_info = env.reset(
                seed=episode_seed
            )

            observation = np.asarray(
                observation,
                dtype=np.float32,
            )

            offset = np.asarray(
                reset_info.get(
                    "cube_offset_m",
                    [0.0, 0.0, 0.0],
                ),
                dtype=np.float64,
            )

            episode_return = 0.0

            grasp_detected = False

            max_cube_lift = 0.0

            residual_l2_sum = 0.0
            residual_abs_sum = 0.0

            residual_element_count = 0
            saturation_count = 0

            final_info = {}

            terminated = False
            truncated = False

            executed_steps = 0

            for _ in range(
                max_steps
            ):

                # ------------------------------------------
                # Deterministic evaluation action
                # ------------------------------------------

                if policy is None:

                    residual_action = np.zeros(
                        action_dim,
                        dtype=np.float32,
                    )

                else:

                    residual_action = (
                        policy.select_action(
                            observation
                        )
                    )

                residual_action = np.clip(
                    residual_action,
                    -1.0,
                    1.0,
                ).astype(
                    np.float32
                )

                # ------------------------------------------
                # Residual statistics
                # ------------------------------------------

                residual_l2_sum += float(
                    np.linalg.norm(
                        residual_action
                    )
                )

                residual_abs_sum += float(
                    np.sum(
                        np.abs(
                            residual_action
                        )
                    )
                )

                residual_element_count += int(
                    residual_action.size
                )

                saturation_count += int(
                    np.count_nonzero(
                        np.abs(
                            residual_action
                        )
                        >= saturation_threshold
                    )
                )

                # ------------------------------------------
                # Environment
                # ------------------------------------------

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    residual_action
                )

                next_observation = np.asarray(
                    next_observation,
                    dtype=np.float32,
                )

                episode_return += float(
                    reward
                )

                executed_steps += 1

                final_info = info

                both_contact = (
                    bool(
                        info.get(
                            "left_contact",
                            False,
                        )
                    )
                    and
                    bool(
                        info.get(
                            "right_contact",
                            False,
                        )
                    )
                )

                grasp_detected = (
                    grasp_detected
                    or both_contact
                )

                cube_lift = float(
                    info.get(
                        "cube_lift_m",
                        0.0,
                    )
                )

                max_cube_lift = max(
                    max_cube_lift,
                    cube_lift,
                )

                observation = (
                    next_observation
                )

                if (
                    terminated
                    or truncated
                ):
                    break

            success = bool(
                final_info.get(
                    "success",
                    False,
                )
            )

            lift_reached = bool(
                max_cube_lift
                >= lift_threshold_m
            )

            if executed_steps > 0:

                mean_residual_l2 = (
                    residual_l2_sum
                    / executed_steps
                )

            else:

                mean_residual_l2 = 0.0

            if residual_element_count > 0:

                mean_abs_residual = (
                    residual_abs_sum
                    / residual_element_count
                )

                saturation_rate = (
                    saturation_count
                    / residual_element_count
                )

            else:

                mean_abs_residual = 0.0
                saturation_rate = 0.0

            rows.append(
                {
                    "angle_deg":
                        float(
                            angle_deg
                        ),

                    "repeat":
                        int(
                            repeat
                        ),

                    "seed":
                        int(
                            episode_seed
                        ),

                    "offset_x_mm":
                        float(
                            offset[0]
                            * 1000.0
                        ),

                    "offset_y_mm":
                        float(
                            offset[1]
                            * 1000.0
                        ),

                    "success":
                        success,

                    "grasp_detected":
                        bool(
                            grasp_detected
                        ),

                    "lift_reached":
                        lift_reached,

                    "max_cube_lift_m":
                        float(
                            max_cube_lift
                        ),

                    "episode_return":
                        float(
                            episode_return
                        ),

                    "mean_residual_l2":
                        float(
                            mean_residual_l2
                        ),

                    "mean_abs_residual":
                        float(
                            mean_abs_residual
                        ),

                    "effective_mean_abs_residual":
                        float(
                            alpha
                            * mean_abs_residual
                        ),

                    "residual_saturation_rate":
                        float(
                            saturation_rate
                        ),

                    "steps":
                        int(
                            executed_steps
                        ),

                    "terminal_reason":
                        final_info.get(
                            "terminal_reason"
                        ),
                }
            )

    # ======================================================
    # Summary
    # ======================================================

    episode_count = len(
        rows
    )

    success_count = sum(
        int(
            row[
                "success"
            ]
        )
        for row in rows
    )

    grasp_count = sum(
        int(
            row[
                "grasp_detected"
            ]
        )
        for row in rows
    )

    lift_count = sum(
        int(
            row[
                "lift_reached"
            ]
        )
        for row in rows
    )

    def mean_field(
        key: str,
    ) -> float:

        if not rows:
            return 0.0

        return float(
            np.mean(
                [
                    row[key]
                    for row in rows
                ]
            )
        )

    summary = {
        "episodes":
            int(
                episode_count
            ),

        "success_count":
            int(
                success_count
            ),

        "success_rate":
            (
                float(
                    success_count
                    / episode_count
                )
                if episode_count
                else 0.0
            ),

        "grasp_count":
            int(
                grasp_count
            ),

        "grasp_rate":
            (
                float(
                    grasp_count
                    / episode_count
                )
                if episode_count
                else 0.0
            ),

        "lift_count":
            int(
                lift_count
            ),

        "lift_rate":
            (
                float(
                    lift_count
                    / episode_count
                )
                if episode_count
                else 0.0
            ),

        "mean_return":
            mean_field(
                "episode_return"
            ),

        "mean_max_cube_lift_m":
            mean_field(
                "max_cube_lift_m"
            ),

        "mean_residual_l2":
            mean_field(
                "mean_residual_l2"
            ),

        "mean_abs_residual":
            mean_field(
                "mean_abs_residual"
            ),

        "effective_mean_abs_residual":
            mean_field(
                "effective_mean_abs_residual"
            ),

        "residual_saturation_rate":
            mean_field(
                "residual_saturation_rate"
            ),
    }

    return (
        summary,
        rows,
    )