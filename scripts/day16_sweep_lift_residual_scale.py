#!/usr/bin/env python3
"""
Day16 - Sweep lift residual scale.

Day13 best.ptを固定したまま、

    lift_residual_scale
        1.00
        0.75
        0.50

を比較する。

目的
----
Day15で最多だったLift系失敗

    insufficient_lift
    no_lift_after_grasp

が減少するか確認する。

TD3は再学習しない。
変更するのはLift phaseのResidual scaleだけ。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml


# ==========================================================
# Paths
# ==========================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

SCRIPT_DIR = (
    REPO_ROOT
    / "scripts"
)

sys.path.insert(
    0,
    str(REPO_ROOT),
)

sys.path.insert(
    0,
    str(SCRIPT_DIR),
)


from rrl import (
    LiftResidualScaleWrapper,
)

from day15_classify_failures import (
    CONFIG_PATH,
    DEFAULT_TRAJECTORY,
    classify_failure,
    cube_position,
    make_env as make_day15_env,
    make_policy,
    observation_vector,
    resolve_device,
    write_csv,
)


# ==========================================================
# CLI
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
    )

    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--num-directions",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--episodes-per-direction",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=[
            "physx_cpu",
            "physx_cuda",
        ],
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/"
            "day16_lift_scale_sweep"
        ),
    )

    return parser.parse_args()


# ==========================================================
# Helpers
# ==========================================================

def resolve_repo_path(
    path,
):

    path = Path(
        path
    )

    if path.is_absolute():
        return path

    return (
        REPO_ROOT
        / path
    )


def mean(
    values,
):

    if not values:
        return 0.0

    return float(
        sum(values)
        / len(values)
    )


# ==========================================================
# Episode
# ==========================================================

def run_episode(
    *,
    env,
    perturb_env,
    policy,
    condition,
    scale,
    angle_deg,
    repeat,
    episode_seed,
    max_steps,
    day15,
    saturation_threshold,
):

    perturb_env.fixed_angle_rad = float(
        np.deg2rad(
            angle_deg
        )
    )

    observation, reset_info = (
        env.reset(
            seed=episode_seed
        )
    )

    observation = observation_vector(
        observation
    )

    action_dim = int(
        env.action_space.shape[0]
    )

    initial_cube_position = (
        cube_position(
            observation
        )
    )

    # ------------------------------------------------------
    # Contact
    # ------------------------------------------------------

    ever_left = False
    ever_right = False
    ever_bilateral = False

    current_bilateral_run = 0
    max_bilateral_run = 0

    reached_lift_threshold = False
    lost_contact_after_lift = False

    # ------------------------------------------------------
    # Cube
    # ------------------------------------------------------

    max_cube_lift = 0.0
    final_cube_lift = 0.0

    max_xy_displacement = 0.0

    # ------------------------------------------------------
    # Safety
    # ------------------------------------------------------

    unsafe_collision = False
    unsafe_collision_observable = False

    # ------------------------------------------------------
    # Residual statistics
    # ------------------------------------------------------

    raw_l2_sum = 0.0
    applied_l2_sum = 0.0

    raw_abs_sum = 0.0
    applied_abs_sum = 0.0

    raw_saturation_count = 0
    applied_saturation_count = 0

    residual_element_count = 0

    lift_raw_l2_sum = 0.0
    lift_applied_l2_sum = 0.0

    lift_applied_saturation_count = 0
    lift_element_count = 0
    lift_steps = 0

    # ------------------------------------------------------
    # General
    # ------------------------------------------------------

    episode_return = 0.0

    final_info = {}

    terminated = False
    truncated = False

    executed_steps = 0

    lift_threshold = float(
        day15[
            "lift_threshold_m"
        ]
    )

    # ======================================================
    # Loop
    # ======================================================

    for _ in range(
        max_steps
    ):

        # --------------------------------------------------
        # Policy
        # --------------------------------------------------

        if policy is None:

            raw_action = np.zeros(
                action_dim,
                dtype=np.float32,
            )

        else:

            raw_action = np.asarray(
                policy.select_action(
                    observation
                ),
                dtype=np.float32,
            )

            raw_action = np.clip(
                raw_action,
                -1.0,
                1.0,
            )

        # --------------------------------------------------
        # Environment
        # --------------------------------------------------

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            raw_action
        )

        next_observation = (
            observation_vector(
                next_observation
            )
        )

        executed_steps += 1

        episode_return += float(
            reward
        )

        final_info = info

        # --------------------------------------------------
        # Raw / applied residual
        # --------------------------------------------------

        raw_logged = np.asarray(
            info.get(
                "raw_residual_action",
                raw_action,
            ),
            dtype=np.float32,
        )

        applied = np.asarray(
            info.get(
                "applied_residual_action",
                raw_action,
            ),
            dtype=np.float32,
        )

        phase_index = int(
            info.get(
                "residual_phase_index",
                -1,
            )
        )

        raw_l2_sum += float(
            np.linalg.norm(
                raw_logged
            )
        )

        applied_l2_sum += float(
            np.linalg.norm(
                applied
            )
        )

        raw_abs_sum += float(
            np.sum(
                np.abs(
                    raw_logged
                )
            )
        )

        applied_abs_sum += float(
            np.sum(
                np.abs(
                    applied
                )
            )
        )

        raw_saturation_count += int(
            np.count_nonzero(
                np.abs(
                    raw_logged
                )
                >= saturation_threshold
            )
        )

        applied_saturation_count += int(
            np.count_nonzero(
                np.abs(
                    applied
                )
                >= saturation_threshold
            )
        )

        residual_element_count += int(
            applied.size
        )

        # Lift phase = 4
        if phase_index == 4:

            lift_steps += 1

            lift_raw_l2_sum += float(
                np.linalg.norm(
                    raw_logged
                )
            )

            lift_applied_l2_sum += float(
                np.linalg.norm(
                    applied
                )
            )

            lift_applied_saturation_count += int(
                np.count_nonzero(
                    np.abs(
                        applied
                    )
                    >= saturation_threshold
                )
            )

            lift_element_count += int(
                applied.size
            )

        # --------------------------------------------------
        # Contact
        # --------------------------------------------------

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

        bilateral = (
            left_contact
            and right_contact
        )

        ever_left = (
            ever_left
            or left_contact
        )

        ever_right = (
            ever_right
            or right_contact
        )

        ever_bilateral = (
            ever_bilateral
            or bilateral
        )

        if bilateral:

            current_bilateral_run += 1

            max_bilateral_run = max(
                max_bilateral_run,
                current_bilateral_run,
            )

        else:

            if reached_lift_threshold:
                lost_contact_after_lift = (
                    True
                )

            current_bilateral_run = 0

        # --------------------------------------------------
        # Cube state
        # --------------------------------------------------

        current_cube_position = (
            cube_position(
                next_observation
            )
        )

        xy_displacement = float(
            np.linalg.norm(
                current_cube_position[:2]
                - initial_cube_position[:2]
            )
        )

        max_xy_displacement = max(
            max_xy_displacement,
            xy_displacement,
        )

        cube_lift = float(
            info.get(
                "cube_lift_m",
                (
                    current_cube_position[2]
                    - initial_cube_position[2]
                ),
            )
        )

        final_cube_lift = (
            cube_lift
        )

        max_cube_lift = max(
            max_cube_lift,
            cube_lift,
        )

        if (
            cube_lift
            >= lift_threshold
        ):

            reached_lift_threshold = (
                True
            )

        # --------------------------------------------------
        # Collision
        # --------------------------------------------------

        if "unsafe_collision" in info:

            unsafe_collision_observable = (
                True
            )

            unsafe_collision = (
                unsafe_collision
                or bool(
                    info[
                        "unsafe_collision"
                    ]
                )
            )

        observation = (
            next_observation
        )

        if (
            terminated
            or truncated
        ):
            break

    # ======================================================
    # Episode metrics
    # ======================================================

    success = bool(
        final_info.get(
            "success",
            False,
        )
    )

    timeout = bool(
        not terminated
        and not truncated
        and executed_steps
        >= max_steps
    )

    metrics = {
        "condition":
            condition,

        "lift_residual_scale":
            float(
                scale
            ),

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

        "success":
            success,

        "episode_return":
            float(
                episode_return
            ),

        "steps":
            int(
                executed_steps
            ),

        "ever_left_contact":
            ever_left,

        "ever_right_contact":
            ever_right,

        "ever_bilateral_contact":
            ever_bilateral,

        "max_consecutive_bilateral_steps":
            int(
                max_bilateral_run
            ),

        "max_cube_lift_m":
            float(
                max_cube_lift
            ),

        "final_cube_lift_m":
            float(
                final_cube_lift
            ),

        "lost_contact_after_lift":
            bool(
                lost_contact_after_lift
            ),

        "max_cube_xy_displacement_m":
            float(
                max_xy_displacement
            ),

        "unsafe_collision_observable":
            bool(
                unsafe_collision_observable
            ),

        "unsafe_collision":
            bool(
                unsafe_collision
            ),

        "terminal_reason":
            final_info.get(
                "terminal_reason"
            ),

        "timeout":
            timeout,

        # ----------------------------------------------
        # Residual
        # ----------------------------------------------

        "mean_raw_residual_l2":
            float(
                raw_l2_sum
                / max(
                    executed_steps,
                    1,
                )
            ),

        "mean_applied_residual_l2":
            float(
                applied_l2_sum
                / max(
                    executed_steps,
                    1,
                )
            ),

        "mean_raw_abs_residual":
            float(
                raw_abs_sum
                / max(
                    residual_element_count,
                    1,
                )
            ),

        "mean_applied_abs_residual":
            float(
                applied_abs_sum
                / max(
                    residual_element_count,
                    1,
                )
            ),

        "raw_saturation_rate":
            float(
                raw_saturation_count
                / max(
                    residual_element_count,
                    1,
                )
            ),

        "applied_saturation_rate":
            float(
                applied_saturation_count
                / max(
                    residual_element_count,
                    1,
                )
            ),

        # ----------------------------------------------
        # Lift-specific residual
        # ----------------------------------------------

        "lift_steps":
            int(
                lift_steps
            ),

        "lift_mean_raw_residual_l2":
            float(
                lift_raw_l2_sum
                / max(
                    lift_steps,
                    1,
                )
            ),

        "lift_mean_applied_residual_l2":
            float(
                lift_applied_l2_sum
                / max(
                    lift_steps,
                    1,
                )
            ),

        "lift_applied_saturation_rate":
            float(
                lift_applied_saturation_count
                / max(
                    lift_element_count,
                    1,
                )
            ),
    }

    metrics[
        "failure_cause"
    ] = classify_failure(
        metrics,
        day15,
    )

    return metrics


# ==========================================================
# Evaluation
# ==========================================================

def evaluate_condition(
    *,
    config,
    trajectory,
    checkpoint,
    device,
    sim_backend,
    radius_mm,
    scale,
    angles_deg,
    episodes_per_direction,
    eval_seed,
    max_steps,
    day15,
    saturation_threshold,
    reference=False,
):

    inner_env, perturb_env = (
        make_day15_env(
            config=config,
            trajectory=trajectory,
            sim_backend=sim_backend,
            radius_mm=radius_mm,
            seed=eval_seed,
        )
    )

    env = LiftResidualScaleWrapper(
        inner_env,
        lift_residual_scale=scale,
    )

    try:

        env.reset(
            seed=eval_seed
        )

        if reference:

            policy = None
            condition = "reference"

        else:

            policy = make_policy(
                env=env,
                config=config,
                device=device,
            )

            policy.load(
                checkpoint
            )

            condition = (
                "residual_td3"
            )

        rows = []

        for (
            angle_index,
            angle_deg,
        ) in enumerate(
            angles_deg
        ):

            for repeat in range(
                episodes_per_direction
            ):

                episode_seed = (
                    eval_seed
                    + angle_index
                    * 1000
                    + repeat
                )

                row = run_episode(
                    env=env,
                    perturb_env=perturb_env,
                    policy=policy,
                    condition=condition,
                    scale=scale,
                    angle_deg=angle_deg,
                    repeat=repeat,
                    episode_seed=episode_seed,
                    max_steps=max_steps,
                    day15=day15,
                    saturation_threshold=(
                        saturation_threshold
                    ),
                )

                rows.append(
                    row
                )

        return rows

    finally:

        env.close()


# ==========================================================
# Summary
# ==========================================================

def summarize_scale(
    *,
    scale,
    rows,
    reference_rows,
    target_failure_causes,
):

    failure_counter = Counter(
        row[
            "failure_cause"
        ]
        for row
        in rows
        if not row[
            "success"
        ]
    )

    reference_map = {
        (
            row[
                "angle_deg"
            ],
            row[
                "repeat"
            ],
        ): row
        for row
        in reference_rows
    }

    candidate_map = {
        (
            row[
                "angle_deg"
            ],
            row[
                "repeat"
            ],
        ): row
        for row
        in rows
    }

    rescued = 0
    broken = 0
    kept_success = 0
    kept_failure = 0

    for key in reference_map:

        ref_success = bool(
            reference_map[
                key
            ][
                "success"
            ]
        )

        candidate_success = bool(
            candidate_map[
                key
            ][
                "success"
            ]
        )

        if (
            not ref_success
            and candidate_success
        ):

            rescued += 1

        elif (
            ref_success
            and not candidate_success
        ):

            broken += 1

        elif (
            ref_success
            and candidate_success
        ):

            kept_success += 1

        else:

            kept_failure += 1

    success_count = sum(
        int(
            row[
                "success"
            ]
        )
        for row
        in rows
    )

    target_failure_count = sum(
        failure_counter.get(
            cause,
            0,
        )
        for cause
        in target_failure_causes
    )

    return {
        "lift_residual_scale":
            float(
                scale
            ),

        "episodes":
            len(
                rows
            ),

        "success_count":
            int(
                success_count
            ),

        "success_rate":
            float(
                success_count
                / len(
                    rows
                )
            ),

        "failure_counts":
            dict(
                failure_counter
            ),

        "target_lift_failure_count":
            int(
                target_failure_count
            ),

        "target_lift_failure_rate":
            float(
                target_failure_count
                / len(
                    rows
                )
            ),

        "rescued":
            int(
                rescued
            ),

        "broken":
            int(
                broken
            ),

        "kept_success":
            int(
                kept_success
            ),

        "kept_failure":
            int(
                kept_failure
            ),

        "net_gain":
            int(
                rescued
                - broken
            ),

        "mean_raw_residual_l2":
            mean(
                [
                    row[
                        "mean_raw_residual_l2"
                    ]
                    for row
                    in rows
                ]
            ),

        "mean_applied_residual_l2":
            mean(
                [
                    row[
                        "mean_applied_residual_l2"
                    ]
                    for row
                    in rows
                ]
            ),

        "applied_saturation_rate":
            mean(
                [
                    row[
                        "applied_saturation_rate"
                    ]
                    for row
                    in rows
                ]
            ),

        "lift_mean_raw_residual_l2":
            mean(
                [
                    row[
                        "lift_mean_raw_residual_l2"
                    ]
                    for row
                    in rows
                ]
            ),

        "lift_mean_applied_residual_l2":
            mean(
                [
                    row[
                        "lift_mean_applied_residual_l2"
                    ]
                    for row
                    in rows
                ]
            ),

        "lift_applied_saturation_rate":
            mean(
                [
                    row[
                        "lift_applied_saturation_rate"
                    ]
                    for row
                    in rows
                ]
            ),
    }


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    day15 = config[
        "day15"
    ]

    day16 = config[
        "day16"
    ]

    checkpoint = (
        args.checkpoint
        if args.checkpoint
        is not None
        else Path(
            day16[
                "checkpoint"
            ]
        )
    )

    checkpoint = resolve_repo_path(
        checkpoint
    )

    trajectory = resolve_repo_path(
        args.trajectory
    )

    scales = (
        args.scales
        if args.scales
        is not None
        else day16[
            "candidate_lift_residual_scales"
        ]
    )

    scales = [
        float(
            scale
        )
        for scale
        in scales
    ]

    if 1.0 not in scales:

        raise ValueError(
            "scales must contain 1.0 "
            "as Day15 baseline"
        )

    radius_mm = float(
        day16[
            "radius_mm"
        ]
    )

    num_directions = int(
        args.num_directions
        if args.num_directions
        is not None
        else day16[
            "num_directions"
        ]
    )

    episodes_per_direction = int(
        args.episodes_per_direction
        if args.episodes_per_direction
        is not None
        else day16[
            "episodes_per_direction"
        ]
    )

    eval_seed = int(
        day16[
            "eval_seed"
        ]
    )

    max_steps = int(
        day16[
            "max_steps_per_episode"
        ]
    )

    saturation_threshold = float(
        day16[
            "residual_saturation_threshold"
        ]
    )

    target_failure_causes = list(
        day16[
            "target_failure_causes"
        ]
    )

    device = resolve_device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    angles_deg = np.linspace(
        0.0,
        360.0,
        num_directions,
        endpoint=False,
    ).tolist()

    # ======================================================
    # Reference
    # ======================================================

    print("=" * 72)
    print("Day 16 Lift Residual Scale Sweep")
    print("=" * 72)

    print(
        "checkpoint:",
        checkpoint,
    )

    print(
        "scales:",
        scales,
    )

    print(
        "directions:",
        num_directions,
    )

    print()

    reference_rows = (
        evaluate_condition(
            config=config,
            trajectory=trajectory,
            checkpoint=checkpoint,
            device=device,
            sim_backend=args.sim_backend,
            radius_mm=radius_mm,
            scale=1.0,
            angles_deg=angles_deg,
            episodes_per_direction=(
                episodes_per_direction
            ),
            eval_seed=eval_seed,
            max_steps=max_steps,
            day15=day15,
            saturation_threshold=(
                saturation_threshold
            ),
            reference=True,
        )
    )

    reference_success = sum(
        int(
            row[
                "success"
            ]
        )
        for row
        in reference_rows
    )

    print(
        "Reference:",
        f"{reference_success}/"
        f"{len(reference_rows)}",
        f"="
        f"{reference_success / len(reference_rows):.3f}",
    )

    # ======================================================
    # Scale sweep
    # ======================================================

    all_episode_rows = list(
        reference_rows
    )

    scale_summaries = []

    for scale in scales:

        print()
        print("-" * 72)

        print(
            "lift_residual_scale =",
            scale,
        )

        rows = evaluate_condition(
            config=config,
            trajectory=trajectory,
            checkpoint=checkpoint,
            device=device,
            sim_backend=args.sim_backend,
            radius_mm=radius_mm,
            scale=scale,
            angles_deg=angles_deg,
            episodes_per_direction=(
                episodes_per_direction
            ),
            eval_seed=eval_seed,
            max_steps=max_steps,
            day15=day15,
            saturation_threshold=(
                saturation_threshold
            ),
            reference=False,
        )

        all_episode_rows.extend(
            rows
        )

        summary = summarize_scale(
            scale=scale,
            rows=rows,
            reference_rows=reference_rows,
            target_failure_causes=(
                target_failure_causes
            ),
        )

        scale_summaries.append(
            summary
        )

        print(
            "success:",
            f"{summary['success_count']}/"
            f"{summary['episodes']}",
            f"="
            f"{summary['success_rate']:.3f}",
        )

        print(
            "failure counts:",
            summary[
                "failure_counts"
            ],
        )

        print(
            "target lift failures:",
            summary[
                "target_lift_failure_count"
            ],
        )

        print(
            "rescued:",
            summary[
                "rescued"
            ],
        )

        print(
            "broken:",
            summary[
                "broken"
            ],
        )

        print(
            "lift applied L2:",
            f"{summary['lift_mean_applied_residual_l2']:.3f}",
        )

        print(
            "lift saturation:",
            f"{summary['lift_applied_saturation_rate']:.3f}",
        )

    # ======================================================
    # Selection
    # ======================================================

    baseline = next(
        summary
        for summary
        in scale_summaries
        if abs(
            summary[
                "lift_residual_scale"
            ]
            - 1.0
        )
        < 1e-9
    )

    eligible = [
        summary
        for summary
        in scale_summaries
        if (
            summary[
                "lift_residual_scale"
            ]
            < 1.0
            and
            summary[
                "success_rate"
            ]
            >= baseline[
                "success_rate"
            ]
            and
            summary[
                "target_lift_failure_count"
            ]
            <
            baseline[
                "target_lift_failure_count"
            ]
        )
    ]

    if eligible:

        selected = max(
            eligible,
            key=lambda row: (
                row[
                    "success_rate"
                ],
                -row[
                    "target_lift_failure_count"
                ],
                -row[
                    "broken"
                ],
                -row[
                    "mean_applied_residual_l2"
                ],
            ),
        )

        selected_scale = float(
            selected[
                "lift_residual_scale"
            ]
        )

        passed = True

    else:

        selected = None
        selected_scale = None
        passed = False

    # ======================================================
    # Save
    # ======================================================

    write_csv(
        args.output_dir
        / "episodes.csv",
        all_episode_rows,
    )

    # failure_countsはdictなのでCSVでは文字列化
    scale_summary_rows = []

    for summary in scale_summaries:

        row = dict(
            summary
        )

        row[
            "failure_counts"
        ] = json.dumps(
            row[
                "failure_counts"
            ],
            ensure_ascii=False,
        )

        scale_summary_rows.append(
            row
        )

    write_csv(
        args.output_dir
        / "scale_summary.csv",
        scale_summary_rows,
    )

    final_summary = {
        "checkpoint":
            str(
                checkpoint
            ),

        "radius_mm":
            radius_mm,

        "num_directions":
            num_directions,

        "episodes_per_direction":
            episodes_per_direction,

        "reference_success_rate":
            float(
                reference_success
                / len(
                    reference_rows
                )
            ),

        "target_failure_causes":
            target_failure_causes,

        "baseline_scale":
            1.0,

        "baseline":
            baseline,

        "candidates":
            scale_summaries,

        "selection_rule": (
            "Success rate must not be lower than scale=1.0, "
            "and target lift failures must decrease. "
            "Then maximize success, minimize target failures, "
            "broken cases, and applied residual."
        ),

        "selected_lift_residual_scale":
            selected_scale,

        "passed":
            passed,
    }

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ======================================================
    # Console
    # ======================================================

    print()
    print("=" * 72)
    print("Day 16 Result")
    print("=" * 72)

    for summary in scale_summaries:

        print(
            f"scale="
            f"{summary['lift_residual_scale']:.2f} "
            f"success="
            f"{summary['success_rate'] * 100:5.1f}% "
            f"lift_fail="
            f"{summary['target_lift_failure_count']:2d} "
            f"rescued="
            f"{summary['rescued']:2d} "
            f"broken="
            f"{summary['broken']:2d} "
            f"lift_L2="
            f"{summary['lift_mean_applied_residual_l2']:.3f} "
            f"lift_sat="
            f"{summary['lift_applied_saturation_rate']:.3f}"
        )

    print()

    print(
        "selected scale:",
        selected_scale,
    )

    print(
        "PASSED:",
        passed,
    )

    print(
        "summary:",
        summary_path,
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )