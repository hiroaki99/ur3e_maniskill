#!/usr/bin/env python3
"""
Day12 Reference-only perturbation calibration.

目的
----
Cube位置をReferenceに対して
5～10 mm程度ずらし、

Reference only
(residual action = 0)

の成功率が20～60%程度になる
難易度を探索する。

TD3は使用しない。
"""

from __future__ import annotations

import argparse
import csv
from html import parser
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml


# ==========================================================
# Paths
# ==========================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT),
)


from rrl import (
    CubePositionPerturbationWrapper,
    ResidualPickLiftEnv,
)


CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)


DEFAULT_TRAJECTORY = (
    REPO_ROOT
    / "trajectories"
    / "day6_pick_lift_reference_v2.json"
)


# ==========================================================
# Args
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
    )

    parser.add_argument(
        "--radii-mm",
        type=float,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--episodes-per-radius",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--direction-mode",
        type=str,
        default=None,
        choices=[
            "random_angle",
            "x_positive",
            "x_negative",
            "y_positive",
            "y_negative",
        ],
    )

    parser.add_argument(
        "--num-directions",
        type=int,
        default=8,
        help=(
            "random_angle時に使用する"
            "固定評価方向数。"
            "例: 8なら45度刻み"
        ),
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
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/"
            "day12_perturbation_scan"
        ),
    )

    return parser.parse_args()


# ==========================================================
# CSV
# ==========================================================

def write_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ==========================================================
# Evaluation
# ==========================================================

def evaluate_radius(
    *,
    radius_mm: float,
    episodes: int,
    direction_mode: str,
    fixed_angle_deg: float | None,
    config: dict,
    trajectory_path: Path,
    sim_backend: str,
    seed: int,
    max_steps: int,
    render: bool,
    sleep: float,
):

    day9 = config.get(
        "day9",
        {},
    )

    # ------------------------------------------------------
    # Base ManiSkill env
    # ------------------------------------------------------

    import envs.ur3e_pick_lift  # noqa: F401

    base_env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config[
            "robot"
        ][
            "uid"
        ],
        num_envs=1,
        obs_mode="state",
        control_mode=config[
            "project"
        ][
            "control_mode"
        ],
        sim_backend=sim_backend,
        render_mode=(
            "human"
            if render
            else None
        ),
        max_episode_steps=int(
            day9.get(
                "base_env_max_episode_steps",
                1000,
            )
        ),
    )

    radius_m = (
        float(radius_mm)
        / 1000.0
    )

    fixed_angle_rad = (
        None
        if fixed_angle_deg is None
        else float(
            np.deg2rad(
                fixed_angle_deg
            )
        )
    )

    # ------------------------------------------------------
    # Small task mismatch
    # ------------------------------------------------------

    perturb_env = (
        CubePositionPerturbationWrapper(
            base_env,
            radius_min_m=radius_m,
            radius_max_m=radius_m,
            direction_mode=direction_mode,
            fixed_angle_rad=(
                fixed_angle_rad
            ),
            seed=seed,
        )
    )

    # ------------------------------------------------------
    # Residual env
    # ------------------------------------------------------

    env = ResidualPickLiftEnv(
        env=perturb_env,
        trajectory_path=(
            trajectory_path
        ),
        config_path=CONFIG_PATH,
    )

    rows = []

    try:

        zero_residual = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )

        for episode in range(
            episodes
        ):

            episode_seed = (
                seed
                + episode
            )

            (
                observation,
                reset_info,
            ) = env.reset(
                seed=episode_seed
            )

            offset = np.asarray(
                reset_info.get(
                    "cube_offset_m",
                    [
                        0.0,
                        0.0,
                        0.0,
                    ],
                ),
                dtype=np.float64,
            )

            episode_return = 0.0

            grasp_detected = False

            max_cube_lift = (
                -np.inf
            )

            max_left_force = 0.0
            max_right_force = 0.0

            terminated = False
            truncated = False

            final_info = {}

            executed_steps = 0

            for step in range(
                max_steps
            ):

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    zero_residual
                )

                executed_steps += 1

                final_info = (
                    info
                )

                episode_return += float(
                    reward
                )

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

                max_left_force = max(
                    max_left_force,
                    float(
                        info.get(
                            "left_contact_force",
                            0.0,
                        )
                    ),
                )

                max_right_force = max(
                    max_right_force,
                    float(
                        info.get(
                            "right_contact_force",
                            0.0,
                        )
                    ),
                )

                if render:

                    env.render()

                    time.sleep(
                        sleep
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

            final_lift = float(
                final_info.get(
                    "cube_lift_m",
                    0.0,
                )
            )

            row = {
                "radius_mm":
                    float(
                        radius_mm
                    ),

                "angle_deg":
                    (
                        float(
                            fixed_angle_deg
                        )
                        if fixed_angle_deg
                        is not None
                        else None
                    ),

                "episode":
                    int(
                        episode
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

                "offset_radius_mm":
                    float(
                        np.linalg.norm(
                            offset[
                                :2
                            ]
                        )
                        * 1000.0
                    ),

                "success":
                    success,

                "grasp_detected":
                    bool(
                        grasp_detected
                    ),

                "final_cube_lift_m":
                    final_lift,

                "max_cube_lift_m":
                    float(
                        max_cube_lift
                    ),

                "final_both_contact_steps":
                    int(
                        final_info.get(
                            "final_both_contact_steps",
                            0,
                        )
                    ),

                "max_left_force_n":
                    float(
                        max_left_force
                    ),

                "max_right_force_n":
                    float(
                        max_right_force
                    ),

                "episode_return":
                    float(
                        episode_return
                    ),

                "executed_steps":
                    int(
                        executed_steps
                    ),

                "terminal_reason":
                    final_info.get(
                        "terminal_reason"
                    ),

                "terminated":
                    bool(
                        terminated
                    ),

                "truncated":
                    bool(
                        truncated
                    ),
            }

            rows.append(
                row
            )

            angle_text = (
                f"{fixed_angle_deg:6.1f} deg"
                if fixed_angle_deg is not None
                else "random"
            )

            print(
                f"radius="
                f"{radius_mm:5.1f} mm "
                f"angle="
                f"{angle_text} "
                f"episode="
                f"{episode:02d} "
                f"offset="
                f"({row['offset_x_mm']:+6.2f}, "
                f"{row['offset_y_mm']:+6.2f}) mm "
                f"success={success} "
                f"lift={final_lift:.4f} "
                f"reason="
                f"{row['terminal_reason']}"
            )

        return rows

    finally:

        env.close()


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

    day12 = config[
        "day12"
    ]

    radii_mm = (
        args.radii_mm
        if args.radii_mm
        is not None
        else [
            float(value)
            for value
            in day12[
                "candidate_radii_mm"
            ]
        ]
    )

    episodes_per_radius = int(
        args.episodes_per_radius
        if args.episodes_per_radius
        is not None
        else day12[
            "episodes_per_radius"
        ]
    )

    direction_mode = (
        args.direction_mode
        if args.direction_mode
        is not None
        else str(
            day12[
                "direction_mode"
            ]
        )
    )

    seed = int(
        args.seed
        if args.seed
        is not None
        else day12[
            "seed"
        ]
    )

    max_steps = int(
        args.max_steps
        if args.max_steps
        is not None
        else day12[
            "max_steps_per_episode"
        ]
    )

    target_min = float(
        day12[
            "target_success_rate_min"
        ]
    )

    target_max = float(
        day12[
            "target_success_rate_max"
        ]
    )

    baseline_min = float(
        day12[
            "baseline_min_success_rate"
        ]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Day 12 Perturbation Calibration")
    print("=" * 72)

    print(
        "radii [mm]:",
        radii_mm,
    )

    print(
        "episodes/radius:",
        episodes_per_radius,
    )

    print(
        "direction:",
        direction_mode,
    )

    print(
        "target success:",
        target_min,
        "to",
        target_max,
    )

    all_rows = []

    radius_summaries = []

    # ======================================================
    # Fixed evaluation directions
    # ======================================================

    if args.num_directions <= 0:

        raise ValueError(
            "--num-directions must be > 0"
        )

    if (
        direction_mode
        == "random_angle"
    ):

        angles_deg = np.linspace(
            0.0,
            360.0,
            args.num_directions,
            endpoint=False,
        ).tolist()

    else:

        # x_positive等の場合は
        # Wrapper側のdirection_modeをそのまま使用
        angles_deg = [
            None
        ]

    print(
        "fixed angles [deg]:",
        angles_deg,
    )

    # ======================================================
    # Sweep
    # Same directions for every radius
    # ======================================================

    for radius_index, radius_mm in enumerate(
        radii_mm
    ):

        print()
        print("=" * 72)

        print(
            f"Radius: "
            f"{radius_mm:.1f} mm"
        )

        print("=" * 72)

        radius_rows = []

        for (
            angle_index,
            angle_deg,
        ) in enumerate(
            angles_deg
        ):

            print()
            print("-" * 72)

            if angle_deg is not None:

                print(
                    f"Angle: "
                    f"{angle_deg:.1f} deg"
                )

            # ----------------------------------------------
            # IMPORTANT:
            # 同じangle_indexは全radiusで同じseedにする。
            #
            # 以前:
            #   seed + radius_index * 10000
            #
            # 今回:
            #   radiusには依存させない。
            # ----------------------------------------------

            evaluation_seed = (
                seed
                + angle_index
                * 1000
            )

            rows = evaluate_radius(
                radius_mm=radius_mm,
                episodes=(
                    episodes_per_radius
                ),
                direction_mode=(
                    direction_mode
                ),
                fixed_angle_deg=(
                    angle_deg
                ),
                config=config,
                trajectory_path=(
                    args.trajectory
                ),
                sim_backend=(
                    args.sim_backend
                ),
                seed=(
                    evaluation_seed
                ),
                max_steps=max_steps,
                render=args.render,
                sleep=args.sleep,
            )

            radius_rows.extend(
                rows
            )

            all_rows.extend(
                rows
            )

        # ----------------------------------------------
        # Radius summary
        # ----------------------------------------------

        success_count = sum(
            bool(
                row[
                    "success"
                ]
            )
            for row in radius_rows
        )

        grasp_count = sum(
            bool(
                row[
                    "grasp_detected"
                ]
            )
            for row in radius_rows
        )

        success_rate = (
            success_count
            / len(
                radius_rows
            )
        )

        grasp_rate = (
            grasp_count
            / len(
                radius_rows
            )
        )

        mean_lift = float(
            np.mean(
                [
                    row[
                        "final_cube_lift_m"
                    ]
                    for row
                    in radius_rows
                ]
            )
        )

        radius_summary = {
            "radius_mm":
                float(
                    radius_mm
                ),

            "directions":
                int(
                    len(
                        angles_deg
                    )
                ),

            "episodes_per_direction":
                int(
                    episodes_per_radius
                ),

            "total_episodes":
                int(
                    len(
                        radius_rows
                    )
                ),

            "success_count":
                int(
                    success_count
                ),

            "success_rate":
                float(
                    success_rate
                ),

            "grasp_count":
                int(
                    grasp_count
                ),

            "grasp_rate":
                float(
                    grasp_rate
                ),

            "mean_final_lift_m":
                mean_lift,
        }

        radius_summaries.append(
            radius_summary
        )

        print()
        print(
            "radius summary:",
            radius_summary,
        )

    # ======================================================
    # Select difficulty
    # ======================================================

    midpoint = (
        target_min
        + target_max
    ) / 2.0

    eligible = [
        summary
        for summary
        in radius_summaries
        if (
            summary[
                "radius_mm"
            ] > 0.0
            and
            target_min
            <= summary[
                "success_rate"
            ]
            <= target_max
        )
    ]

    selected = None

    if eligible:

        selected = min(
            eligible,
            key=lambda item: (
                abs(
                    item[
                        "success_rate"
                    ]
                    - midpoint
                ),
                item[
                    "radius_mm"
                ],
            ),
        )

    # ======================================================
    # Baseline regression
    # ======================================================

    baseline_summary = next(
        (
            summary
            for summary
            in radius_summaries
            if np.isclose(
                summary[
                    "radius_mm"
                ],
                0.0,
            )
        ),
        None,
    )

    baseline_ok = True

    errors = []

    if (
        baseline_summary
        is not None
    ):

        baseline_ok = (
            baseline_summary[
                "success_rate"
            ]
            >= baseline_min
        )

        if not baseline_ok:

            errors.append(
                "0 mm Reference-only "
                "regression detected: "
                f"success_rate="
                f"{baseline_summary['success_rate']}"
            )

    if selected is None:

        errors.append(
            "No perturbation in the "
            "requested success-rate range"
        )

    passed = (
        baseline_ok
        and selected is not None
    )

    # ======================================================
    # Save
    # ======================================================

    episodes_csv = (
        args.output_dir
        / "episodes.csv"
    )

    write_csv(
        episodes_csv,
        all_rows,
    )

    report = {
        "trajectory":
            str(
                args.trajectory
            ),

        "direction_mode":
            direction_mode,

        "fixed_angles_deg":
            [
                (
                    float(angle)
                    if angle is not None
                    else None
                )
                for angle
                in angles_deg
            ],

        "num_directions":
            int(
                len(
                    angles_deg
                )
            ),

        "candidate_radii_mm":
            [
                float(value)
                for value in radii_mm
            ],

        "episodes_per_radius":
            episodes_per_radius,

        "target_success_rate":
            [
                target_min,
                target_max,
            ],

        "baseline_min_success_rate":
            baseline_min,

        "radius_results":
            radius_summaries,

        "selected_radius_mm":
            (
                float(
                    selected[
                        "radius_mm"
                    ]
                )
                if selected
                is not None
                else None
            ),

        "selected_success_rate":
            (
                float(
                    selected[
                        "success_rate"
                    ]
                )
                if selected
                is not None
                else None
            ),

        "baseline_regression_passed":
            bool(
                baseline_ok
            ),

        "migration_note": {
            "source_yaginuma":
                (
                    "expert policy corrected "
                    "under varying object states"
                ),

            "ported_condition":
                (
                    "fixed Day6 reference with "
                    "small XY cube-position "
                    "mismatch"
                ),
        },

        "errors":
            errors,

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
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ======================================================
    # Print
    # ======================================================

    print()
    print("=" * 72)
    print("Day 12 Result")
    print("=" * 72)

    for summary in radius_summaries:

        print(
            f"{summary['radius_mm']:5.1f} mm : "
            f"{summary['success_count']:3d}/"
            f"{summary['total_episodes']:3d} "
            f"success "
            f"("
            f"{summary['success_rate'] * 100:5.1f}%"
            f") "
            f"grasp="
            f"{summary['grasp_rate'] * 100:5.1f}%"
        )

    print()

    print(
        "selected radius:",
        (
            f"{selected['radius_mm']} mm"
            if selected is not None
            else "NONE"
        ),
    )

    print(
        "selected success rate:",
        (
            selected[
                "success_rate"
            ]
            if selected is not None
            else None
        ),
    )

    print(
        "errors:",
        errors,
    )

    print(
        "PASSED:",
        passed,
    )

    print(
        "summary:",
        summary_path,
    )

    return (
        0
        if passed
        else 2
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )