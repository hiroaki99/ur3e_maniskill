#!/usr/bin/env python3
"""
Day 14 Gate 2 evaluation.

Reference-only と Day13 best TD3 を、
同じ32方向・同じseedで比較する。

Outputs
-------
summary.json
episodes.csv
paired_results.csv
angle_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT),
)


from rrl import (
    CubePositionPerturbationWrapper,
    ResidualPickLiftEnv,
    TD3,
    evaluate_residual_policy,
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
# Utilities
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
            "reports/day14_gate2"
        ),
    )

    return parser.parse_args()


def cfg_get(
    config,
    *paths,
    default=None,
):

    for path in paths:

        current = config

        try:

            for key in path:
                current = current[key]

            return current

        except (KeyError, TypeError):
            pass

    return default


def resolve_device(
    requested,
):

    if requested == "auto":

        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    return requested


def write_csv(
    path: Path,
    rows,
):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for row in rows:

        for key in row.keys():

            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def make_env(
    *,
    config,
    trajectory,
    sim_backend,
    radius_mm,
    seed,
):

    import envs.ur3e_pick_lift  # noqa: F401

    robot_uid = cfg_get(
        config,
        ("robot", "uid"),
        default="ur3e_ezgripper",
    )

    control_mode = cfg_get(
        config,
        ("project", "control_mode"),
        ("environment", "control_mode"),
        default="pd_joint_delta_pos",
    )

    base_max_steps = int(
        cfg_get(
            config,
            (
                "day9",
                "base_env_max_episode_steps",
            ),
            default=1000,
        )
    )

    base_env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=robot_uid,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend=sim_backend,
        max_episode_steps=base_max_steps,
    )

    radius_m = (
        float(radius_mm)
        / 1000.0
    )

    perturb_env = (
        CubePositionPerturbationWrapper(
            base_env,
            radius_min_m=radius_m,
            radius_max_m=radius_m,
            direction_mode="random_angle",
            fixed_angle_rad=None,
            seed=seed,
        )
    )

    env = ResidualPickLiftEnv(
        env=perturb_env,
        trajectory_path=trajectory,
        config_path=CONFIG_PATH,
    )

    return env, perturb_env


def make_policy(
    *,
    env,
    config,
    device,
):

    day11 = config["day11"]

    state_dim = int(
        env.observation_space.shape[0]
    )

    action_dim = int(
        env.action_space.shape[0]
    )

    max_action = float(
        env.action_space.high[0]
    )

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,

        actor_hidden_dim=int(
            day11["actor_hidden_dim"]
        ),

        critic_hidden_dim=int(
            day11["critic_hidden_dim"]
        ),

        discount=float(
            day11["discount"]
        ),

        tau=float(
            day11["tau"]
        ),

        policy_noise=float(
            day11["policy_noise"]
        ),

        noise_clip=float(
            day11["noise_clip"]
        ),

        policy_freq=int(
            day11["policy_freq"]
        ),

        actor_lr=float(
            day11["actor_lr"]
        ),

        critic_lr=float(
            day11["critic_lr"]
        ),

        weight_decay=float(
            day11["weight_decay"]
        ),

        device=device,

        seed=int(
            config["day13"]["train_seed"]
        ),
    )

    return policy


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

    day13 = config["day13"]
    day14 = config["day14"]

    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else REPO_ROOT
        / day14["checkpoint"]
    )

    radius_mm = float(
        day14["radius_mm"]
    )

    num_directions = int(
        args.num_directions
        if args.num_directions is not None
        else day14["num_directions"]
    )

    episodes_per_direction = int(
        args.episodes_per_direction
        if args.episodes_per_direction is not None
        else day14[
            "episodes_per_direction"
        ]
    )

    eval_seed = int(
        day14["eval_seed"]
    )

    max_steps = int(
        day14[
            "max_steps_per_episode"
        ]
    )

    gate_min_success = float(
        day14[
            "gate_min_success_rate"
        ]
    )

    gate_min_improvement_pt = float(
        day14[
            "gate_min_improvement_percentage_points"
        ]
    )

    saturation_threshold = float(
        day14[
            "residual_saturation_threshold"
        ]
    )

    device = resolve_device(
        args.device
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env, perturb_env = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
    )

    try:

        # 状態次元を確定
        env.reset(
            seed=eval_seed
        )

        policy = make_policy(
            env=env,
            config=config,
            device=device,
        )

        policy.load(
            checkpoint
        )

        angles_deg = np.linspace(
            0.0,
            360.0,
            num_directions,
            endpoint=False,
        ).tolist()

        common_kwargs = dict(
            env=env,
            perturbation_wrapper=perturb_env,
            angles_deg=angles_deg,
            episodes_per_direction=(
                episodes_per_direction
            ),
            seed=eval_seed,
            max_steps=max_steps,
            lift_threshold_m=float(
                day13["lift_threshold_m"]
            ),
            saturation_threshold=(
                saturation_threshold
            ),
        )

        # ==================================================
        # Reference-only
        # ==================================================

        reference_summary, reference_rows = (
            evaluate_residual_policy(
                policy=None,
                **common_kwargs,
            )
        )

        # ==================================================
        # Residual TD3
        # ==================================================

        residual_summary, residual_rows = (
            evaluate_residual_policy(
                policy=policy,
                **common_kwargs,
            )
        )

        for row in reference_rows:
            row["condition"] = "reference"

        for row in residual_rows:
            row["condition"] = "residual_td3"

        # ==================================================
        # Paired analysis
        # ==================================================

        ref_map = {
            (
                float(row["angle_deg"]),
                int(row["repeat"]),
            ): row
            for row in reference_rows
        }

        residual_map = {
            (
                float(row["angle_deg"]),
                int(row["repeat"]),
            ): row
            for row in residual_rows
        }

        paired_rows = []

        rescued_count = 0
        broken_count = 0
        kept_success_count = 0
        kept_failure_count = 0

        for key in sorted(
            ref_map.keys()
        ):

            ref = ref_map[key]
            residual = residual_map[key]

            ref_success = bool(
                ref["success"]
            )

            residual_success = bool(
                residual["success"]
            )

            if (
                not ref_success
                and residual_success
            ):

                category = "rescued"
                rescued_count += 1

            elif (
                ref_success
                and not residual_success
            ):

                category = "broken"
                broken_count += 1

            elif (
                ref_success
                and residual_success
            ):

                category = "kept_success"
                kept_success_count += 1

            else:

                category = "kept_failure"
                kept_failure_count += 1

            paired_rows.append(
                {
                    "angle_deg":
                        float(
                            ref["angle_deg"]
                        ),

                    "repeat":
                        int(
                            ref["repeat"]
                        ),

                    "offset_x_mm":
                        float(
                            ref["offset_x_mm"]
                        ),

                    "offset_y_mm":
                        float(
                            ref["offset_y_mm"]
                        ),

                    "reference_success":
                        ref_success,

                    "residual_success":
                        residual_success,

                    "category":
                        category,

                    "reference_grasp":
                        bool(
                            ref[
                                "grasp_detected"
                            ]
                        ),

                    "residual_grasp":
                        bool(
                            residual[
                                "grasp_detected"
                            ]
                        ),

                    "reference_lift":
                        bool(
                            ref[
                                "lift_reached"
                            ]
                        ),

                    "residual_lift":
                        bool(
                            residual[
                                "lift_reached"
                            ]
                        ),

                    "reference_return":
                        float(
                            ref[
                                "episode_return"
                            ]
                        ),

                    "residual_return":
                        float(
                            residual[
                                "episode_return"
                            ]
                        ),

                    "return_difference":
                        float(
                            residual[
                                "episode_return"
                            ]
                            - ref[
                                "episode_return"
                            ]
                        ),

                    "residual_l2":
                        float(
                            residual[
                                "mean_residual_l2"
                            ]
                        ),

                    "residual_mean_abs":
                        float(
                            residual[
                                "mean_abs_residual"
                            ]
                        ),

                    "residual_saturation_rate":
                        float(
                            residual[
                                "residual_saturation_rate"
                            ]
                        ),

                    "reference_terminal_reason":
                        ref.get(
                            "terminal_reason"
                        ),

                    "residual_terminal_reason":
                        residual.get(
                            "terminal_reason"
                        ),
                }
            )

        # 1 repeatならangle summaryとpairedは同じだが、
        # 将来repeatを増やした場合に備えて分離して保存する。
        angle_summary_rows = []

        for angle_deg in angles_deg:

            rows = [
                row
                for row in paired_rows
                if abs(
                    row["angle_deg"]
                    - angle_deg
                ) < 1e-6
            ]

            if not rows:
                continue

            angle_summary_rows.append(
                {
                    "angle_deg":
                        float(
                            angle_deg
                        ),

                    "episodes":
                        len(rows),

                    "reference_success_rate":
                        float(
                            np.mean(
                                [
                                    float(
                                        row[
                                            "reference_success"
                                        ]
                                    )
                                    for row
                                    in rows
                                ]
                            )
                        ),

                    "residual_success_rate":
                        float(
                            np.mean(
                                [
                                    float(
                                        row[
                                            "residual_success"
                                        ]
                                    )
                                    for row
                                    in rows
                                ]
                            )
                        ),

                    "residual_mean_l2":
                        float(
                            np.mean(
                                [
                                    row[
                                        "residual_l2"
                                    ]
                                    for row
                                    in rows
                                ]
                            )
                        ),

                    "residual_saturation_rate":
                        float(
                            np.mean(
                                [
                                    row[
                                        "residual_saturation_rate"
                                    ]
                                    for row
                                    in rows
                                ]
                            )
                        ),
                }
            )

        # ==================================================
        # Gate 2
        # ==================================================

        reference_success_rate = float(
            reference_summary[
                "success_rate"
            ]
        )

        residual_success_rate = float(
            residual_summary[
                "success_rate"
            ]
        )

        improvement = (
            residual_success_rate
            - reference_success_rate
        )

        improvement_pt = (
            improvement
            * 100.0
        )

        gate_by_success = bool(
            residual_success_rate
            >= gate_min_success
        )

        gate_by_improvement = bool(
            improvement_pt
            >= gate_min_improvement_pt
        )

        passed = bool(
            gate_by_success
            or gate_by_improvement
        )

        summary = {
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

            "episodes_per_condition":
                int(
                    num_directions
                    * episodes_per_direction
                ),

            "reference":
                reference_summary,

            "residual_td3":
                residual_summary,

            "success_improvement":
                float(
                    improvement
                ),

            "success_improvement_percentage_points":
                float(
                    improvement_pt
                ),

            "paired_analysis": {
                "rescued":
                    rescued_count,

                "broken":
                    broken_count,

                "kept_success":
                    kept_success_count,

                "kept_failure":
                    kept_failure_count,

                "net_gain":
                    (
                        rescued_count
                        - broken_count
                    ),
            },

            "gate": {
                "min_success_rate":
                    gate_min_success,

                "min_improvement_percentage_points":
                    gate_min_improvement_pt,

                "passed_by_success_rate":
                    gate_by_success,

                "passed_by_improvement":
                    gate_by_improvement,

                "passed":
                    passed,
            },

            "migration_note": (
                "Conceptual Yaginuma-style RRL migration: "
                "Day6 reference trajectory + "
                "6-DoF normalized joint residual TD3."
            ),
        }

        # ==================================================
        # Save
        # ==================================================

        write_csv(
            output_dir
            / "episodes.csv",
            reference_rows
            + residual_rows,
        )

        write_csv(
            output_dir
            / "paired_results.csv",
            paired_rows,
        )

        write_csv(
            output_dir
            / "angle_summary.csv",
            angle_summary_rows,
        )

        with (
            output_dir
            / "summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # ==================================================
        # Console
        # ==================================================

        print("=" * 72)
        print("Day 14 Gate 2 Evaluation")
        print("=" * 72)

        print(
            "checkpoint:",
            checkpoint,
        )

        print(
            "radius [mm]:",
            radius_mm,
        )

        print(
            "directions:",
            num_directions,
        )

        print(
            "episodes/condition:",
            num_directions
            * episodes_per_direction,
        )

        print()

        print(
            "Reference success:",
            reference_success_rate,
        )

        print(
            "Residual success:",
            residual_success_rate,
        )

        print(
            "Improvement [pt]:",
            improvement_pt,
        )

        print()

        print(
            "Rescued:",
            rescued_count,
        )

        print(
            "Broken:",
            broken_count,
        )

        print(
            "Net gain:",
            rescued_count
            - broken_count,
        )

        print()

        print(
            "Residual grasp:",
            residual_summary[
                "grasp_rate"
            ],
        )

        print(
            "Residual lift:",
            residual_summary[
                "lift_rate"
            ],
        )

        print(
            "Residual L2:",
            residual_summary[
                "mean_residual_l2"
            ],
        )

        print(
            "Saturation:",
            residual_summary[
                "residual_saturation_rate"
            ],
        )

        print()

        print(
            "Gate by success:",
            gate_by_success,
        )

        print(
            "Gate by improvement:",
            gate_by_improvement,
        )

        print(
            "GATE 2 PASSED:",
            passed,
        )

        print(
            "summary:",
            output_dir
            / "summary.json",
        )

        return (
            0
            if passed
            else 2
        )

    finally:

        env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )