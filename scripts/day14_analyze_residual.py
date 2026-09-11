#!/usr/bin/env python3
"""
Day14 Residual diagnostics.

Day13 best TD3について、

- phase別 residual
- joint別 residual
- phase × joint residual
- saturation

を32方向で解析する。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml


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
    TD3,
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


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


PHASE_NAMES = [
    "to_pregrasp",
    "descend",
    "grasp",
    "stable_hold",
    "lift",
    "final_hold",
    "terminal",
]


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
            "reports/day14_residual_analysis"
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

    return requested


def write_csv(
    path,
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

        for key in row:

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


def phase_from_observation(
    observation,
):

    observation = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1)

    # Day9 observation contract:
    # phase one-hot = [33:40]
    if observation.size >= 40:

        phase_onehot = observation[
            33:40
        ]

        index = int(
            np.argmax(
                phase_onehot
            )
        )

        if (
            0
            <= index
            < len(PHASE_NAMES)
        ):

            return PHASE_NAMES[
                index
            ]

    return "unknown"


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

    base_env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=robot_uid,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend=sim_backend,
        max_episode_steps=int(
            cfg_get(
                config,
                (
                    "day9",
                    "base_env_max_episode_steps",
                ),
                default=1000,
            )
        ),
    )

    radius_m = (
        radius_mm
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
    env,
    config,
    device,
):

    day11 = config["day11"]

    policy = TD3(
        state_dim=int(
            env.observation_space.shape[0]
        ),

        action_dim=int(
            env.action_space.shape[0]
        ),

        max_action=float(
            env.action_space.high[0]
        ),

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


def mean(
    values,
):

    if not values:
        return 0.0

    return float(
        sum(values)
        / len(values)
    )


def main():

    args = parse_args()

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

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

    eval_seed = int(
        day14["eval_seed"]
    )

    max_steps = int(
        day14[
            "max_steps_per_episode"
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

    args.output_dir.mkdir(
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

        observation, _ = env.reset(
            seed=eval_seed
        )

        policy = make_policy(
            env,
            config,
            device,
        )

        policy.load(
            checkpoint
        )

        alpha = float(
            env.alpha
        )

        angles_deg = np.linspace(
            0.0,
            360.0,
            num_directions,
            endpoint=False,
        ).tolist()

        step_rows = []

        episode_results = []

        # ==================================================
        # Run 32 directions
        # ==================================================

        for (
            angle_index,
            angle_deg,
        ) in enumerate(
            angles_deg
        ):

            perturb_env.fixed_angle_rad = float(
                np.deg2rad(
                    angle_deg
                )
            )

            episode_seed = (
                eval_seed
                + angle_index * 1000
            )

            observation, reset_info = (
                env.reset(
                    seed=episode_seed
                )
            )

            observation = np.asarray(
                observation,
                dtype=np.float32,
            )

            final_info = {}

            episode_return = 0.0

            for step in range(
                max_steps
            ):

                phase = (
                    phase_from_observation(
                        observation
                    )
                )

                residual_action = (
                    policy.select_action(
                        observation
                    )
                )

                residual_action = np.asarray(
                    residual_action,
                    dtype=np.float32,
                )

                residual_action = np.clip(
                    residual_action,
                    -1.0,
                    1.0,
                )

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    residual_action
                )

                episode_return += float(
                    reward
                )

                residual_l2 = float(
                    np.linalg.norm(
                        residual_action
                    )
                )

                residual_abs_mean = float(
                    np.mean(
                        np.abs(
                            residual_action
                        )
                    )
                )

                saturation_rate = float(
                    np.mean(
                        np.abs(
                            residual_action
                        )
                        >= saturation_threshold
                    )
                )

                row = {
                    "angle_deg":
                        float(
                            angle_deg
                        ),

                    "seed":
                        int(
                            episode_seed
                        ),

                    "step":
                        int(
                            step
                        ),

                    "phase":
                        phase,

                    "residual_l2":
                        residual_l2,

                    "mean_abs_residual":
                        residual_abs_mean,

                    "effective_mean_abs_residual":
                        float(
                            alpha
                            * residual_abs_mean
                        ),

                    "saturation_rate":
                        saturation_rate,

                    "reward":
                        float(
                            reward
                        ),
                }

                for (
                    joint_index,
                    joint_name,
                ) in enumerate(
                    JOINT_NAMES
                ):

                    value = float(
                        residual_action[
                            joint_index
                        ]
                    )

                    row[
                        f"{joint_name}_residual"
                    ] = value

                    row[
                        f"{joint_name}_effective"
                    ] = (
                        alpha
                        * value
                    )

                step_rows.append(
                    row
                )

                final_info = info

                observation = np.asarray(
                    next_observation,
                    dtype=np.float32,
                )

                if (
                    terminated
                    or truncated
                ):
                    break

            episode_results.append(
                {
                    "angle_deg":
                        float(
                            angle_deg
                        ),

                    "seed":
                        int(
                            episode_seed
                        ),

                    "success":
                        bool(
                            final_info.get(
                                "success",
                                False,
                            )
                        ),

                    "episode_return":
                        float(
                            episode_return
                        ),

                    "terminal_reason":
                        final_info.get(
                            "terminal_reason"
                        ),
                }
            )

        # ==================================================
        # Phase summary
        # ==================================================

        phase_groups = defaultdict(
            list
        )

        for row in step_rows:
            phase_groups[
                row["phase"]
            ].append(
                row
            )

        phase_summary_rows = []

        for phase in PHASE_NAMES:

            rows = phase_groups.get(
                phase,
                [],
            )

            if not rows:
                continue

            phase_summary_rows.append(
                {
                    "phase":
                        phase,

                    "steps":
                        len(rows),

                    "mean_residual_l2":
                        mean(
                            [
                                row[
                                    "residual_l2"
                                ]
                                for row
                                in rows
                            ]
                        ),

                    "mean_abs_residual":
                        mean(
                            [
                                row[
                                    "mean_abs_residual"
                                ]
                                for row
                                in rows
                            ]
                        ),

                    "effective_mean_abs_residual":
                        mean(
                            [
                                row[
                                    "effective_mean_abs_residual"
                                ]
                                for row
                                in rows
                            ]
                        ),

                    "saturation_rate":
                        mean(
                            [
                                row[
                                    "saturation_rate"
                                ]
                                for row
                                in rows
                            ]
                        ),
                }
            )

        # ==================================================
        # Joint summary
        # ==================================================

        joint_summary_rows = []

        for joint_name in JOINT_NAMES:

            values = [
                float(
                    row[
                        f"{joint_name}_residual"
                    ]
                )
                for row
                in step_rows
            ]

            joint_summary_rows.append(
                {
                    "joint":
                        joint_name,

                    "steps":
                        len(values),

                    "mean_signed_residual":
                        mean(values),

                    "mean_abs_residual":
                        mean(
                            [
                                abs(value)
                                for value
                                in values
                            ]
                        ),

                    "effective_mean_abs_residual":
                        (
                            alpha
                            * mean(
                                [
                                    abs(value)
                                    for value
                                    in values
                                ]
                            )
                        ),

                    "saturation_rate":
                        mean(
                            [
                                float(
                                    abs(value)
                                    >= saturation_threshold
                                )
                                for value
                                in values
                            ]
                        ),
                }
            )

        # ==================================================
        # Phase x Joint summary
        # ==================================================

        phase_joint_rows = []

        for phase in PHASE_NAMES:

            rows = phase_groups.get(
                phase,
                [],
            )

            if not rows:
                continue

            for joint_name in JOINT_NAMES:

                values = [
                    float(
                        row[
                            f"{joint_name}_residual"
                        ]
                    )
                    for row
                    in rows
                ]

                phase_joint_rows.append(
                    {
                        "phase":
                            phase,

                        "joint":
                            joint_name,

                        "steps":
                            len(values),

                        "mean_signed_residual":
                            mean(
                                values
                            ),

                        "mean_abs_residual":
                            mean(
                                [
                                    abs(value)
                                    for value
                                    in values
                                ]
                            ),

                        "effective_mean_abs_residual":
                            (
                                alpha
                                * mean(
                                    [
                                        abs(value)
                                        for value
                                        in values
                                    ]
                                )
                            ),

                        "saturation_rate":
                            mean(
                                [
                                    float(
                                        abs(value)
                                        >= saturation_threshold
                                    )
                                    for value
                                    in values
                                ]
                            ),
                    }
                )

        # ==================================================
        # Overall summary
        # ==================================================

        success_count = sum(
            int(
                row["success"]
            )
            for row
            in episode_results
        )

        summary = {
            "checkpoint":
                str(
                    checkpoint
                ),

            "radius_mm":
                radius_mm,

            "directions":
                num_directions,

            "episodes":
                len(
                    episode_results
                ),

            "success_count":
                success_count,

            "success_rate":
                float(
                    success_count
                    / max(
                        len(
                            episode_results
                        ),
                        1,
                    )
                ),

            "alpha":
                alpha,

            "saturation_threshold":
                saturation_threshold,

            "overall_mean_residual_l2":
                mean(
                    [
                        row[
                            "residual_l2"
                        ]
                        for row
                        in step_rows
                    ]
                ),

            "overall_mean_abs_residual":
                mean(
                    [
                        row[
                            "mean_abs_residual"
                        ]
                        for row
                        in step_rows
                    ]
                ),

            "overall_saturation_rate":
                mean(
                    [
                        row[
                            "saturation_rate"
                        ]
                        for row
                        in step_rows
                    ]
                ),
        }

        # ==================================================
        # Save
        # ==================================================

        write_csv(
            args.output_dir
            / "steps.csv",
            step_rows,
        )

        write_csv(
            args.output_dir
            / "episodes.csv",
            episode_results,
        )

        write_csv(
            args.output_dir
            / "phase_summary.csv",
            phase_summary_rows,
        )

        write_csv(
            args.output_dir
            / "joint_summary.csv",
            joint_summary_rows,
        )

        write_csv(
            args.output_dir
            / "phase_joint_summary.csv",
            phase_joint_rows,
        )

        with (
            args.output_dir
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
        print("Day14 Residual Analysis")
        print("=" * 72)

        print(
            "success:",
            f"{success_count}/"
            f"{len(episode_results)}",
        )

        print(
            "overall L2:",
            summary[
                "overall_mean_residual_l2"
            ],
        )

        print(
            "overall |residual|:",
            summary[
                "overall_mean_abs_residual"
            ],
        )

        print(
            "overall saturation:",
            summary[
                "overall_saturation_rate"
            ],
        )

        print()

        print("Phase summary")

        for row in phase_summary_rows:

            print(
                f"{row['phase']:14s} "
                f"L2="
                f"{row['mean_residual_l2']:.3f} "
                f"|res|="
                f"{row['mean_abs_residual']:.3f} "
                f"sat="
                f"{row['saturation_rate']:.3f}"
            )

        print()

        print("Joint summary")

        for row in joint_summary_rows:

            print(
                f"{row['joint']:15s} "
                f"mean="
                f"{row['mean_signed_residual']:+.3f} "
                f"|res|="
                f"{row['mean_abs_residual']:.3f} "
                f"sat="
                f"{row['saturation_rate']:.3f}"
            )

        print()

        print(
            "output:",
            args.output_dir,
        )

        return 0

    finally:

        env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )