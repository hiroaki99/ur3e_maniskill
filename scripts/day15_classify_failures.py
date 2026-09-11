#!/usr/bin/env python3
"""
Day 15 - Failure classification.

Reference-only と Day13 best TD3 を
同一32方向で再評価し、失敗原因を分類する。

Failure labels
--------------
approach_failure
unilateral_contact
unstable_grasp
no_lift_after_grasp
insufficient_lift
drop_after_lift
final_hold_failure
knocked_away
collision
timeout
other_failure

重要:
この分類はログから定義した「診断ラベル」であり、
物理的因果関係を完全に証明するものではない。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
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


# ==========================================================
# Observation contract
#
# Day9:
# [ 0: 6] arm qpos
# [ 6:12] arm qvel
# [12:15] grasp center
# [15:18] cube position
# [18:24] reference qpos
# [24:30] reference error
# [30:33] cube_to_grasp
# [33:40] phase onehot
# [40:42] contact flags
# [42]    cube lift
# [43]    phase progress
# ==========================================================

PHASE_NAMES = [
    "to_pregrasp",
    "descend",
    "grasp",
    "stable_hold",
    "lift",
    "final_hold",
    "terminal",
]


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
            "reports/day15_failure_analysis"
        ),
    )

    return parser.parse_args()


# ==========================================================
# Utilities
# ==========================================================

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


def resolve_repo_path(
    path: Path,
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


def info_float(
    info,
    keys,
    default=0.0,
):

    for key in keys:

        if key in info:

            try:
                return float(
                    info[key]
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

    return float(
        default
    )


def observation_vector(
    observation,
):

    return np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1)


def cube_position(
    observation,
):

    observation = observation_vector(
        observation
    )

    return np.asarray(
        observation[15:18],
        dtype=np.float64,
    )


def cube_to_grasp_distance(
    observation,
):

    observation = observation_vector(
        observation
    )

    vector = observation[
        30:33
    ]

    return float(
        np.linalg.norm(
            vector
        )
    )


def phase_index(
    observation,
):

    observation = observation_vector(
        observation
    )

    onehot = observation[
        33:40
    ]

    return int(
        np.argmax(
            onehot
        )
    )


def phase_name(
    observation,
):

    index = phase_index(
        observation
    )

    if (
        0
        <= index
        < len(
            PHASE_NAMES
        )
    ):

        return PHASE_NAMES[
            index
        ]

    return "unknown"


# ==========================================================
# Environment
# ==========================================================

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
        float(
            radius_mm
        )
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

    return (
        env,
        perturb_env,
    )


def make_policy(
    *,
    env,
    config,
    device,
):

    day11 = config[
        "day11"
    ]

    policy = TD3(
        state_dim=int(
            env.observation_space.shape[
                0
            ]
        ),

        action_dim=int(
            env.action_space.shape[
                0
            ]
        ),

        max_action=float(
            env.action_space.high[
                0
            ]
        ),

        actor_hidden_dim=int(
            day11[
                "actor_hidden_dim"
            ]
        ),

        critic_hidden_dim=int(
            day11[
                "critic_hidden_dim"
            ]
        ),

        discount=float(
            day11[
                "discount"
            ]
        ),

        tau=float(
            day11[
                "tau"
            ]
        ),

        policy_noise=float(
            day11[
                "policy_noise"
            ]
        ),

        noise_clip=float(
            day11[
                "noise_clip"
            ]
        ),

        policy_freq=int(
            day11[
                "policy_freq"
            ]
        ),

        actor_lr=float(
            day11[
                "actor_lr"
            ]
        ),

        critic_lr=float(
            day11[
                "critic_lr"
            ]
        ),

        weight_decay=float(
            day11[
                "weight_decay"
            ]
        ),

        device=device,

        seed=int(
            config[
                "day13"
            ][
                "train_seed"
            ]
        ),
    )

    return policy


# ==========================================================
# Failure classifier
# ==========================================================

def classify_failure(
    metrics,
    day15,
):

    if metrics[
        "success"
    ]:
        return "success"

    # ------------------------------------------------------
    # 1. Safety / collision
    # ------------------------------------------------------

    if (
        metrics[
            "unsafe_collision_observable"
        ]
        and metrics[
            "unsafe_collision"
        ]
    ):
        return "collision"

    lift_threshold = float(
        day15[
            "lift_threshold_m"
        ]
    )

    lift_start_threshold = float(
        day15[
            "lift_start_threshold_m"
        ]
    )

    stable_grasp_steps = int(
        day15[
            "stable_grasp_steps"
        ]
    )

    drop_margin = float(
        day15[
            "drop_margin_m"
        ]
    )

    knock_xy_threshold = float(
        day15[
            "knock_xy_threshold_m"
        ]
    )

    # ------------------------------------------------------
    # 2. Reached success height but final success failed
    # ------------------------------------------------------

    if metrics[
        "max_cube_lift_m"
    ] >= lift_threshold:

        # 一度5 cmまで上げた後、
        # 1 cm以上下がった場合
        if (
            metrics[
                "final_cube_lift_m"
            ]
            <
            metrics[
                "max_cube_lift_m"
            ]
            - drop_margin
        ):

            return "drop_after_lift"

        if metrics[
            "lost_contact_after_lift"
        ]:

            return "drop_after_lift"

        return "final_hold_failure"

    # ------------------------------------------------------
    # 3. Cube knocked horizontally
    # ------------------------------------------------------

    if (
        metrics[
            "max_cube_xy_displacement_m"
        ]
        >= knock_xy_threshold
    ):

        return "knocked_away"

    # ------------------------------------------------------
    # 4. No contact
    # ------------------------------------------------------

    if (
        not metrics[
            "ever_left_contact"
        ]
        and
        not metrics[
            "ever_right_contact"
        ]
    ):

        return "approach_failure"

    # ------------------------------------------------------
    # 5. Only one-sided contact
    # ------------------------------------------------------

    if not metrics[
        "ever_bilateral_contact"
    ]:

        return "unilateral_contact"

    # ------------------------------------------------------
    # 6. Bilateral contact was too short
    # ------------------------------------------------------

    if (
        metrics[
            "terminal_reason"
        ]
        == "unstable_grasp"
    ):

        return "unstable_grasp"

    if (
        metrics[
            "max_consecutive_bilateral_steps"
        ]
        < stable_grasp_steps
    ):

        return "unstable_grasp"

    # ------------------------------------------------------
    # 7. Stable grasp but almost no lift
    # ------------------------------------------------------

    if (
        metrics[
            "max_cube_lift_m"
        ]
        < lift_start_threshold
    ):

        return "no_lift_after_grasp"

    # ------------------------------------------------------
    # 8. Lifted slightly but did not reach 5 cm
    # ------------------------------------------------------

    if (
        metrics[
            "max_cube_lift_m"
        ]
        < lift_threshold
    ):

        return "insufficient_lift"

    # ------------------------------------------------------
    # 9. Timeout
    # ------------------------------------------------------

    if metrics[
        "timeout"
    ]:

        return "timeout"

    return "other_failure"


# ==========================================================
# Single episode
# ==========================================================

def run_episode(
    *,
    env,
    perturb_env,
    policy,
    condition,
    angle_deg,
    episode_seed,
    max_steps,
    day15,
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
        env.action_space.shape[
            0
        ]
    )

    initial_cube_position = (
        cube_position(
            observation
        )
    )

    min_grasp_distance = (
        cube_to_grasp_distance(
            observation
        )
    )

    # ------------------------------------------------------
    # Contact state
    # ------------------------------------------------------

    ever_left = False
    ever_right = False
    ever_bilateral = False

    current_bilateral_run = 0
    max_bilateral_run = 0

    first_left_step = None
    first_right_step = None
    first_bilateral_step = None

    lost_contact_after_bilateral = False
    lost_contact_after_lift = False

    # ------------------------------------------------------
    # Lift / cube displacement
    # ------------------------------------------------------

    max_cube_lift = 0.0
    final_cube_lift = 0.0

    max_xy_displacement = 0.0

    first_lift_start_step = None
    first_lift_threshold_step = None

    reached_lift_threshold = False

    # ------------------------------------------------------
    # Residual statistics
    # ------------------------------------------------------

    residual_l2_sum = 0.0
    residual_abs_sum = 0.0
    residual_elements = 0
    residual_saturated_elements = 0

    saturation_threshold = float(
        day15[
            "residual_saturation_threshold"
        ]
    )

    # ------------------------------------------------------
    # Safety / forces
    # ------------------------------------------------------

    unsafe_collision = False
    unsafe_collision_observable = False

    max_left_force = 0.0
    max_right_force = 0.0

    force_observable = False

    # ------------------------------------------------------
    # General
    # ------------------------------------------------------

    episode_return = 0.0

    max_phase_index = (
        phase_index(
            observation
        )
    )

    final_info = {}

    terminated = False
    truncated = False

    executed_steps = 0

    lift_threshold = float(
        day15[
            "lift_threshold_m"
        ]
    )

    lift_start_threshold = float(
        day15[
            "lift_start_threshold_m"
        ]
    )

    # ======================================================
    # Episode loop
    # ======================================================

    for step in range(
        max_steps
    ):

        # --------------------------------------------------
        # Deterministic residual
        # --------------------------------------------------

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

            residual_action = np.asarray(
                residual_action,
                dtype=np.float32,
            )

            residual_action = np.clip(
                residual_action,
                -1.0,
                1.0,
            )

        # --------------------------------------------------
        # Residual statistics
        # --------------------------------------------------

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

        residual_elements += int(
            residual_action.size
        )

        residual_saturated_elements += int(
            np.count_nonzero(
                np.abs(
                    residual_action
                )
                >= saturation_threshold
            )
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
            residual_action
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
        # Phase
        # --------------------------------------------------

        current_phase_index = (
            phase_index(
                next_observation
            )
        )

        max_phase_index = max(
            max_phase_index,
            current_phase_index,
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

        if left_contact:

            if not ever_left:
                first_left_step = step

            ever_left = True

        if right_contact:

            if not ever_right:
                first_right_step = step

            ever_right = True

        if bilateral:

            if not ever_bilateral:
                first_bilateral_step = step

            ever_bilateral = True

            current_bilateral_run += 1

            max_bilateral_run = max(
                max_bilateral_run,
                current_bilateral_run,
            )

        else:

            if ever_bilateral:
                lost_contact_after_bilateral = True

            if reached_lift_threshold:
                lost_contact_after_lift = True

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
                current_cube_position[
                    :2
                ]
                - initial_cube_position[
                    :2
                ]
            )
        )

        max_xy_displacement = max(
            max_xy_displacement,
            xy_displacement,
        )

        cube_lift = info_float(
            info,
            [
                "cube_lift_m",
            ],
            default=(
                current_cube_position[
                    2
                ]
                - initial_cube_position[
                    2
                ]
            ),
        )

        final_cube_lift = (
            cube_lift
        )

        max_cube_lift = max(
            max_cube_lift,
            cube_lift,
        )

        if (
            first_lift_start_step
            is None
            and cube_lift
            >= lift_start_threshold
        ):

            first_lift_start_step = (
                step
            )

        if (
            not reached_lift_threshold
            and cube_lift
            >= lift_threshold
        ):

            reached_lift_threshold = True

            first_lift_threshold_step = (
                step
            )

        # --------------------------------------------------
        # TCP / grasp center distance
        # --------------------------------------------------

        min_grasp_distance = min(
            min_grasp_distance,
            cube_to_grasp_distance(
                next_observation
            ),
        )

        # --------------------------------------------------
        # Collision
        #
        # 現在のEnvが unsafe_collision を
        # infoへ出している場合のみ有効。
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

        # --------------------------------------------------
        # Contact force
        # --------------------------------------------------

        left_force_keys = [
            "left_contact_force_n",
            "left_force_n",
        ]

        right_force_keys = [
            "right_contact_force_n",
            "right_force_n",
        ]

        if any(
            key in info
            for key in left_force_keys
        ):

            force_observable = True

        if any(
            key in info
            for key in right_force_keys
        ):

            force_observable = True

        left_force = info_float(
            info,
            left_force_keys,
            default=0.0,
        )

        right_force = info_float(
            info,
            right_force_keys,
            default=0.0,
        )

        max_left_force = max(
            max_left_force,
            left_force,
        )

        max_right_force = max(
            max_right_force,
            right_force,
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
    # Episode result
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

    mean_residual_l2 = (
        residual_l2_sum
        / max(
            executed_steps,
            1,
        )
    )

    mean_abs_residual = (
        residual_abs_sum
        / max(
            residual_elements,
            1,
        )
    )

    saturation_rate = (
        residual_saturated_elements
        / max(
            residual_elements,
            1,
        )
    )

    metrics = {
        "condition":
            condition,

        "angle_deg":
            float(
                angle_deg
            ),

        "seed":
            int(
                episode_seed
            ),

        "offset_x_mm":
            float(
                reset_info.get(
                    "cube_offset_m",
                    [0.0, 0.0, 0.0],
                )[0]
                * 1000.0
            ),

        "offset_y_mm":
            float(
                reset_info.get(
                    "cube_offset_m",
                    [0.0, 0.0, 0.0],
                )[1]
                * 1000.0
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

        "max_phase_index":
            int(
                max_phase_index
            ),

        "max_phase_name":
            (
                PHASE_NAMES[
                    max_phase_index
                ]
                if 0
                <= max_phase_index
                < len(
                    PHASE_NAMES
                )
                else "unknown"
            ),

        # Contact
        "ever_left_contact":
            bool(
                ever_left
            ),

        "ever_right_contact":
            bool(
                ever_right
            ),

        "ever_bilateral_contact":
            bool(
                ever_bilateral
            ),

        "first_left_step":
            first_left_step,

        "first_right_step":
            first_right_step,

        "first_bilateral_step":
            first_bilateral_step,

        "max_consecutive_bilateral_steps":
            int(
                max_bilateral_run
            ),

        "final_consecutive_bilateral_steps":
            int(
                current_bilateral_run
            ),

        "lost_contact_after_bilateral":
            bool(
                lost_contact_after_bilateral
            ),

        "lost_contact_after_lift":
            bool(
                lost_contact_after_lift
            ),

        # Distance / cube
        "min_cube_to_grasp_m":
            float(
                min_grasp_distance
            ),

        "max_cube_lift_m":
            float(
                max_cube_lift
            ),

        "final_cube_lift_m":
            float(
                final_cube_lift
            ),

        "first_lift_start_step":
            first_lift_start_step,

        "first_lift_threshold_step":
            first_lift_threshold_step,

        "reached_lift_threshold":
            bool(
                reached_lift_threshold
            ),

        "max_cube_xy_displacement_m":
            float(
                max_xy_displacement
            ),

        # Safety
        "unsafe_collision_observable":
            bool(
                unsafe_collision_observable
            ),

        "unsafe_collision":
            bool(
                unsafe_collision
            ),

        "force_observable":
            bool(
                force_observable
            ),

        "max_left_contact_force_n":
            float(
                max_left_force
            ),

        "max_right_contact_force_n":
            float(
                max_right_force
            ),

        # Residual
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
                env.alpha
                * mean_abs_residual
            ),

        "residual_saturation_rate":
            float(
                saturation_rate
            ),

        # Terminal
        "terminal_reason":
            final_info.get(
                "terminal_reason"
            ),

        "timeout":
            timeout,
    }

    metrics[
        "failure_cause"
    ] = classify_failure(
        metrics,
        day15,
    )

    return metrics


# ==========================================================
# Count helpers
# ==========================================================

def failure_counter(
    rows,
):

    return Counter(
        row[
            "failure_cause"
        ]
        for row in rows
        if not row[
            "success"
        ]
    )


def top_causes(
    counter,
):

    if not counter:
        return []

    max_count = max(
        counter.values()
    )

    return sorted(
        [
            cause
            for (
                cause,
                count,
            )
            in counter.items()
            if count
            == max_count
        ]
    )


def make_failure_count_rows(
    condition,
    rows,
):

    failures = [
        row
        for row in rows
        if not row[
            "success"
        ]
    ]

    counter = failure_counter(
        rows
    )

    output = []

    for (
        cause,
        count,
    ) in counter.most_common():

        output.append(
            {
                "condition":
                    condition,

                "failure_cause":
                    cause,

                "count":
                    int(
                        count
                    ),

                "rate_among_failures":
                    float(
                        count
                        / max(
                            len(
                                failures
                            ),
                            1,
                        )
                    ),

                "rate_all_episodes":
                    float(
                        count
                        / max(
                            len(
                                rows
                            ),
                            1,
                        )
                    ),
            }
        )

    return output


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

    checkpoint = (
        args.checkpoint
        if args.checkpoint
        is not None
        else Path(
            day15[
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

    radius_mm = float(
        day15[
            "radius_mm"
        ]
    )

    num_directions = int(
        args.num_directions
        if args.num_directions
        is not None
        else day15[
            "num_directions"
        ]
    )

    episodes_per_direction = int(
        args.episodes_per_direction
        if args.episodes_per_direction
        is not None
        else day15[
            "episodes_per_direction"
        ]
    )

    eval_seed = int(
        day15[
            "eval_seed"
        ]
    )

    max_steps = int(
        day15[
            "max_steps_per_episode"
        ]
    )

    device = resolve_device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # Environment
    # ======================================================

    env, perturb_env = make_env(
        config=config,
        trajectory=trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
    )

    try:

        # Initialize dimensions
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

        reference_rows = []
        residual_rows = []

        # ==================================================
        # Reference-only
        # ==================================================

        print("=" * 72)
        print("Reference-only diagnostics")
        print("=" * 72)

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
                    policy=None,
                    condition="reference",
                    angle_deg=angle_deg,
                    episode_seed=(
                        episode_seed
                    ),
                    max_steps=max_steps,
                    day15=day15,
                )

                row[
                    "repeat"
                ] = repeat

                reference_rows.append(
                    row
                )

                print(
                    f"angle="
                    f"{angle_deg:6.2f} "
                    f"success="
                    f"{row['success']} "
                    f"cause="
                    f"{row['failure_cause']}"
                )

        # ==================================================
        # Residual TD3
        # ==================================================

        print()
        print("=" * 72)
        print("Residual TD3 diagnostics")
        print("=" * 72)

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
                    condition="residual_td3",
                    angle_deg=angle_deg,
                    episode_seed=(
                        episode_seed
                    ),
                    max_steps=max_steps,
                    day15=day15,
                )

                row[
                    "repeat"
                ] = repeat

                residual_rows.append(
                    row
                )

                print(
                    f"angle="
                    f"{angle_deg:6.2f} "
                    f"success="
                    f"{row['success']} "
                    f"cause="
                    f"{row['failure_cause']} "
                    f"L2="
                    f"{row['mean_residual_l2']:.3f} "
                    f"sat="
                    f"{row['residual_saturation_rate']:.3f}"
                )

        # ==================================================
        # Paired comparison
        # ==================================================

        reference_map = {
            (
                float(
                    row[
                        "angle_deg"
                    ]
                ),
                int(
                    row[
                        "repeat"
                    ]
                ),
            ): row
            for row
            in reference_rows
        }

        residual_map = {
            (
                float(
                    row[
                        "angle_deg"
                    ]
                ),
                int(
                    row[
                        "repeat"
                    ]
                ),
            ): row
            for row
            in residual_rows
        }

        paired_rows = []

        rescued = 0
        broken = 0
        kept_success = 0
        kept_failure = 0

        for key in sorted(
            reference_map.keys()
        ):

            reference = (
                reference_map[
                    key
                ]
            )

            residual = (
                residual_map[
                    key
                ]
            )

            reference_success = bool(
                reference[
                    "success"
                ]
            )

            residual_success = bool(
                residual[
                    "success"
                ]
            )

            if (
                not reference_success
                and residual_success
            ):

                category = "rescued"
                rescued += 1

            elif (
                reference_success
                and not residual_success
            ):

                category = "broken"
                broken += 1

            elif (
                reference_success
                and residual_success
            ):

                category = (
                    "kept_success"
                )

                kept_success += 1

            else:

                category = (
                    "kept_failure"
                )

                kept_failure += 1

            paired_rows.append(
                {
                    "angle_deg":
                        reference[
                            "angle_deg"
                        ],

                    "repeat":
                        reference[
                            "repeat"
                        ],

                    "category":
                        category,

                    "reference_success":
                        reference_success,

                    "residual_success":
                        residual_success,

                    "reference_failure_cause":
                        reference[
                            "failure_cause"
                        ],

                    "residual_failure_cause":
                        residual[
                            "failure_cause"
                        ],

                    "reference_max_lift_m":
                        reference[
                            "max_cube_lift_m"
                        ],

                    "residual_max_lift_m":
                        residual[
                            "max_cube_lift_m"
                        ],

                    "reference_bilateral_steps":
                        reference[
                            "max_consecutive_bilateral_steps"
                        ],

                    "residual_bilateral_steps":
                        residual[
                            "max_consecutive_bilateral_steps"
                        ],

                    "reference_xy_displacement_m":
                        reference[
                            "max_cube_xy_displacement_m"
                        ],

                    "residual_xy_displacement_m":
                        residual[
                            "max_cube_xy_displacement_m"
                        ],

                    "residual_l2":
                        residual[
                            "mean_residual_l2"
                        ],

                    "residual_saturation_rate":
                        residual[
                            "residual_saturation_rate"
                        ],

                    "reference_return":
                        reference[
                            "episode_return"
                        ],

                    "residual_return":
                        residual[
                            "episode_return"
                        ],
                }
            )

        # ==================================================
        # Failure counts
        # ==================================================

        failure_count_rows = []

        failure_count_rows.extend(
            make_failure_count_rows(
                "reference",
                reference_rows,
            )
        )

        failure_count_rows.extend(
            make_failure_count_rows(
                "residual_td3",
                residual_rows,
            )
        )

        reference_counter = (
            failure_counter(
                reference_rows
            )
        )

        residual_counter = (
            failure_counter(
                residual_rows
            )
        )

        broken_rows = [
            residual_map[
                (
                    float(
                        row[
                            "angle_deg"
                        ]
                    ),
                    int(
                        row[
                            "repeat"
                        ]
                    ),
                )
            ]
            for row
            in paired_rows
            if row[
                "category"
            ]
            == "broken"
        ]

        broken_counter = (
            failure_counter(
                broken_rows
            )
        )

        # ==================================================
        # Observability
        # ==================================================

        collision_observable = any(
            row[
                "unsafe_collision_observable"
            ]
            for row
            in (
                reference_rows
                + residual_rows
            )
        )

        force_observable = any(
            row[
                "force_observable"
            ]
            for row
            in (
                reference_rows
                + residual_rows
            )
        )

        # ==================================================
        # Summary
        # ==================================================

        reference_success_count = sum(
            int(
                row[
                    "success"
                ]
            )
            for row
            in reference_rows
        )

        residual_success_count = sum(
            int(
                row[
                    "success"
                ]
            )
            for row
            in residual_rows
        )

        reference_success_rate = (
            reference_success_count
            / len(
                reference_rows
            )
        )

        residual_success_rate = (
            residual_success_count
            / len(
                residual_rows
            )
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
                len(
                    reference_rows
                ),

            "reference": {
                "success_count":
                    reference_success_count,

                "success_rate":
                    reference_success_rate,

                "failure_counts":
                    dict(
                        reference_counter
                    ),

                "most_frequent_failure_causes":
                    top_causes(
                        reference_counter
                    ),
            },

            "residual_td3": {
                "success_count":
                    residual_success_count,

                "success_rate":
                    residual_success_rate,

                "failure_counts":
                    dict(
                        residual_counter
                    ),

                "most_frequent_failure_causes":
                    top_causes(
                        residual_counter
                    ),
            },

            "paired": {
                "rescued":
                    rescued,

                "broken":
                    broken,

                "kept_success":
                    kept_success,

                "kept_failure":
                    kept_failure,

                "net_gain":
                    rescued
                    - broken,

                "broken_failure_counts":
                    dict(
                        broken_counter
                    ),

                "most_frequent_broken_failure_causes":
                    top_causes(
                        broken_counter
                    ),
            },

            "instrumentation": {
                "unsafe_collision_observable":
                    collision_observable,

                "contact_force_observable":
                    force_observable,
            },

            "day15_decision": {
                "primary_residual_failure_causes":
                    top_causes(
                        residual_counter
                    ),

                "primary_broken_failure_causes":
                    top_causes(
                        broken_counter
                    ),
            },

            "classification_note": (
                "Failure labels are operational diagnostic "
                "definitions derived from logged signals; "
                "they are not proof of physical causality."
            ),
        }

        # ==================================================
        # Save
        # ==================================================

        write_csv(
            args.output_dir
            / "episode_diagnostics.csv",
            (
                reference_rows
                + residual_rows
            ),
        )

        write_csv(
            args.output_dir
            / "failure_counts.csv",
            failure_count_rows,
        )

        write_csv(
            args.output_dir
            / "paired_results.csv",
            paired_rows,
        )

        summary_path = (
            args.output_dir
            / "summary.json"
        )

        with summary_path.open(
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
        # Console summary
        # ==================================================

        print()
        print("=" * 72)
        print("Day 15 Failure Classification")
        print("=" * 72)

        print(
            "Reference success:",
            f"{reference_success_count}/"
            f"{len(reference_rows)}",
            f"= "
            f"{reference_success_rate:.3f}",
        )

        print(
            "Residual success:",
            f"{residual_success_count}/"
            f"{len(residual_rows)}",
            f"= "
            f"{residual_success_rate:.3f}",
        )

        print()

        print(
            "Reference failures:",
            dict(
                reference_counter
            ),
        )

        print(
            "Residual failures:",
            dict(
                residual_counter
            ),
        )

        print()

        print(
            "Rescued:",
            rescued,
        )

        print(
            "Broken:",
            broken,
        )

        print(
            "Net gain:",
            rescued
            - broken,
        )

        print(
            "Broken causes:",
            dict(
                broken_counter
            ),
        )

        print()

        print(
            "Primary residual failure cause(s):",
            top_causes(
                residual_counter
            ),
        )

        print(
            "Primary broken failure cause(s):",
            top_causes(
                broken_counter
            ),
        )

        print()

        print(
            "Collision observable:",
            collision_observable,
        )

        print(
            "Contact force observable:",
            force_observable,
        )

        print()

        print(
            "summary:",
            summary_path,
        )

        return 0

    finally:

        env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )