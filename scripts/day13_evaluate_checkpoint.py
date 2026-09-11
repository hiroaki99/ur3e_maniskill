#!/usr/bin/env python3
"""
Evaluate Day13 checkpoint against Reference-only.

Same:
- radius
- directions
- seeds
- episode count

Reference:
    residual = 0

Residual RL:
    deterministic TD3 Actor
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


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
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
            "day13_best_checkpoint_eval"
        ),
    )

    return parser.parse_args()


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


def main():

    args = parse_args()

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    day9 = config.get(
        "day9",
        {},
    )

    day11 = config[
        "day11"
    ]

    day12 = config[
        "day12"
    ]

    day13 = config[
        "day13"
    ]

    radius_mm = float(
        day13.get(
            "train_radius_mm",
            day12[
                "selected_radius_mm"
            ],
        )
    )

    episodes_per_direction = int(
        args.episodes_per_direction
        if args.episodes_per_direction
        is not None
        else day13[
            "final_eval_episodes_per_direction"
        ]
    )

    eval_seed = int(
        day13[
            "eval_seed"
        ]
    )

    num_directions = int(
        day13[
            "eval_num_directions"
        ]
    )

    max_steps = int(
        day13[
            "max_steps_per_episode"
        ]
    )

    device = resolve_device(
        args.device
    )

    import envs.ur3e_pick_lift  # noqa

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
        sim_backend=args.sim_backend,
        max_episode_steps=int(
            day9.get(
                "base_env_max_episode_steps",
                1000,
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
            seed=eval_seed,
        )
    )

    env = ResidualPickLiftEnv(
        env=perturb_env,
        trajectory_path=(
            args.trajectory
        ),
        config_path=CONFIG_PATH,
    )

    try:

        observation, _ = env.reset(
            seed=eval_seed
        )

        state_dim = int(
            env.observation_space.shape[
                0
            ]
        )

        action_dim = int(
            env.action_space.shape[
                0
            ]
        )

        policy = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,

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
                day13[
                    "train_seed"
                ]
            ),
        )

        policy.load(
            args.checkpoint
        )

        angles_deg = (
            np.linspace(
                0.0,
                360.0,
                num_directions,
                endpoint=False,
            )
            .tolist()
        )

        common_args = dict(
            env=env,
            perturbation_wrapper=(
                perturb_env
            ),
            angles_deg=angles_deg,
            episodes_per_direction=(
                episodes_per_direction
            ),
            seed=eval_seed,
            max_steps=max_steps,
            lift_threshold_m=float(
                day13[
                    "lift_threshold_m"
                ]
            ),
            saturation_threshold=float(
                day13[
                    "residual_saturation_threshold"
                ]
            ),
        )

        # ==================================================
        # Reference only
        # ==================================================

        reference_summary, reference_rows = (
            evaluate_residual_policy(
                policy=None,
                **common_args,
            )
        )

        # ==================================================
        # Residual RL
        # ==================================================

        residual_summary, residual_rows = (
            evaluate_residual_policy(
                policy=policy,
                **common_args,
            )
        )

        for row in reference_rows:

            row[
                "condition"
            ] = "reference"

        for row in residual_rows:

            row[
                "condition"
            ] = "residual_td3"

        improvement = (
            residual_summary[
                "success_rate"
            ]
            - reference_summary[
                "success_rate"
            ]
        )

        result = {
            "checkpoint":
                str(
                    args.checkpoint
                ),

            "radius_mm":
                radius_mm,

            "episodes_per_direction":
                episodes_per_direction,

            "total_episodes_per_condition":
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
                    improvement
                    * 100.0
                ),
        }

        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_csv(
            args.output_dir
            / "episodes.csv",
            reference_rows
            + residual_rows,
        )

        with (
            args.output_dir
            / "summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print("=" * 72)
        print("Day13 Checkpoint Evaluation")
        print("=" * 72)

        print(
            "Reference success:",
            reference_summary[
                "success_rate"
            ],
        )

        print(
            "Residual success:",
            residual_summary[
                "success_rate"
            ],
        )

        print(
            "Improvement [pt]:",
            improvement
            * 100.0,
        )

        print(
            "Reference grasp:",
            reference_summary[
                "grasp_rate"
            ],
        )

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
            "Saturation rate:",
            residual_summary[
                "residual_saturation_rate"
            ],
        )

        return 0

    finally:

        env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )