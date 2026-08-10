#!/usr/bin/env python3
"""
Day9 ResidualPickLiftEnv integration test.

最重要検証:
    residual_action = 0

のとき、

    executed_action == reference_action

となり、Day6/7のReference Pick-and-Liftを
再現できることを確認する。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT),
)


from rrl import (
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


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
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
        "--max-steps",
        type=int,
        default=600,
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
        "--output",
        type=Path,
        default=Path(
            "reports/"
            "day9_residual_env_zero_test.json"
        ),
    )

    return parser.parse_args()


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

    tolerance = float(
        day9.get(
            "zero_residual_tolerance",
            1.0e-7,
        )
    )

    base_max_steps = int(
        day9.get(
            "base_env_max_episode_steps",
            1000,
        )
    )

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
        sim_backend=args.sim_backend,
        render_mode=(
            "human"
            if args.render
            else None
        ),
        max_episode_steps=(
            base_max_steps
        ),
    )

    env = ResidualPickLiftEnv(
        env=base_env,
        trajectory_path=(
            args.trajectory
        ),
        config_path=CONFIG_PATH,
    )

    try:

        (
            observation,
            info,
        ) = env.reset(
            seed=int(
                config[
                    "project"
                ][
                    "seed"
                ]
            )
        )

        print("=" * 72)
        print("Day 9 Residual Environment Test")
        print("=" * 72)

        print(
            "action space:",
            env.action_space,
        )

        print(
            "observation space:",
            env.observation_space,
        )

        print(
            "observation shape:",
            observation.shape,
        )

        print(
            "phase:",
            info[
                "phase_name"
            ],
        )

        print()
        print("Observation layout")

        for (
            name,
            span,
        ) in env.observation_layout.items():

            print(
                f"{name:20s}: "
                f"{span}"
            )

        errors = []

        if (
            observation.shape
            != (
                env.OBSERVATION_DIM,
            )
        ):

            errors.append(
                "initial observation shape mismatch"
            )

        if not np.all(
            np.isfinite(
                observation
            )
        ):

            errors.append(
                "initial observation has NaN/inf"
            )

        if not env.observation_space.contains(
            observation
        ):

            errors.append(
                "initial observation is outside "
                "observation_space"
            )

        if (
            env.action_space.shape
            != (6,)
        ):

            errors.append(
                "Residual action space is not (6,)"
            )

        # --------------------------------------------------
        # Zero residual episode
        # --------------------------------------------------

        zero_residual = np.zeros(
            6,
            dtype=np.float32,
        )

        max_zero_error = 0.0

        max_executed_action = 0.0

        nan_observation_count = 0

        action_bound_violation_count = 0

        phase_steps = {
            name: 0
            for name
            in env.PHASE_NAMES
        }

        terminated = False
        truncated = False

        final_info = {}

        executed_steps = 0

        print()
        print("=" * 72)
        print("Zero Residual Episode")
        print("=" * 72)

        for step in range(
            args.max_steps
        ):

            (
                observation,
                reward,
                terminated,
                truncated,
                step_info,
            ) = env.step(
                zero_residual
            )

            executed_steps += 1

            final_info = (
                step_info
            )

            phase_name = (
                step_info[
                    "phase_name_before"
                ]
            )

            phase_steps[
                phase_name
            ] += 1

            reference_action = np.asarray(
                step_info[
                    "reference_action"
                ],
                dtype=np.float64,
            )

            executed_action = np.asarray(
                step_info[
                    "executed_arm_action"
                ],
                dtype=np.float64,
            )

            zero_error = float(
                np.max(
                    np.abs(
                        executed_action
                        - reference_action
                    )
                )
            )

            max_zero_error = max(
                max_zero_error,
                zero_error,
            )

            max_executed_action = max(
                max_executed_action,
                float(
                    np.max(
                        np.abs(
                            executed_action
                        )
                    )
                ),
            )

            if (
                np.any(
                    executed_action
                    < -1.0
                )
                or np.any(
                    executed_action
                    > 1.0
                )
            ):

                action_bound_violation_count += 1

            if not np.all(
                np.isfinite(
                    observation
                )
            ):

                nan_observation_count += 1

            if (
                step % 20 == 0
                or step_info[
                    "phase_name"
                ]
                != phase_name
                or terminated
                or truncated
            ):

                print(
                    f"step={step:3d} "
                    f"phase="
                    f"{phase_name:14s} "
                    f"-> "
                    f"{step_info['phase_name']:14s} "
                    f"lift="
                    f"{step_info['cube_lift_m']:.5f} "
                    f"L="
                    f"{step_info['left_contact']} "
                    f"R="
                    f"{step_info['right_contact']} "
                    f"zero_err="
                    f"{zero_error:.3e}"
                )

            if args.render:

                env.render()

                time.sleep(
                    args.sleep
                )

            if (
                terminated
                or truncated
            ):
                break

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

        if not (
            terminated
            or truncated
        ):

            errors.append(
                "episode did not finish "
                f"within {args.max_steps} steps"
            )

        if (
            max_zero_error
            > tolerance
        ):

            errors.append(
                "zero residual does not reproduce "
                "reference action: "
                f"{max_zero_error}"
            )

        if (
            action_bound_violation_count
            != 0
        ):

            errors.append(
                "executed action exceeded [-1,1]"
            )

        if (
            nan_observation_count
            != 0
        ):

            errors.append(
                "observation contained NaN/inf"
            )

        success = bool(
            final_info.get(
                "success",
                False,
            )
        )

        cube_lift = float(
            final_info.get(
                "cube_lift_m",
                0.0,
            )
        )

        final_both_contact_steps = int(
            final_info.get(
                "final_both_contact_steps",
                0,
            )
        )

        success_lift_height = float(
            config[
                "day6"
            ][
                "success_lift_height_m"
            ]
        )

        success_hold_steps = int(
            config[
                "day6"
            ][
                "success_hold_steps"
            ]
        )

        if not success:

            errors.append(
                "zero residual episode failed"
            )

        if (
            cube_lift
            < success_lift_height
        ):

            errors.append(
                "cube lift below threshold: "
                f"{cube_lift}"
            )

        if (
            final_both_contact_steps
            < success_hold_steps
        ):

            errors.append(
                "final bilateral contact "
                "hold is insufficient"
            )

        passed = (
            len(errors) == 0
        )

        report = {
            "trajectory":
                str(
                    args.trajectory
                ),

            "action_dim":
                int(
                    env.action_space.shape[
                        0
                    ]
                ),

            "observation_dim":
                int(
                    env.OBSERVATION_DIM
                ),

            "observation_layout":
                env.observation_layout,

            "alpha":
                float(
                    env.alpha
                ),

            "migration_contract":
                env.migration_contract(),

            "zero_residual_test": {
                "executed_steps":
                    executed_steps,

                "max_action_difference":
                    max_zero_error,

                "tolerance":
                    tolerance,

                "max_abs_executed_action":
                    max_executed_action,

                "action_bound_violation_count":
                    action_bound_violation_count,

                "nan_observation_count":
                    nan_observation_count,

                "phase_steps":
                    phase_steps,
            },

            "task_result": {
                "success":
                    success,

                "terminal_reason":
                    final_info.get(
                        "terminal_reason"
                    ),

                "cube_lift_m":
                    cube_lift,

                "required_lift_m":
                    success_lift_height,

                "final_both_contact_steps":
                    final_both_contact_steps,

                "required_hold_steps":
                    success_hold_steps,
            },

            "errors":
                errors,

            "passed":
                passed,
        }

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print("=" * 72)
        print("Day 9 Result")
        print("=" * 72)

        print("executed steps:", executed_steps)

        print("observation dim:", env.OBSERVATION_DIM)

        print("residual action dim:", env.action_space.shape[0])

        print("max zero residual error:", max_zero_error)

        print("max |executed action|:", max_executed_action)

        print(
            "action bound violations:",
            action_bound_violation_count,
        )

        print(
            "NaN observations:",
            nan_observation_count,
        )

        print(
            "cube lift:",
            cube_lift,
            "m",
        )

        print(
            "final both contact:",
            final_both_contact_steps,
        )

        print(
            "success:",
            success,
        )

        print(
            "terminal reason:",
            final_info.get(
                "terminal_reason"
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
            "report:",
            args.output,
        )

        return (
            0
            if passed
            else 1
        )

    finally:

        env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )