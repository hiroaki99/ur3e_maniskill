#!/usr/bin/env python3
"""
Day10 Reward Test.

テスト内容
----------
A. Synthetic unit test
   - 接近すると報酬増加
   - 両側接触で報酬増加
   - Cube上昇で報酬増加
   - Successで報酬増加
   - Residualを大きくすると報酬低下
   - 過大接触力で報酬低下
   - Unsafe collisionで報酬低下

B. Zero-residual integration test
   Day9と同じReference-only動作を実行し、
   - 成功する
   - rewardがfinite
   - reward termsの和がtotalと一致
   - 各項を集計可能
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    ResidualPickLiftReward,
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
        "--output",
        type=Path,
        default=Path(
            "reports/"
            "day10_reward_test.json"
        ),
    )

    return parser.parse_args()


# ==========================================================
# Synthetic observation
# ==========================================================

def make_observation(
    layout,
    *,
    cube_to_grasp,
    cube_lift,
):

    obs = np.zeros(
        44,
        dtype=np.float32,
    )

    start, stop = (
        layout[
            "cube_to_grasp"
        ]
    )

    obs[
        start:stop
    ] = np.asarray(
        cube_to_grasp,
        dtype=np.float32,
    )

    start, stop = (
        layout[
            "cube_lift"
        ]
    )

    obs[
        start:stop
    ] = cube_lift

    return obs


def reward_case(
    reward_model,
    layout,
    *,
    previous_distance,
    current_distance,
    previous_lift=0.0,
    current_lift=0.0,
    phase="to_pregrasp",
    left_contact=False,
    right_contact=False,
    left_force=0.0,
    right_force=0.0,
    residual=None,
    terminated=False,
    success=False,
    unsafe_collision=False,
):

    if residual is None:

        residual = np.zeros(
            6,
            dtype=np.float32,
        )

    previous_obs = make_observation(
        layout,
        cube_to_grasp=[
            0.0,
            0.0,
            previous_distance,
        ],
        cube_lift=previous_lift,
    )

    current_obs = make_observation(
        layout,
        cube_to_grasp=[
            0.0,
            0.0,
            current_distance,
        ],
        cube_lift=current_lift,
    )

    info = {
        "phase_name_before":
            phase,

        "left_contact":
            left_contact,

        "right_contact":
            right_contact,

        "left_contact_force":
            left_force,

        "right_contact_force":
            right_force,

        "unsafe_collision":
            unsafe_collision,

        "success":
            success,
    }

    return reward_model.compute(
        previous_observation=(
            previous_obs
        ),
        observation=(
            current_obs
        ),
        residual_action=(
            residual
        ),
        info=info,
        terminated=terminated,
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

    import envs.ur3e_pick_lift  # noqa

    day9 = config.get(
        "day9",
        {},
    )

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

    env = ResidualPickLiftEnv(
        env=base_env,
        trajectory_path=(
            args.trajectory
        ),
        config_path=CONFIG_PATH,
    )

    errors = []

    try:

        # ==================================================
        # Synthetic tests
        # ==================================================

        reward_model = (
            env.reward_model
        )

        layout = (
            env.observation_layout
        )

        print("=" * 72)
        print("Day 10 Reward Unit Tests")
        print("=" * 72)

        synthetic = {}

        # --------------------------------------------------
        # Approach
        # --------------------------------------------------

        approach_good, terms = reward_case(
            reward_model,
            layout,
            previous_distance=0.10,
            current_distance=0.09,
            phase="to_pregrasp",
        )

        approach_bad, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.09,
            current_distance=0.10,
            phase="to_pregrasp",
        )

        synthetic[
            "approach_good"
        ] = approach_good

        synthetic[
            "approach_bad"
        ] = approach_bad

        if not (
            approach_good
            > approach_bad
        ):

            errors.append(
                "approach reward direction failed"
            )

        # --------------------------------------------------
        # Grasp
        # --------------------------------------------------

        grasp_yes, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.01,
            current_distance=0.01,
            phase="grasp",
            left_contact=True,
            right_contact=True,
            left_force=10.0,
            right_force=10.0,
        )

        grasp_no, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.01,
            current_distance=0.01,
            phase="grasp",
        )

        synthetic[
            "grasp_yes"
        ] = grasp_yes

        synthetic[
            "grasp_no"
        ] = grasp_no

        if not (
            grasp_yes
            > grasp_no
        ):

            errors.append(
                "grasp reward failed"
            )

        # --------------------------------------------------
        # Lift
        # --------------------------------------------------

        lift_up, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.01,
            current_distance=0.01,
            previous_lift=0.020,
            current_lift=0.025,
            phase="lift",
            left_contact=True,
            right_contact=True,
        )

        lift_static, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.01,
            current_distance=0.01,
            previous_lift=0.020,
            current_lift=0.020,
            phase="lift",
            left_contact=True,
            right_contact=True,
        )

        synthetic[
            "lift_up"
        ] = lift_up

        synthetic[
            "lift_static"
        ] = lift_static

        if not (
            lift_up
            > lift_static
        ):

            errors.append(
                "lift reward failed"
            )

        # --------------------------------------------------
        # Residual
        # --------------------------------------------------

        residual_zero, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.05,
            current_distance=0.05,
            residual=np.zeros(
                6,
                dtype=np.float32,
            ),
        )

        residual_large, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.05,
            current_distance=0.05,
            residual=np.ones(
                6,
                dtype=np.float32,
            ),
        )

        synthetic[
            "residual_zero"
        ] = residual_zero

        synthetic[
            "residual_large"
        ] = residual_large

        if not (
            residual_zero
            > residual_large
        ):

            errors.append(
                "residual penalty failed"
            )

        # --------------------------------------------------
        # Excessive force
        # --------------------------------------------------

        safe_force, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.01,
            current_distance=0.01,
            phase="grasp",
            left_contact=True,
            right_contact=True,
            left_force=10.0,
            right_force=10.0,
        )

        excessive_force, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.01,
            current_distance=0.01,
            phase="grasp",
            left_contact=True,
            right_contact=True,
            left_force=60.0,
            right_force=60.0,
        )

        synthetic[
            "safe_force"
        ] = safe_force

        synthetic[
            "excessive_force"
        ] = excessive_force

        if not (
            safe_force
            > excessive_force
        ):

            errors.append(
                "force penalty failed"
            )

        # --------------------------------------------------
        # Collision
        # --------------------------------------------------

        no_collision, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.05,
            current_distance=0.05,
        )

        collision, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.05,
            current_distance=0.05,
            unsafe_collision=True,
        )

        synthetic[
            "no_collision"
        ] = no_collision

        synthetic[
            "collision"
        ] = collision

        if not (
            no_collision
            > collision
        ):

            errors.append(
                "collision penalty failed"
            )

        # --------------------------------------------------
        # Terminal
        # --------------------------------------------------

        terminal_success, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.01,
            current_distance=0.01,
            previous_lift=0.05,
            current_lift=0.05,
            phase="final_hold",
            left_contact=True,
            right_contact=True,
            terminated=True,
            success=True,
        )

        terminal_failure, _ = reward_case(
            reward_model,
            layout,
            previous_distance=0.01,
            current_distance=0.01,
            phase="final_hold",
            terminated=True,
            success=False,
        )

        synthetic[
            "terminal_success"
        ] = terminal_success

        synthetic[
            "terminal_failure"
        ] = terminal_failure

        if not (
            terminal_success
            > terminal_failure
        ):

            errors.append(
                "terminal reward failed"
            )

        for (
            name,
            value,
        ) in synthetic.items():

            print(
                f"{name:24s}: "
                f"{value:+.6f}"
            )

        # ==================================================
        # Integration test
        # ==================================================

        print()
        print("=" * 72)
        print("Zero Residual Reward Episode")
        print("=" * 72)

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

        zero_residual = np.zeros(
            6,
            dtype=np.float32,
        )

        cumulative_reward = 0.0

        cumulative_terms = (
            defaultdict(float)
        )

        reward_min = float("inf")
        reward_max = float("-inf")

        nan_reward_count = 0

        executed_steps = 0

        final_info = {}

        terminated = False
        truncated = False

        reward_term_names = [
            "approach",
            "grasp",
            "hold",
            "lift",
            "success",
            "residual",
            "excessive_force",
            "unsafe_collision",
            "failure",
        ]

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

            final_info = step_info

            reward = float(
                reward
            )

            if not np.isfinite(
                reward
            ):

                nan_reward_count += 1

            cumulative_reward += (
                reward
            )

            reward_min = min(
                reward_min,
                reward,
            )

            reward_max = max(
                reward_max,
                reward,
            )

            terms = step_info[
                "reward_terms"
            ]

            term_sum = sum(
                float(
                    terms[name]
                )
                for name
                in reward_term_names
            )

            if not np.isclose(
                term_sum,
                reward,
                atol=1e-8,
            ):

                errors.append(
                    "reward terms do not "
                    "sum to reward"
                )

                break

            for name in (
                reward_term_names
            ):

                cumulative_terms[
                    name
                ] += float(
                    terms[name]
                )

            if (
                step % 20 == 0
                or terminated
                or truncated
            ):

                print(
                    f"step={step:3d} "
                    f"phase="
                    f"{step_info['phase_name_before']:14s} "
                    f"r={reward:+.4f} "
                    f"return="
                    f"{cumulative_reward:+.4f} "
                    f"lift="
                    f"{step_info['cube_lift_m']:.5f}"
                )

            if (
                terminated
                or truncated
            ):

                break

        if not final_info.get(
            "success",
            False,
        ):

            errors.append(
                "zero residual reference "
                "episode failed"
            )

        if nan_reward_count != 0:

            errors.append(
                "reward contains NaN/inf"
            )

        # Zero residualなので
        # residual penaltyはほぼ0のはず
        if abs(
            cumulative_terms[
                "residual"
            ]
        ) > 1e-9:

            errors.append(
                "zero residual produced "
                "residual penalty"
            )

        passed = (
            len(errors) == 0
        )

        # ==================================================
        # Report
        # ==================================================

        report = {

            "synthetic_tests":
                synthetic,

            "integration": {

                "executed_steps":
                    executed_steps,

                "success":
                    bool(
                        final_info.get(
                            "success",
                            False,
                        )
                    ),

                "terminal_reason":
                    final_info.get(
                        "terminal_reason"
                    ),

                "cube_lift_m":
                    float(
                        final_info.get(
                            "cube_lift_m",
                            0.0,
                        )
                    ),

                "episode_return":
                    float(
                        cumulative_reward
                    ),

                "reward_min":
                    float(
                        reward_min
                    ),

                "reward_max":
                    float(
                        reward_max
                    ),

                "nan_reward_count":
                    nan_reward_count,

                "cumulative_reward_terms":
                    dict(
                        cumulative_terms
                    ),
            },

            "reward_config":
                config.get(
                    "day10",
                    {},
                ),

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

        # ==================================================
        # Print
        # ==================================================

        print()
        print("=" * 72)
        print("Cumulative Reward Terms")
        print("=" * 72)

        for name in reward_term_names:

            print(
                f"{name:20s}: "
                f"{cumulative_terms[name]:+.6f}"
            )

        print()
        print("=" * 72)
        print("Day 10 Result")
        print("=" * 72)

        print(
            "executed steps:",
            executed_steps,
        )

        print(
            "success:",
            final_info.get(
                "success",
                False,
            ),
        )

        print(
            "cube lift:",
            final_info.get(
                "cube_lift_m",
                0.0,
            ),
        )

        print(
            "episode return:",
            cumulative_reward,
        )

        print(
            "reward range:",
            reward_min,
            "to",
            reward_max,
        )

        print(
            "NaN rewards:",
            nan_reward_count,
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