#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
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

sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "scripts"
    ),
)

from day6_common import (
    build_controller_slices,
    find_controller,
    first_env,
    get_controller_joint_names,
    get_grasp_center,
    plan_cartesian_segment,
)


CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lift-height",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "trajectories/"
            "day6_pick_lift_reference.json"
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

    day6 = config["day6"]

    lift_height = (
        args.lift_height
        if args.lift_height
        is not None
        else float(
            day6[
                "reference_lift_height_m"
            ]
        )
    )

    import envs.ur3e_pick_lift  # noqa

    env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config[
            "robot"
        ]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config[
            "project"
        ]["control_mode"],
        sim_backend="physx_cpu",
    )

    try:

        env.reset(
            seed=int(
                config[
                    "project"
                ]["seed"]
            )
        )

        base_env = env.unwrapped
        robot = base_env.agent.robot

        controllers = (
            base_env
            .agent
            .controller
            .controllers
        )

        slices = (
            build_controller_slices(
                controllers
            )
        )

        arm_name = find_controller(
            controllers,
            "arm",
        )

        gripper_name = (
            find_controller(
                controllers,
                "gripper",
            )
        )

        arm_slice = slices[
            arm_name
        ]

        gripper_slice = slices[
            gripper_name
        ]

        active_names = [
            joint.name
            for joint
            in robot.active_joints
        ]

        name_to_index = {
            name: i
            for i, name
            in enumerate(active_names)
        }

        arm_names = (
            get_controller_joint_names(
                controllers[
                    arm_name
                ]
            )
        )

        arm_indices = [
            name_to_index[name]
            for name in arm_names
        ]

        max_delta = float(
            config[
                "residual"
            ][
                "max_joint_delta_rad"
            ]
        )

        pre_contact = float(
            config[
                "gripper"
            ][
                "pre_contact_action"
            ]
        )

        # --------------------------------------------------
        # Gripperをpre-contact状態へ
        # --------------------------------------------------

        initial_arm = first_env(
            robot.get_qpos()
        ).astype(float)[
            arm_indices
        ].copy()

        for _ in range(60):

            current = first_env(
                robot.get_qpos()
            ).astype(float)

            action = np.zeros(
                env.action_space.shape,
                dtype=np.float32,
            )

            error = (
                initial_arm
                - current[
                    arm_indices
                ]
            )

            action[
                arm_slice
            ] = np.clip(
                error / max_delta,
                -1.0,
                1.0,
            )

            action[
                gripper_slice
            ] = pre_contact

            env.step(action)

        # --------------------------------------------------
        # 基準状態
        # --------------------------------------------------

        full_qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        home_arm = (
            full_qpos[
                arm_indices
            ].copy()
        )

        qlimits = first_env(
            robot.get_qlimits()
        ).astype(float)

        qlimits_arm = (
            qlimits[
                arm_indices
            ]
        )

        left_links = (
            base_env.left_contact_links
        )

        right_links = (
            base_env.right_contact_links
        )

        home_center = (
            get_grasp_center(
                left_links,
                right_links,
            )
        )

        cube_position = first_env(
            base_env.cube.pose.p
        ).astype(float)

        # --------------------------------------------------
        # Cartesian targets
        # --------------------------------------------------

        grasp_offset = np.asarray(
            day6[
                "grasp_center_offset_m"
            ],
            dtype=np.float64,
        )

        grasp_target = (
            cube_position
            + grasp_offset
        )

        pregrasp_target = (
            grasp_target.copy()
        )

        pregrasp_target[
            2
        ] += float(
            day6[
                "pregrasp_height_m"
            ]
        )

        lift_target = (
            grasp_target.copy()
        )

        lift_target[
            2
        ] += lift_height

        print("=" * 72)
        print("Day 6 Reference Planner")
        print("=" * 72)

        print(
            "home center    :",
            home_center.tolist(),
        )

        print(
            "cube position  :",
            cube_position.tolist(),
        )

        print(
            "pregrasp target:",
            pregrasp_target.tolist(),
        )

        print(
            "grasp target   :",
            grasp_target.tolist(),
        )

        print(
            "lift target    :",
            lift_target.tolist(),
        )

        # --------------------------------------------------
        # IK parameters
        # --------------------------------------------------

        common_kwargs = {
            "robot": robot,
            "device": base_env.device,
            "left_links": left_links,
            "right_links": right_links,
            "template_qpos": full_qpos,
            "arm_indices": arm_indices,
            "qlimits_arm": qlimits_arm,
            "epsilon": float(
                day6[
                    "ik_epsilon_rad"
                ]
            ),
            "damping": float(
                day6[
                    "ik_damping"
                ]
            ),
            "tolerance": float(
                day6[
                    "ik_tolerance_m"
                ]
            ),
            "max_iterations": int(
                day6[
                    "ik_max_iterations"
                ]
            ),
            "max_step": float(
                day6[
                    "ik_max_step_rad"
                ]
            ),
        }

        # --------------------------------------------------
        # Home -> Pre-grasp
        # --------------------------------------------------

        print()
        print("Planning Home -> Pre-grasp")

        to_pregrasp = (
            plan_cartesian_segment(
                start_arm_qpos=home_arm,
                target_position=(
                    pregrasp_target
                ),
                waypoint_count=int(
                    day6[
                        "to_pregrasp_waypoints"
                    ]
                ),
                **common_kwargs,
            )
        )

        q_pregrasp = np.asarray(
            to_pregrasp[
                "final_arm_qpos"
            ]
        )

        print(
            "final error:",
            to_pregrasp[
                "errors_m"
            ][-1],
        )

        # --------------------------------------------------
        # Pre-grasp -> Grasp
        # --------------------------------------------------

        print()
        print("Planning Pre-grasp -> Grasp")

        descend = (
            plan_cartesian_segment(
                start_arm_qpos=q_pregrasp,
                target_position=(
                    grasp_target
                ),
                waypoint_count=int(
                    day6[
                        "descend_waypoints"
                    ]
                ),
                **common_kwargs,
            )
        )

        q_grasp = np.asarray(
            descend[
                "final_arm_qpos"
            ]
        )

        print(
            "final error:",
            descend[
                "errors_m"
            ][-1],
        )

        # --------------------------------------------------
        # Grasp -> Lift
        # --------------------------------------------------

        print()
        print("Planning Grasp -> Lift")

        lift = (
            plan_cartesian_segment(
                start_arm_qpos=q_grasp,
                target_position=(
                    lift_target
                ),
                waypoint_count=int(
                    day6[
                        "lift_waypoints"
                    ]
                ),
                **common_kwargs,
            )
        )

        print(
            "final error:",
            lift[
                "errors_m"
            ][-1],
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        all_reference_qpos = (
            to_pregrasp[
                "arm_qpos"
            ]
            + descend[
                "arm_qpos"
            ]
            + lift[
                "arm_qpos"
            ]
        )

        report = {
            "robot_uid":
                config[
                    "robot"
                ]["uid"],

            "control_mode":
                config[
                    "project"
                ][
                    "control_mode"
                ],

            "arm_joint_names":
                arm_names,

            "home_arm_qpos":
                home_arm.tolist(),

            "home_grasp_center_m":
                home_center.tolist(),

            "cube_initial_position_m":
                cube_position.tolist(),

            "pregrasp_target_m":
                pregrasp_target.tolist(),

            "grasp_target_m":
                grasp_target.tolist(),

            "lift_target_m":
                lift_target.tolist(),

            "planned_lift_height_m":
                float(
                    lift_height
                ),

            "segments": {
                "to_pregrasp":
                    to_pregrasp,

                "descend":
                    descend,

                "lift":
                    lift,
            },

            "reference_arm_qpos":
                all_reference_qpos,
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
        print("Reference trajectory generated")
        print("=" * 72)

        print(
            "total waypoints:",
            len(
                all_reference_qpos
            ),
        )

        print(
            "output:",
            args.output,
        )

        return 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )