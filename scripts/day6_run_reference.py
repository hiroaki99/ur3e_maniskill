#!/usr/bin/env python3

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
    scalar_first,
)


CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path(
            "trajectories/"
            "day6_pick_lift_reference.json"
        ),
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
        "--success-lift-height",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/"
            "day6_reference_pick_lift.json"
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

    with args.trajectory.open(
        "r",
        encoding="utf-8",
    ) as file:

        trajectory = json.load(
            file
        )

    day6 = config["day6"]

    success_lift_height = (
        args.success_lift_height
        if args.success_lift_height
        is not None
        else float(
            day6[
                "success_lift_height_m"
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
        render_mode=(
            "human"
            if args.render
            else None
        ),
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

        if (
            arm_names
            != trajectory[
                "arm_joint_names"
            ]
        ):

            raise RuntimeError(
                "trajectoryと現在の"
                "arm joint構成が一致しません"
            )

        arm_indices = [
            name_to_index[name]
            for name in arm_names
        ]

        total_dim = (
            env.action_space.shape[0]
        )

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

        close_action = float(
            config[
                "gripper"
            ][
                "close_action"
            ]
        )

        contact_threshold = float(
            config[
                "gripper"
            ][
                "contact_force_threshold_n"
            ]
        )

        max_force = float(
            day6[
                "max_contact_force_n"
            ]
        )

        control_steps = int(
            day6[
                "control_steps_per_waypoint"
            ]
        )

        # --------------------------------------------------
        # Robot command
        # --------------------------------------------------

        def step_robot(
            target_arm,
            gripper_command,
        ):

            current_qpos = first_env(
                robot.get_qpos()
            ).astype(float)

            error = (
                np.asarray(
                    target_arm
                )
                - current_qpos[
                    arm_indices
                ]
            )

            action = np.zeros(
                total_dim,
                dtype=np.float32,
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
            ] = gripper_command

            result = env.step(
                action
            )

            if args.render:

                env.render()

                time.sleep(
                    args.sleep
                )

            return result

        def run_path(
            segment,
            segment_name,
            gripper_command,
            abort_on_cube_contact=False,
        ):

            print()
            print("=" * 72)
            print(segment_name)
            print("=" * 72)

            q_path = (
                trajectory[
                    "segments"
                ][
                    segment
                ][
                    "arm_qpos"
                ]
            )

            last_info = None

            for index, q_target in enumerate(
                q_path
            ):

                for _ in range(
                    control_steps
                ):

                    (
                        obs,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = step_robot(
                        q_target,
                        gripper_command,
                    )

                    last_info = info

                left_contact = bool(
                    scalar_first(
                        info[
                            "left_contact"
                        ]
                    )
                )

                right_contact = bool(
                    scalar_first(
                        info[
                            "right_contact"
                        ]
                    )
                )

                print(
                    f"waypoint "
                    f"{index + 1:02d}/"
                    f"{len(q_path):02d} "
                    f"L={left_contact} "
                    f"R={right_contact}"
                )

                if (
                    abort_on_cube_contact
                    and (
                        left_contact
                        or right_contact
                    )
                ):

                    raise RuntimeError(
                        "Grasp開始前にCubeへ"
                        "接触しました。"
                        "軌道または位置合わせを"
                        "確認してください。"
                    )

            return last_info

        # --------------------------------------------------
        # Cube initial state
        # --------------------------------------------------

        cube_initial_position = (
            first_env(
                base_env.cube.pose.p
            )
            .astype(float)
            .copy()
        )

        print("=" * 72)
        print("Day 6 Reference Pick-and-Lift")
        print("=" * 72)

        print(
            "cube initial:",
            cube_initial_position.tolist(),
        )

        # --------------------------------------------------
        # Gripper pre-contact
        # --------------------------------------------------

        current_arm = first_env(
            robot.get_qpos()
        ).astype(float)[
            arm_indices
        ]

        for _ in range(60):

            step_robot(
                current_arm,
                pre_contact,
            )

        # --------------------------------------------------
        # Home -> Pre-grasp
        # --------------------------------------------------

        run_path(
            "to_pregrasp",
            "Phase 1: Home -> Pre-grasp",
            pre_contact,
            abort_on_cube_contact=True,
        )

        final_pregrasp = np.asarray(
            trajectory[
                "segments"
            ][
                "to_pregrasp"
            ][
                "final_arm_qpos"
            ]
        )

        for _ in range(
            int(
                day6[
                    "pregrasp_hold_steps"
                ]
            )
        ):

            step_robot(
                final_pregrasp,
                pre_contact,
            )

        # --------------------------------------------------
        # Descend
        # --------------------------------------------------

        run_path(
            "descend",
            "Phase 2: Pre-grasp -> Grasp pose",
            pre_contact,
            abort_on_cube_contact=True,
        )

        grasp_arm_qpos = np.asarray(
            trajectory[
                "segments"
            ][
                "descend"
            ][
                "final_arm_qpos"
            ]
        )

        # --------------------------------------------------
        # Grasp
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("Phase 3: Grasp")
        print("=" * 72)

        grasp_ramp_steps = int(
            day6[
                "grasp_ramp_steps"
            ]
        )

        bilateral_required = int(
            day6[
                "bilateral_contact_streak"
            ]
        )

        commands = np.linspace(
            pre_contact,
            close_action,
            grasp_ramp_steps,
        )

        bilateral_streak = 0

        grasp_command = None

        first_both_left = None
        first_both_right = None

        for step, command in enumerate(
            commands
        ):

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = step_robot(
                grasp_arm_qpos,
                float(command),
            )

            left_force = scalar_first(
                info[
                    "left_contact_force"
                ]
            )

            right_force = scalar_first(
                info[
                    "right_contact_force"
                ]
            )

            left_contact = (
                left_force
                >= contact_threshold
            )

            right_contact = (
                right_force
                >= contact_threshold
            )

            print(
                f"step={step:3d} "
                f"cmd={command:+.4f} "
                f"L={left_force:7.3f} N "
                f"R={right_force:7.3f} N"
            )

            if (
                left_force > max_force
                or right_force > max_force
            ):

                raise RuntimeError(
                    "接触力が暫定上限を"
                    "超えました"
                )

            if (
                left_contact
                and right_contact
            ):

                bilateral_streak += 1

                if (
                    first_both_left
                    is None
                ):

                    first_both_left = (
                        left_force
                    )

                    first_both_right = (
                        right_force
                    )

            else:

                bilateral_streak = 0

            if (
                bilateral_streak
                >= bilateral_required
            ):

                grasp_command = float(
                    command
                )

                print(
                    "Bilateral grasp detected."
                )

                break

        if grasp_command is None:

            raise RuntimeError(
                "Graspに失敗しました"
            )

        direction = np.sign(
            close_action
            - pre_contact
        )

        grasp_command = float(
            np.clip(
                grasp_command
                + direction
                * float(
                    day6[
                        "squeeze_margin_action"
                    ]
                ),
                -1.0,
                1.0,
            )
        )

        # --------------------------------------------------
        # Hold before lift
        # --------------------------------------------------

                # --------------------------------------------------
        # Pre-Lift Stable Grasp Validation
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("Phase 3.5: Stable Grasp Check")
        print("=" * 72)

        pre_lift_hold_steps = int(
            day6.get(
                "pre_lift_hold_steps",
                30,
            )
        )

        required_ratio = float(
            day6.get(
                "pre_lift_required_contact_ratio",
                0.8,
            )
        )

        stable_grasp_force = float(
            day6.get(
                "stable_grasp_force_n",
                5.0,
            )
        )

        stable_both_steps = 0

        pre_lift_peak_left = 0.0
        pre_lift_peak_right = 0.0

        last_left_force = 0.0
        last_right_force = 0.0

        for step in range(
            pre_lift_hold_steps
        ):

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = step_robot(
                grasp_arm_qpos,
                grasp_command,
            )

            left_force = scalar_first(
                info[
                    "left_contact_force"
                ]
            )

            right_force = scalar_first(
                info[
                    "right_contact_force"
                ]
            )

            last_left_force = (
                left_force
            )

            last_right_force = (
                right_force
            )

            pre_lift_peak_left = max(
                pre_lift_peak_left,
                left_force,
            )

            pre_lift_peak_right = max(
                pre_lift_peak_right,
                right_force,
            )

            left_stable = (
                left_force
                >= stable_grasp_force
            )

            right_stable = (
                right_force
                >= stable_grasp_force
            )

            if (
                left_stable
                and right_stable
            ):

                stable_both_steps += 1

            if (
                step % 5 == 0
                or step
                == pre_lift_hold_steps - 1
            ):

                print(
                    f"hold={step:3d} "
                    f"L={left_force:8.3f} N "
                    f"R={right_force:8.3f} N "
                    f"stable="
                    f"{left_stable and right_stable}"
                )

        required_steps = int(
            np.ceil(
                pre_lift_hold_steps
                * required_ratio
            )
        )

        stable_grasp = (
            stable_both_steps
            >= required_steps
        )

        print()
        print(
            "stable both-contact steps:",
            stable_both_steps,
            "/",
            pre_lift_hold_steps,
        )

        print(
            "required:",
            required_steps,
        )

        print(
            "final forces:",
            f"L={last_left_force:.3f} N,",
            f"R={last_right_force:.3f} N",
        )

        print(
            "stable grasp:",
            stable_grasp,
        )

        if not stable_grasp:

            raise RuntimeError(
                "Lift開始前に安定把持を"
                "維持できませんでした。 "
                f"grasp_command="
                f"{grasp_command:.4f}, "
                f"stable_steps="
                f"{stable_both_steps}/"
                f"{pre_lift_hold_steps}"
            )

        cube_before_lift = (
            first_env(
                base_env.cube.pose.p
            )
            .astype(float)
            .copy()
        )

        print(
            "cube before lift:",
            cube_before_lift.tolist(),
        )

        # --------------------------------------------------
        # Lift
        # --------------------------------------------------

        run_path(
            "lift",
            "Phase 4: Lift",
            grasp_command,
            abort_on_cube_contact=False,
        )

        lift_arm_qpos = np.asarray(
            trajectory[
                "segments"
            ][
                "lift"
            ][
                "final_arm_qpos"
            ]
        )

        # --------------------------------------------------
        # Final hold
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("Phase 5: Final Hold")
        print("=" * 72)

        final_hold_steps = int(
            day6[
                "final_hold_steps"
            ]
        )

        final_both_contact_steps = 0

        final_peak_left = 0.0
        final_peak_right = 0.0

        for _ in range(
            final_hold_steps
        ):

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = step_robot(
                lift_arm_qpos,
                grasp_command,
            )

            left_force = scalar_first(
                info[
                    "left_contact_force"
                ]
            )

            right_force = scalar_first(
                info[
                    "right_contact_force"
                ]
            )

            left_contact = (
                left_force
                >= contact_threshold
            )

            right_contact = (
                right_force
                >= contact_threshold
            )

            if (
                left_contact
                and right_contact
            ):
                final_both_contact_steps += 1

            final_peak_left = max(
                final_peak_left,
                left_force,
            )

            final_peak_right = max(
                final_peak_right,
                right_force,
            )

        cube_final_position = (
            first_env(
                base_env.cube.pose.p
            )
            .astype(float)
            .copy()
        )

        # initial table positionから評価
        cube_lift = float(
            cube_final_position[2]
            - cube_initial_position[2]
        )

        success_hold_steps = int(
            day6[
                "success_hold_steps"
            ]
        )

        success = (
            cube_lift
            >= success_lift_height
            and final_both_contact_steps
            >= success_hold_steps
        )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("Day 6 Result")
        print("=" * 72)

        print(
            "grasp command:",
            grasp_command,
        )

        print(
            "cube initial z:",
            cube_initial_position[2],
        )

        print(
            "cube final z  :",
            cube_final_position[2],
        )

        print(
            "cube lift     :",
            cube_lift,
            "m",
        )

        print(
            "final both contact:",
            final_both_contact_steps,
            "/",
            final_hold_steps,
        )

        print(
            "SUCCESS:",
            success,
        )

        report = {
            "trajectory":
                str(
                    args.trajectory
                ),

            "cube_initial_position_m":
                cube_initial_position.tolist(),

            "cube_before_lift_position_m":
                cube_before_lift.tolist(),

            "cube_final_position_m":
                cube_final_position.tolist(),

            "cube_lift_m":
                cube_lift,

            "success_lift_height_m":
                success_lift_height,

            "grasp_command":
                grasp_command,

            "first_both_contact_left_force_n":
                first_both_left,

            "first_both_contact_right_force_n":
                first_both_right,

            "final_hold_steps":
                final_hold_steps,

            "final_both_contact_steps":
                final_both_contact_steps,

            "final_peak_left_force_n":
                final_peak_left,

            "final_peak_right_force_n":
                final_peak_right,

            "success":
                bool(
                    success
                ),
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

        print(
            "report:",
            args.output,
        )

        return (
            0
            if success
            else 1
        )

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )