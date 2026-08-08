#!/usr/bin/env python3
"""
Day 5:
EZGripperによるGrasp -> Hold -> Micro Lift試験。

目的
----
1. Gripperをpre-contact位置へ移動
2. Cubeを指間中央へ1回だけ配置
3. Gripperを徐々に閉じる
4. 左右接触を検出した時点で閉動作を停止
5. Cubeの重力を有効化
6. 一定時間保持
7. 数値JacobianからUR3eの1 cm上昇関節変化を求める
8. 実際に腕を動かしCubeが一緒に上昇するか確認

注意
----
CubeのteleportはGrasp開始前だけ使用する。
Grasp成立後はPoseを強制変更しない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from mani_skill.utils.geometry.trimesh_utils import (
    get_component_mesh,
)
from mani_skill.utils.structs.pose import Pose


# ==========================================================
# Path
# ==========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT),
)

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-id",
        type=str,
        default="UR3ePickLift-v0",
    )

    parser.add_argument(
        "--sim-backend",
        type=str,
        default="physx_cpu",
        choices=[
            "physx_cpu",
            "physx_cuda",
        ],
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
        "--skip-lift",
        action="store_true",
        help="Grasp/Holdだけを検証する",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/day5_grasp_hold_lift.json"
        ),
    )

    return parser.parse_args()


# ==========================================================
# Utility
# ==========================================================

def to_numpy(value: Any):

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def first_env(value: Any):

    array = to_numpy(value)

    if (
        array.ndim >= 2
        and array.shape[0] == 1
    ):
        return array[0]

    return array


def scalar_first(value) -> float:

    array = np.asarray(
        first_env(value)
    )

    if array.size != 1:

        raise ValueError(
            "scalarを想定していますが "
            f"shape={array.shape}"
        )

    return float(
        array.reshape(-1)[0]
    )


def get_controller_space(
    controller,
):

    if hasattr(
        controller,
        "single_action_space",
    ):
        return (
            controller.single_action_space
        )

    return controller.action_space


def build_controller_slices(
    controllers: Mapping,
):

    result = {}

    offset = 0

    for name, controller in (
        controllers.items()
    ):

        space = get_controller_space(
            controller
        )

        dim = int(
            np.prod(space.shape)
        )

        result[str(name)] = slice(
            offset,
            offset + dim,
        )

        offset += dim

    return result


def find_controller(
    controllers,
    keyword,
):

    for name in controllers:

        if keyword in str(
            name
        ).lower():

            return str(name)

    raise RuntimeError(
        f"{keyword} controllerが"
        "見つかりません: "
        f"{list(controllers.keys())}"
    )


def get_controller_joint_names(
    controller,
):

    names = getattr(
        controller.config,
        "joint_names",
        None,
    )

    if names is None:
        return []

    return [
        str(name)
        for name in names
    ]


# ==========================================================
# Collision geometry
# ==========================================================

def get_collision_center(link):
    """
    Link originではなくcollision mesh中心を取得。
    """

    raw_link = link._objs[0]

    mesh = get_component_mesh(
        raw_link,
        to_world_frame=True,
    )

    if mesh is None:

        raise RuntimeError(
            f"{link.name}: "
            "collision meshがありません"
        )

    bounds = np.asarray(
        mesh.bounds,
        dtype=np.float64,
    )

    return (
        bounds[0]
        + bounds[1]
    ) / 2.0


def get_grasp_center(
    left_links,
    right_links,
):

    left_centers = np.stack(
        [
            get_collision_center(link)
            for link in left_links
        ],
        axis=0,
    )

    right_centers = np.stack(
        [
            get_collision_center(link)
            for link in right_links
        ],
        axis=0,
    )

    left_center = (
        left_centers.mean(
            axis=0
        )
    )

    right_center = (
        right_centers.mean(
            axis=0
        )
    )

    center = (
        left_center
        + right_center
    ) / 2.0

    return (
        center,
        left_center,
        right_center,
    )


# ==========================================================
# Arm control
# ==========================================================

def compute_arm_target_action(
    current_qpos,
    arm_joint_indices,
    target_arm_qpos,
    max_delta_rad,
):
    """
    pd_joint_delta_pos用。

    target_arm_qpos - current_qposを
    [-1,1] actionへ変換する。
    """

    current_arm = current_qpos[
        arm_joint_indices
    ]

    error = (
        target_arm_qpos
        - current_arm
    )

    delta = np.clip(
        error,
        -max_delta_rad,
        max_delta_rad,
    )

    normalized = (
        delta
        / max_delta_rad
    )

    return np.clip(
        normalized,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    # ------------------------------------------------------
    # Config
    # ------------------------------------------------------

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    robot_uid = (
        config["robot"]["uid"]
    )

    control_mode = (
        config[
            "project"
        ][
            "control_mode"
        ]
    )

    gripper_cfg = (
        config["gripper"]
    )

    day5_cfg = (
        config["day5"]
    )

    pre_contact_action = float(
        gripper_cfg[
            "pre_contact_action"
        ]
    )

    close_action = float(
        gripper_cfg[
            "close_action"
        ]
    )

    contact_threshold = float(
        gripper_cfg[
            "contact_force_threshold_n"
        ]
    )

    max_joint_delta = float(
        config[
            "residual"
        ][
            "max_joint_delta_rad"
        ]
    )

    grasp_ramp_steps = int(
        day5_cfg[
            "grasp_ramp_steps"
        ]
    )

    bilateral_required = int(
        day5_cfg[
            "bilateral_contact_streak"
        ]
    )

    squeeze_margin = float(
        day5_cfg[
            "squeeze_margin_action"
        ]
    )

    hold_steps = int(
        day5_cfg[
            "hold_steps"
        ]
    )

    lift_height = float(
        day5_cfg[
            "micro_lift_height_m"
        ]
    )

    lift_steps = int(
        day5_cfg[
            "lift_steps"
        ]
    )

    jac_eps = float(
        day5_cfg[
            "jacobian_epsilon_rad"
        ]
    )

    max_lift_joint_change = float(
        day5_cfg[
            "max_lift_joint_change_rad"
        ]
    )

    max_contact_force = float(
        day5_cfg[
            "max_contact_force_n"
        ]
    )

    max_hold_drop = float(
        day5_cfg[
            "max_hold_drop_m"
        ]
    )

    lift_success_threshold = float(
        day5_cfg[
            "micro_lift_success_threshold_m"
        ]
    )

    # ------------------------------------------------------
    # Register environment
    # ------------------------------------------------------

    import envs.ur3e_pick_lift  # noqa: F401

    env = gym.make(
        args.env_id,
        robot_uids=robot_uid,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend=args.sim_backend,
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
                ][
                    "seed"
                ]
            )
        )

        base_env = env.unwrapped

        robot = (
            base_env.agent.robot
        )

        # --------------------------------------------------
        # Contact links
        # --------------------------------------------------

        left_links = (
            base_env.left_contact_links
        )

        right_links = (
            base_env.right_contact_links
        )

        # --------------------------------------------------
        # Controllers
        # --------------------------------------------------

        controllers = (
            base_env
            .agent
            .controller
            .controllers
        )

        if not isinstance(
            controllers,
            Mapping,
        ):

            raise RuntimeError(
                "CombinedControllerではありません"
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

        gripper_name = find_controller(
            controllers,
            "gripper",
        )

        arm_slice = slices[
            arm_name
        ]

        gripper_slice = slices[
            gripper_name
        ]

        # --------------------------------------------------
        # Arm joint indices
        # --------------------------------------------------

        active_joint_names = [
            joint.name
            for joint
            in robot.active_joints
        ]

        name_to_index = {
            name: index
            for index, name
            in enumerate(
                active_joint_names
            )
        }

        arm_joint_names = (
            get_controller_joint_names(
                controllers[
                    arm_name
                ]
            )
        )

        arm_joint_indices = [
            name_to_index[name]
            for name
            in arm_joint_names
        ]

        total_action_dim = (
            env.action_space.shape[0]
        )

        # --------------------------------------------------
        # Helper
        # --------------------------------------------------

        current_arm_target = first_env(
            robot.get_qpos()
        ).astype(float)[
            arm_joint_indices
        ].copy()

        current_gripper_action = (
            pre_contact_action
        )

        def step_robot(
            arm_target,
            gripper_command,
        ):

            current_qpos = first_env(
                robot.get_qpos()
            ).astype(float)

            action = np.zeros(
                total_action_dim,
                dtype=np.float32,
            )

            action[
                arm_slice
            ] = (
                compute_arm_target_action(
                    current_qpos,
                    arm_joint_indices,
                    arm_target,
                    max_joint_delta,
                )
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

        # ==================================================
        # Phase 1
        # Pre-contact姿勢
        # ==================================================

        print()
        print("=" * 72)
        print("Phase 1: Pre-contact")
        print("=" * 72)

        for _ in range(60):

            step_robot(
                current_arm_target,
                pre_contact_action,
            )

        (
            grasp_center,
            left_center,
            right_center,
        ) = get_grasp_center(
            left_links,
            right_links,
        )

        print(
            "left collision center :",
            left_center.tolist(),
        )

        print(
            "right collision center:",
            right_center.tolist(),
        )

        print(
            "grasp center:",
            grasp_center.tolist(),
        )

        # ==================================================
        # Phase 2
        # Micro-lift用数値JacobianをGrasp前に取得
        # ==================================================

        print()
        print("=" * 72)
        print("Phase 2: Numerical Jacobian")
        print("=" * 72)

        base_qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        base_grasp_center = (
            grasp_center.copy()
        )

        jacobian = np.zeros(
            (3, len(arm_joint_indices)),
            dtype=np.float64,
        )

        for column, joint_index in enumerate(
            arm_joint_indices
        ):

            perturbed_qpos = (
                base_qpos.copy()
            )

            perturbed_qpos[
                joint_index
            ] += jac_eps

            qpos_tensor = torch.tensor(
                perturbed_qpos,
                dtype=torch.float32,
                device=base_env.device,
            ).unsqueeze(0)

            robot.set_qpos(
                qpos_tensor
            )

            robot.set_qvel(
                torch.zeros_like(
                    qpos_tensor
                )
            )

            (
                perturbed_center,
                _,
                _,
            ) = get_grasp_center(
                left_links,
                right_links,
            )

            jacobian[
                :,
                column,
            ] = (
                perturbed_center
                - base_grasp_center
            ) / jac_eps

        # restore
        base_qpos_tensor = torch.tensor(
            base_qpos,
            dtype=torch.float32,
            device=base_env.device,
        ).unsqueeze(0)

        robot.set_qpos(
            base_qpos_tensor
        )

        robot.set_qvel(
            torch.zeros_like(
                base_qpos_tensor
            )
        )

        print(
            "Numerical position Jacobian:"
        )

        print(
            jacobian
        )

        desired_translation = (
            np.array(
                [
                    0.0,
                    0.0,
                    lift_height,
                ],
                dtype=np.float64,
            )
        )

        delta_q = (
            np.linalg.pinv(
                jacobian,
                rcond=1e-4,
            )
            @ desired_translation
        )

        max_abs_dq = float(
            np.max(
                np.abs(
                    delta_q
                )
            )
        )

        print(
            "requested lift:",
            desired_translation,
        )

        print(
            "delta q:",
            delta_q,
        )

        print(
            "max |delta q|:",
            max_abs_dq,
        )

        if (
            not np.all(
                np.isfinite(
                    delta_q
                )
            )
        ):

            raise RuntimeError(
                "Jacobianから求めた"
                "delta_qにNaN/infがあります"
            )

        if (
            max_abs_dq
            > max_lift_joint_change
        ):

            raise RuntimeError(
                "1 cm liftに必要な関節変化量が"
                "大きすぎます: "
                f"{max_abs_dq:.4f} rad > "
                f"{max_lift_joint_change:.4f} rad"
            )

        # ==================================================
        # Phase 3
        # Cubeを1回だけGrasp centerへ配置
        # ==================================================

        print()
        print("=" * 72)
        print("Phase 3: Place Cube")
        print("=" * 72)

        # Grasp成立までは空中のCubeが落下しないようにする
        base_env.cube.disable_gravity = (
            True
        )

        cube_p = torch.tensor(
            grasp_center,
            dtype=torch.float32,
            device=base_env.device,
        ).unsqueeze(0)

        cube_q = torch.tensor(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ],
            dtype=torch.float32,
            device=base_env.device,
        )

        base_env.cube.set_pose(
            Pose.create_from_pq(
                p=cube_p,
                q=cube_q,
            )
        )

        zero_velocity = torch.zeros(
            (1, 3),
            dtype=torch.float32,
            device=base_env.device,
        )

        base_env.cube.set_linear_velocity(
            zero_velocity
        )

        base_env.cube.set_angular_velocity(
            zero_velocity
        )

        print(
            "cube position:",
            first_env(
                base_env.cube.pose.p
            ).tolist(),
        )

        # ==================================================
        # Phase 4
        # 徐々にGripperを閉じる
        # ==================================================

        print()
        print("=" * 72)
        print("Phase 4: Gradual Grasp")
        print("=" * 72)

        close_commands = np.linspace(
            pre_contact_action,
            close_action,
            grasp_ramp_steps,
        )

        bilateral_streak = 0

        grasp_command = None

        grasp_step = None

        peak_left = 0.0
        peak_right = 0.0

        first_both_left = None
        first_both_right = None

        for step, command in enumerate(
            close_commands
        ):

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = step_robot(
                current_arm_target,
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

            peak_left = max(
                peak_left,
                left_force,
            )

            peak_right = max(
                peak_right,
                right_force,
            )

            print(
                f"step={step:3d} "
                f"cmd={command:+.4f} "
                f"L={left_force:8.3f} N "
                f"R={right_force:8.3f} N "
                f"L={left_contact} "
                f"R={right_contact}"
            )

            if (
                left_force
                > max_contact_force
                or right_force
                > max_contact_force
            ):

                raise RuntimeError(
                    "接触力がDay5暫定上限を"
                    "超えました。"
                    f"L={left_force:.2f} N, "
                    f"R={right_force:.2f} N"
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

                grasp_step = int(
                    step
                )

                print()
                print(
                    "Bilateral contact "
                    "established."
                )

                print(
                    "grasp command:",
                    grasp_command,
                )

                break

        if grasp_command is None:

            raise RuntimeError(
                "両側接触を安定して"
                "取得できませんでした"
            )

        # 必要ならほんの少しだけ追加で閉じる
        close_direction = np.sign(
            close_action
            - pre_contact_action
        )

        hold_gripper_action = (
            grasp_command
            + close_direction
            * squeeze_margin
        )

        hold_gripper_action = float(
            np.clip(
                hold_gripper_action,
                -1.0,
                1.0,
            )
        )

        print(
            "hold gripper action:",
            hold_gripper_action,
        )

        # ==================================================
        # Phase 5
        # Gravityを戻してHold
        # ==================================================

        print()
        print("=" * 72)
        print("Phase 5: Hold under gravity")
        print("=" * 72)

        cube_z_before_hold = float(
            first_env(
                base_env.cube.pose.p
            )[2]
        )

        base_env.cube.disable_gravity = (
            False
        )

        hold_left_contact_steps = 0
        hold_right_contact_steps = 0
        hold_both_contact_steps = 0

        max_hold_force_left = 0.0
        max_hold_force_right = 0.0

        for step in range(
            hold_steps
        ):

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = step_robot(
                current_arm_target,
                hold_gripper_action,
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

            if left_contact:
                hold_left_contact_steps += 1

            if right_contact:
                hold_right_contact_steps += 1

            if (
                left_contact
                and right_contact
            ):
                hold_both_contact_steps += 1

            max_hold_force_left = max(
                max_hold_force_left,
                left_force,
            )

            max_hold_force_right = max(
                max_hold_force_right,
                right_force,
            )

            if (
                left_force
                > max_contact_force
                or right_force
                > max_contact_force
            ):

                raise RuntimeError(
                    "Hold中に接触力上限を"
                    "超えました。"
                )

        cube_z_after_hold = float(
            first_env(
                base_env.cube.pose.p
            )[2]
        )

        hold_drop = (
            cube_z_before_hold
            - cube_z_after_hold
        )

        held_under_gravity = (
            hold_drop
            <= max_hold_drop
            and hold_both_contact_steps
            >= int(
                hold_steps * 0.8
            )
        )

        print(
            "cube z before hold:",
            cube_z_before_hold,
        )

        print(
            "cube z after hold :",
            cube_z_after_hold,
        )

        print(
            "hold drop:",
            hold_drop,
        )

        print(
            "both contact steps:",
            hold_both_contact_steps,
            "/",
            hold_steps,
        )

        print(
            "held_under_gravity:",
            held_under_gravity,
        )

        # ==================================================
        # Phase 6
        # Micro Lift
        # ==================================================

        micro_lift_success = False

        cube_lift = 0.0

        if (
            held_under_gravity
            and not args.skip_lift
        ):

            print()
            print("=" * 72)
            print("Phase 6: Micro Lift")
            print("=" * 72)

            arm_qpos_before_lift = (
                first_env(
                    robot.get_qpos()
                )
                .astype(float)[
                    arm_joint_indices
                ]
                .copy()
            )

            target_arm_qpos = (
                arm_qpos_before_lift
                + delta_q
            )

            cube_z_before_lift = float(
                first_env(
                    base_env.cube.pose.p
                )[2]
            )

            for step in range(
                lift_steps
            ):

                progress = (
                    step + 1
                ) / lift_steps

                intermediate_target = (
                    arm_qpos_before_lift
                    + progress
                    * delta_q
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = step_robot(
                    intermediate_target,
                    hold_gripper_action,
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

                if (
                    left_force
                    > max_contact_force
                    or right_force
                    > max_contact_force
                ):

                    print(
                        "[WARN] Micro Lift中に"
                        "接触力上限へ到達。"
                    )

                    break

            cube_z_after_lift = float(
                first_env(
                    base_env.cube.pose.p
                )[2]
            )

            cube_lift = (
                cube_z_after_lift
                - cube_z_before_lift
            )

            # 最終接触
            info = base_env.evaluate()

            final_left_contact = bool(
                scalar_first(
                    info[
                        "left_contact"
                    ]
                )
            )

            final_right_contact = bool(
                scalar_first(
                    info[
                        "right_contact"
                    ]
                )
            )

            micro_lift_success = (
                cube_lift
                >= lift_success_threshold
                and final_left_contact
                and final_right_contact
            )

            print(
                "cube z before lift:",
                cube_z_before_lift,
            )

            print(
                "cube z after lift :",
                cube_z_after_lift,
            )

            print(
                "cube lift:",
                cube_lift,
            )

            print(
                "final left contact :",
                final_left_contact,
            )

            print(
                "final right contact:",
                final_right_contact,
            )

            print(
                "micro_lift_success:",
                micro_lift_success,
            )

        # ==================================================
        # Report
        # ==================================================

        report = {

            "pre_contact_action":
                pre_contact_action,

            "close_action":
                close_action,

            "grasp_ramp_steps":
                grasp_ramp_steps,

            "grasp_step":
                grasp_step,

            "grasp_command":
                grasp_command,

            "hold_gripper_action":
                hold_gripper_action,

            "first_both_contact_left_force_n":
                first_both_left,

            "first_both_contact_right_force_n":
                first_both_right,

            "peak_grasp_left_force_n":
                peak_left,

            "peak_grasp_right_force_n":
                peak_right,

            "hold_steps":
                hold_steps,

            "hold_both_contact_steps":
                hold_both_contact_steps,

            "cube_z_before_hold_m":
                cube_z_before_hold,

            "cube_z_after_hold_m":
                cube_z_after_hold,

            "hold_drop_m":
                hold_drop,

            "held_under_gravity":
                bool(
                    held_under_gravity
                ),

            "numerical_jacobian":
                jacobian.tolist(),

            "micro_lift_target_m":
                lift_height,

            "lift_delta_q_rad":
                delta_q.tolist(),

            "cube_micro_lift_m":
                float(
                    cube_lift
                ),

            "micro_lift_success":
                bool(
                    micro_lift_success
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

        print()
        print("=" * 72)
        print("Day 5 Result")
        print("=" * 72)

        print(
            "Grasp:",
            grasp_command is not None,
        )

        print(
            "Hold:",
            held_under_gravity,
        )

        print(
            "Micro Lift:",
            micro_lift_success,
        )

        print(
            "report:",
            args.output,
        )

        if not held_under_gravity:

            return 2

        if (
            not args.skip_lift
            and not micro_lift_success
        ):

            return 3

        return 0

    finally:

        env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )