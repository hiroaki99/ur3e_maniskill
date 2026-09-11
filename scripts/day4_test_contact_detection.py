#!/usr/bin/env python3
"""
Day 4:
EZGripper左右指とキューブとの接触判定を検証する。

このスクリプトは接触検証のため、
キューブを一時的にグリッパ付近へteleportする。

これはPick動作ではなく、接触APIの単体テストである。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from mani_skill.utils.structs.pose import Pose

from mani_skill.utils.geometry.trimesh_utils import (
    get_component_mesh,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)


# ----------------------------------------------------------
# Arguments
# ----------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-id",
        default="UR3ePickLift-v0",
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
        "--scenario",
        default="center",
        choices=[
            "far",
            "center",
            "left",
            "right",
        ],
    )

    parser.add_argument(
        "--pre-contact-steps",
        type=int,
        default=60,
        help="pre-contact姿勢を安定させる制御ステップ数",
    )

    parser.add_argument(
        "--close-steps",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--lateral-offset-factor",
        type=float,
        default=0.60,
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
        default=None,
    )

    return parser.parse_args()


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

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


def scalar_first(value):
    """num_envs=1の値をPythonスカラーへ変換する。"""
    array = np.asarray(first_env(value))

    if array.size != 1:
        raise ValueError(
            f"スカラー値を想定しました: shape={array.shape}"
        )

    return array.reshape(-1)[0].item()


def get_controller_space(controller):

    if hasattr(
        controller,
        "single_action_space",
    ):
        return controller.single_action_space

    return controller.action_space


def build_controller_slices(
    controllers: Mapping,
):

    result = {}

    offset = 0

    for name, controller in controllers.items():

        space = get_controller_space(
            controller
        )

        dimension = int(
            np.prod(space.shape)
        )

        result[str(name)] = slice(
            offset,
            offset + dimension,
        )

        offset += dimension

    return result


def get_controller_joint_names(
    controller,
):

    config = controller.config

    names = getattr(
        config,
        "joint_names",
        None,
    )

    if names is None:
        return []

    return [
        str(name)
        for name in names
    ]


def find_controller(
    controllers,
    keyword,
):

    for name in controllers:

        if keyword in name.lower():
            return name

    raise RuntimeError(
        f"{keyword} controllerが見つかりません: "
        f"{list(controllers.keys())}"
    )


def compute_arm_hold_action(
    current_qpos,
    arm_joint_indices,
    target_qpos,
    max_delta_rad,
):

    current = current_qpos[
        arm_joint_indices
    ]

    error = (
        target_qpos
        - current
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


def get_collision_center(link):

    mesh = get_component_mesh(
        link._objs[0],
        to_world_frame=True,
    )

    if mesh is None:
        raise RuntimeError(
            f"{link.name}: collision meshがありません"
        )

    bounds = np.asarray(
        mesh.bounds,
        dtype=np.float64,
    )

    return (
        bounds[0]
        + bounds[1]
    ) / 2.0


def get_current_grasp_center(
    base_env,
):

    left_centers = np.stack(
        [
            get_collision_center(link)
            for link
            in base_env.left_contact_links
        ]
    )

    right_centers = np.stack(
        [
            get_collision_center(link)
            for link
            in base_env.right_contact_links
        ]
    )

    left = left_centers.mean(
        axis=0
    )

    right = right_centers.mean(
        axis=0
    )

    return (
        left + right
    ) / 2.0

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    args = parse_args()

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    robot_uid = (
        os.environ.get(
            "UR3E_EZGRIPPER_UID"
        )
        or config["robot"]["uid"]
    )

    control_mode = (
        config["project"]["control_mode"]
    )

    gripper_config = (
        config["gripper"]
    )

    pre_contact_action = float(
        gripper_config["pre_contact_action"]
    )

    close_value = float(
        gripper_config["close_action"]
    )

    # open_value = float(
    #     gripper_config["open_action"]
    # )

    print(
        "pre-contact action:",
        pre_contact_action,
    )
    print(
        "pre-contact steps :",
        args.pre_contact_steps,
    )

    close_value = float(
        gripper_config["close_action"]
    )

    max_arm_delta = float(
        config["residual"][
            "max_joint_delta_rad"
        ]
    )

    import envs.ur3e_pick_lift  # noqa

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
                config["project"]["seed"]
            )
        )

        base_env = env.unwrapped

        # Day 4接触判定の単体試験中だけ、落下を防ぐ
        base_env.cube.disable_gravity = True

        robot = base_env.agent.robot

        combined_controller = (
            base_env.agent.controller
        )

        controllers = (
            combined_controller.controllers
        )

        if not isinstance(
            controllers,
            Mapping,
        ):
            raise RuntimeError(
                "CombinedControllerではありません"
            )

        slices = build_controller_slices(
            controllers
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
        # Active joints
        # --------------------------------------------------

        active_joint_names = [
            joint.name
            for joint in robot.active_joints
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
                controllers[arm_name]
            )
        )

        arm_joint_indices = [
            name_to_index[name]
            for name in arm_joint_names
        ]

        initial_qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        arm_hold_qpos = (
            initial_qpos[
                arm_joint_indices
            ].copy()
        )

        total_action_dim = (
            env.action_space.shape[0]
        )

        # --------------------------------------------------
        # Helper: one step
        # --------------------------------------------------

        def step_with_gripper(
            gripper_command
        ):

            current_qpos = first_env(
                robot.get_qpos()
            ).astype(float)

            action = np.zeros(
                total_action_dim,
                dtype=np.float32,
            )

            action[arm_slice] = (
                compute_arm_hold_action(
                    current_qpos,
                    arm_joint_indices,
                    arm_hold_qpos,
                    max_arm_delta,
                )
            )

            action[gripper_slice] = (
                gripper_command
            )

            result = env.step(
                action
            )

            if args.render:
                env.render()
                time.sleep(args.sleep)

            return result

        # ==================================================
        # 1. Gripperをpre-contact姿勢へ移動
        # ==================================================

        pre_contact_value = float(
            config["gripper"]["pre_contact_action"]
        )

        for _ in range(
            args.pre_contact_steps
        ):
            step_with_gripper(
                pre_contact_value
            )

        # ==================================================
        # 2. pre-contact状態のcollision中心を計算
        # ==================================================

        left_collision_centers = np.stack(
            [
                get_collision_center(link)
                for link
                in base_env.left_contact_links
            ],
            axis=0,
        )

        right_collision_centers = np.stack(
            [
                get_collision_center(link)
                for link
                in base_env.right_contact_links
            ],
            axis=0,
        )

        left_center = (
            left_collision_centers.mean(
                axis=0
            )
        )

        right_center = (
            right_collision_centers.mean(
                axis=0
            )
        )

        center = (
            left_center
            + right_center
        ) / 2.0

        finger_vector = (
            right_center
            - left_center
        )

        finger_distance = float(
            np.linalg.norm(
                finger_vector
            )
        )

        if finger_distance < 1e-6:
            raise RuntimeError(
                "左右finger collision centerが"
                "ほぼ同一です。"
            )

        finger_axis = (
            finger_vector
            / finger_distance
        )

        cube_size = float(
            config["cube"]["size"]
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
            "pre-contact grasp center:",
            center.tolist(),
        )

        print(
            "finger distance:",
            finger_distance,
        )

        # ==================================================
        # 3. ScenarioごとのCube位置
        # ==================================================

        if args.scenario == "far":

            target_position = np.asarray(
                config["cube"]["position"],
                dtype=np.float32,
            )

        elif args.scenario == "center":

            target_position = (
                center.copy()
            )

        elif args.scenario == "left":

            target_position = (
                center
                - finger_axis
                * cube_size
                * args.lateral_offset_factor
            )

        elif args.scenario == "right":

            target_position = (
                center
                + finger_axis
                * cube_size
                * args.lateral_offset_factor
            )

        else:

            raise ValueError(
                args.scenario
            )

        print("=" * 72)
        print("Day 4 Contact Test")
        print("=" * 72)

        print(
            "scenario       :",
            args.scenario,
        )

        print(
            "gripper center :",
            center.tolist(),
        )

        print(
            "cube target    :",
            target_position.tolist(),
        )

        # ==================================================
        # 4. Cubeを試験位置へteleport
        # ==================================================

        p = torch.tensor(
            target_position,
            dtype=torch.float32,
            device=base_env.device,
        ).unsqueeze(0)

        q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=base_env.device,
        )

        zero_velocity = torch.zeros(
            (1, 3),
            dtype=torch.float32,
            device=base_env.device,
        )

        base_env.cube.set_pose(
            Pose.create_from_pq(
                p=p,
                q=q,
            )
        )

        base_env.cube.set_linear_velocity(
            zero_velocity
        )

        base_env.cube.set_angular_velocity(
            zero_velocity
        )

        actual_cube_position = first_env(
            base_env.cube.pose.p
        ).astype(float)

        print(
            "cube requested:",
            target_position.tolist(),
        )

        print(
            "cube actual   :",
            actual_cube_position.tolist(),
        )

        # ==================================================
        # 5. Closeしながら接触力を記録
        # ==================================================

        records = []

        peak_left = 0.0
        peak_right = 0.0

        left_contact_steps = 0
        right_contact_steps = 0
        both_contact_steps = 0

        contact_started = False

        for step in range(
            args.close_steps
        ):

            # ----------------------------------------------
            # Day 4の単体試験では、
            # 接触が始まるまではCube位置を保持する。
            # ----------------------------------------------

            if not contact_started:

                p = torch.tensor(
                    target_position,
                    dtype=torch.float32,
                    device=base_env.device,
                ).unsqueeze(0)

                q = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                    device=base_env.device,
                )

                base_env.cube.set_pose(
                    Pose.create_from_pq(
                        p=p,
                        q=q,
                    )
                )

                base_env.cube.set_linear_velocity(
                    zero_velocity
                )

                base_env.cube.set_angular_velocity(
                    zero_velocity
                )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = step_with_gripper(
                close_value
            )

            # ----------------------------------------------
            # Contact force
            # ----------------------------------------------

            left_force = float(
                scalar_first(
                    info[
                        "left_contact_force"
                    ]
                )
            )

            right_force = float(
                scalar_first(
                    info[
                        "right_contact_force"
                    ]
                )
            )

            left_contact = (
                left_force
                >= base_env.contact_force_threshold
            )

            right_contact = (
                right_force
                >= base_env.contact_force_threshold
            )

            # ----------------------------------------------
            # Grasp information
            # ----------------------------------------------

            grasp_candidate = bool(
                scalar_first(
                    info[
                        "is_grasp_candidate"
                    ]
                )
            )

            center_distance = float(
                scalar_first(
                    info[
                        "cube_to_gripper_center_distance"
                    ]
                )
            )

            # ----------------------------------------------
            # Peak force更新
            # ----------------------------------------------

            peak_left = max(
                peak_left,
                left_force,
            )

            peak_right = max(
                peak_right,
                right_force,
            )

            # ----------------------------------------------
            # Contact step数更新
            # ----------------------------------------------

            if left_contact:
                left_contact_steps += 1

            if right_contact:
                right_contact_steps += 1

            if (
                left_contact
                and right_contact
            ):
                both_contact_steps += 1

            # ----------------------------------------------
            # 最初の接触が発生したら
            # teleportによる保持を終了
            # ----------------------------------------------

            if (
                left_contact
                or right_contact
            ):
                contact_started = True

            # ----------------------------------------------
            # Record
            # ----------------------------------------------

            records.append(
                {
                    "step": int(step),
                    "left_force_n": (
                        left_force
                    ),
                    "right_force_n": (
                        right_force
                    ),
                    "left_contact": bool(
                        left_contact
                    ),
                    "right_contact": bool(
                        right_contact
                    ),
                    "both_contact": bool(
                        left_contact
                        and right_contact
                    ),
                    "grasp_candidate": (
                        grasp_candidate
                    ),
                    "center_distance_m": (
                        center_distance
                    ),
                }
            )

            if (
                step % 5 == 0
                or left_contact
                or right_contact
            ):

                print(
                    f"step={step:3d} "
                    f"L={left_force:8.4f} N "
                    f"R={right_force:8.4f} N "
                    f"Lcontact={left_contact} "
                    f"Rcontact={right_contact} "
                    f"grasp={grasp_candidate}"
                )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        print("-" * 72)

        print(
            "peak left force   :",
            f"{peak_left:.6f} N",
        )

        print(
            "peak right force  :",
            f"{peak_right:.6f} N",
        )

        print(
            "left contact steps:",
            left_contact_steps,
        )

        print(
            "right contact steps:",
            right_contact_steps,
        )

        print(
            "both contact steps:",
            both_contact_steps,
        )

        output = args.output

        if output is None:

            output = Path(
                "reports/"
                f"day4_contact_detection_"
                f"{args.scenario}.json"
            )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "scenario": args.scenario,

            "contact_force_threshold_n": (
                base_env.contact_force_threshold
            ),

            "finger_distance_m": (
                float(finger_distance)
            ),

            "cube_target_position_m": (
                target_position.tolist()
            ),

            "peak_left_force_n": (
                peak_left
            ),

            "peak_right_force_n": (
                peak_right
            ),

            "left_contact_steps": (
                left_contact_steps
            ),

            "right_contact_steps": (
                right_contact_steps
            ),

            "both_contact_steps": (
                both_contact_steps
            ),

            "records": records,
        }

        with output.open(
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
            output,
        )

        return 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())