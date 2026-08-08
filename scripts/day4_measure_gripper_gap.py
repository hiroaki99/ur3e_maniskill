#!/usr/bin/env python3
"""
EZGripperを open / pre-contact / close 状態へ動かし、
左右finger pad collision間の距離を測定する。

目的:
- 50 mm cubeを現在のグリッパ設定で挟めるか確認
- open_action / close_actionの方向確認
- collision geometryの実際の移動量確認
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

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


def to_numpy(value):

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def first_env(value):

    array = to_numpy(value)

    if (
        array.ndim >= 2
        and array.shape[0] == 1
    ):
        return array[0]

    return array


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


def find_controller(
    controllers,
    keyword,
):

    for name in controllers:

        if keyword in str(name).lower():
            return str(name)

    raise RuntimeError(
        f"{keyword} controllerがありません: "
        f"{list(controllers.keys())}"
    )


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
    ).astype(np.float32)


def get_collision_bounds(link):
    """
    Linkのcollision geometryをworld座標AABBとして取得。
    """

    raw_link = link._objs[0]

    mesh = get_component_mesh(
        raw_link,
        to_world_frame=True,
    )

    if mesh is None:
        raise RuntimeError(
            f"{link.name}: collision meshがありません"
        )

    return np.asarray(
        mesh.bounds,
        dtype=np.float64,
    )


def measure_gap(
    left_link,
    right_link,
):
    """
    現在姿勢で左右finger pad collisionを測定する。

    今回は左右finger padの中心差がほぼworld X方向なので、
    X方向を閉方向としてinner gapも計算する。
    """

    left_bounds = get_collision_bounds(
        left_link
    )

    right_bounds = get_collision_bounds(
        right_link
    )

    left_center = (
        left_bounds[0]
        + left_bounds[1]
    ) / 2.0

    right_center = (
        right_bounds[0]
        + right_bounds[1]
    ) / 2.0

    center_distance = np.linalg.norm(
        right_center
        - left_center
    )

    # 左padの右端と右padの左端
    inner_gap_x = (
        right_bounds[0][0]
        - left_bounds[1][0]
    )

    grasp_center = (
        left_center
        + right_center
    ) / 2.0

    return {
        "left_bounds": left_bounds,
        "right_bounds": right_bounds,
        "left_center": left_center,
        "right_center": right_center,
        "center_distance": center_distance,
        "inner_gap_x": inner_gap_x,
        "grasp_center": grasp_center,
    }


def print_measurement(
    label,
    measurement,
    cube_size,
):

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    print(
        "left collision center :",
        measurement[
            "left_center"
        ],
    )

    print(
        "right collision center:",
        measurement[
            "right_center"
        ],
    )

    print(
        "grasp center          :",
        measurement[
            "grasp_center"
        ],
    )

    print(
        "center distance       :",
        f"{measurement['center_distance']:.6f} m",
    )

    print(
        "inner gap X           :",
        f"{measurement['inner_gap_x']:.6f} m",
    )

    print(
        "cube size             :",
        f"{cube_size:.6f} m",
    )

    clearance = (
        measurement[
            "inner_gap_x"
        ]
        - cube_size
    )

    print(
        "gap - cube size       :",
        f"{clearance:.6f} m",
    )

    if measurement["inner_gap_x"] <= cube_size:

        print(
            "[OK] この姿勢では50 mm cubeに"
            "接触可能な間隔です。"
        )

    else:

        print(
            "[INFO] この姿勢ではcubeより"
            f"{clearance * 1000:.1f} mm広いです。"
        )


def main():

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

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
        render_mode="human",
    )

    try:

        env.reset(
            seed=int(
                config["project"]["seed"]
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
        # Arm joint indices
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
                controllers[
                    arm_name
                ]
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

        max_arm_delta = float(
            config[
                "residual"
            ][
                "max_joint_delta_rad"
            ]
        )

        total_action_dim = (
            env.action_space.shape[0]
        )

        # --------------------------------------------------
        # Finger collision links
        # --------------------------------------------------

        left_name = config[
            "gripper"
        ][
            "left_contact_links"
        ][0]

        right_name = config[
            "gripper"
        ][
            "right_contact_links"
        ][0]

        left_link = (
            robot.links_map[
                left_name
            ]
        )

        right_link = (
            robot.links_map[
                right_name
            ]
        )

        cube_size = float(
            config[
                "cube"
            ]["size"]
        )

        # --------------------------------------------------
        # Helper
        # --------------------------------------------------

        def run_gripper(
            command,
            steps,
        ):

            for _ in range(steps):

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
                    compute_arm_hold_action(
                        current_qpos,
                        arm_joint_indices,
                        arm_hold_qpos,
                        max_arm_delta,
                    )
                )

                action[
                    gripper_slice
                ] = command

                env.step(
                    action
                )

                env.render()

                time.sleep(
                    0.01
                )

        # --------------------------------------------------
        # Commands
        # --------------------------------------------------

        open_action = float(
            config[
                "gripper"
            ][
                "open_action"
            ]
        )

        close_action = float(
            config[
                "gripper"
            ][
                "close_action"
            ]
        )

        pre_contact_action = float(
            config[
                "gripper"
            ].get(
                "pre_contact_action",
                0.0,
            )
        )

        # --------------------------------------------------
        # OPEN
        # --------------------------------------------------

        run_gripper(
            open_action,
            60,
        )

        open_measurement = (
            measure_gap(
                left_link,
                right_link,
            )
        )

        print_measurement(
            "OPEN",
            open_measurement,
            cube_size,
        )

        # --------------------------------------------------
        # PRE-CONTACT
        # --------------------------------------------------

        run_gripper(
            pre_contact_action,
            60,
        )

        pre_measurement = (
            measure_gap(
                left_link,
                right_link,
            )
        )

        print_measurement(
            "PRE-CONTACT",
            pre_measurement,
            cube_size,
        )

        # --------------------------------------------------
        # CLOSE
        # --------------------------------------------------

        run_gripper(
            close_action,
            100,
        )

        close_measurement = (
            measure_gap(
                left_link,
                right_link,
            )
        )

        print_measurement(
            "CLOSE",
            close_measurement,
            cube_size,
        )

        print()
        print("=" * 72)
        print("Summary")
        print("=" * 72)

        print(
            "OPEN gap:",
            open_measurement[
                "inner_gap_x"
            ],
        )

        print(
            "PRE gap :",
            pre_measurement[
                "inner_gap_x"
            ],
        )

        print(
            "CLOSE gap:",
            close_measurement[
                "inner_gap_x"
            ],
        )

        print(
            "Cube     :",
            cube_size,
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()