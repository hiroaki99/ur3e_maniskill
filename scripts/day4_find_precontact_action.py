#!/usr/bin/env python3

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

    if array.ndim >= 2 and array.shape[0] == 1:
        return array[0]

    return array


def get_controller_space(controller):

    if hasattr(controller, "single_action_space"):
        return controller.single_action_space

    return controller.action_space


def build_controller_slices(controllers):

    result = {}
    offset = 0

    for name, controller in controllers.items():

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

        if keyword in str(name).lower():
            return str(name)

    raise RuntimeError(
        f"{keyword} controllerがありません"
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

    return list(names)


def compute_arm_hold_action(
    current_qpos,
    indices,
    target,
    max_delta,
):

    current = current_qpos[
        indices
    ]

    error = (
        target
        - current
    )

    delta = np.clip(
        error,
        -max_delta,
        max_delta,
    )

    return np.clip(
        delta / max_delta,
        -1.0,
        1.0,
    ).astype(np.float32)


def get_collision_bounds(link):

    mesh = get_component_mesh(
        link._objs[0],
        to_world_frame=True,
    )

    if mesh is None:
        raise RuntimeError(
            f"{link.name}: collision meshなし"
        )

    return np.asarray(
        mesh.bounds,
        dtype=np.float64,
    )


def measure(
    left_link,
    right_link,
):

    lb = get_collision_bounds(
        left_link
    )

    rb = get_collision_bounds(
        right_link
    )

    left_center = (
        lb[0] + lb[1]
    ) / 2.0

    right_center = (
        rb[0] + rb[1]
    ) / 2.0

    center = (
        left_center
        + right_center
    ) / 2.0

    # 現在の姿勢では開閉方向はほぼworld X
    inner_gap = (
        rb[0][0]
        - lb[1][0]
    )

    return (
        float(inner_gap),
        center,
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
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config[
            "project"
        ]["control_mode"],
        sim_backend="physx_cpu",
        render_mode=None,
    )

    try:

        env.reset(
            seed=int(
                config["project"]["seed"]
            )
        )

        base_env = env.unwrapped

        robot = (
            base_env.agent.robot
        )

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

        initial_qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        arm_target = (
            initial_qpos[
                arm_indices
            ].copy()
        )

        max_delta = float(
            config[
                "residual"
            ][
                "max_joint_delta_rad"
            ]
        )

        left_name = (
            config["gripper"]
            ["left_contact_links"][0]
        )

        right_name = (
            config["gripper"]
            ["right_contact_links"][0]
        )

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

        total_dim = (
            env.action_space.shape[0]
        )

        def move_gripper(
            command,
            steps=80,
        ):

            for _ in range(steps):

                qpos = first_env(
                    robot.get_qpos()
                ).astype(float)

                action = np.zeros(
                    total_dim,
                    dtype=np.float32,
                )

                action[
                    arm_slice
                ] = (
                    compute_arm_hold_action(
                        qpos,
                        arm_indices,
                        arm_target,
                        max_delta,
                    )
                )

                action[
                    gripper_slice
                ] = command

                env.step(action)

        cube_size = float(
            config["cube"]["size"]
        )

        target_gap = 0.055

        # -1～+1を0.05刻み
        commands = np.linspace(
            -1.0,
            1.0,
            41,
        )

        results = []

        print(
            "command      gap[m]      "
            "gap[mm]     center xyz"
        )

        print("-" * 90)

        for command in commands:

            move_gripper(
                float(command)
            )

            gap, center = measure(
                left_link,
                right_link,
            )

            results.append(
                (
                    command,
                    gap,
                    center,
                )
            )

            print(
                f"{command:+.2f}      "
                f"{gap:.6f}    "
                f"{gap * 1000:7.2f}    "
                f"{center}"
            )

        # 55 mmへ最も近いもの
        best = min(
            results,
            key=lambda item:
                abs(
                    item[1]
                    - target_gap
                ),
        )

        print()
        print("=" * 72)

        print(
            "Cube size:",
            cube_size,
            "m",
        )

        print(
            "Target pre-contact gap:",
            target_gap,
            "m",
        )

        print(
            "Recommended command:",
            float(best[0]),
        )

        print(
            "Measured gap:",
            float(best[1]),
            "m",
        )

        print(
            "Grasp center:",
            best[2],
        )

        print("=" * 72)

    finally:

        env.close()


if __name__ == "__main__":
    main()