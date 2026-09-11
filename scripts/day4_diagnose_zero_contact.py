#!/usr/bin/env python3

from __future__ import annotations

import sys
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


def scalar(value) -> float:
    """
    numpy 1.25以降の
    float(array([x]))
    DeprecationWarningも避ける。
    """
    a = to_numpy(value)
    return float(a.reshape(-1)[0])


def inspect_link_collision(
    name,
    link,
):
    print()
    print("-" * 72)
    print("link:", name)

    # ManiSkill Linkが内部で管理している
    # PhysX articulation link
    raw_link = link._objs[0]

    collision_shapes = (
        raw_link.get_collision_shapes()
    )

    print(
        "collision shape count:",
        len(collision_shapes),
    )

    for i, shape in enumerate(
        collision_shapes
    ):
        try:
            groups = (
                shape.get_collision_groups()
            )
        except Exception:
            groups = None

        print(
            f"  shape[{i}] "
            f"type={type(shape).__name__} "
            f"groups={groups}"
        )

    if len(collision_shapes) == 0:
        print(
            "[ERROR] このLinkには"
            "collision shapeがありません"
        )

        return None

    # collision geometryをworld座標meshへ変換
    try:
        mesh = get_component_mesh(
            raw_link,
            to_world_frame=True,
        )

    except Exception as exc:

        print(
            "[ERROR] collision mesh取得失敗:",
            repr(exc),
        )

        return None

    if mesh is None:
        print(
            "[ERROR] collision mesh=None"
        )
        return None

    bounds = np.asarray(
        mesh.bounds,
        dtype=float,
    )

    center = (
        bounds[0]
        + bounds[1]
    ) / 2.0

    extent = (
        bounds[1]
        - bounds[0]
    )

    print(
        "collision AABB min:",
        bounds[0],
    )

    print(
        "collision AABB max:",
        bounds[1],
    )

    print(
        "collision center  :",
        center,
    )

    print(
        "collision extent  :",
        extent,
    )

    return center


def main():

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

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
    )

    try:

        env.reset(
            seed=int(
                config["project"]["seed"]
            )
        )

        base_env = env.unwrapped

        # --------------------------------------------------
        # 少し物理を進めて、
        # cubeをテーブル上で安定させる
        # --------------------------------------------------

        action = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )

        # Day3で確認したscript値。
        # グリッパを開いておくだけ。
        if (
            "gripper" in config
            and "open_action"
            in config["gripper"]
            and env.action_space.shape[0] >= 7
        ):
            action[6] = float(
                config["gripper"][
                    "open_action"
                ]
            )

        for _ in range(30):
            env.step(action)

        # --------------------------------------------------
        # 1. cube-table接触テスト
        # --------------------------------------------------

        print("=" * 72)
        print("1. Cube - Table Contact API Sanity Check")
        print("=" * 72)

        table_force_vector = (
            base_env.scene
            .get_pairwise_contact_forces(
                base_env.cube,
                base_env.table_scene.table,
            )
        )

        table_force = np.linalg.norm(
            to_numpy(
                table_force_vector
            ),
            axis=1,
        )

        print(
            "cube-table force vector:",
            to_numpy(
                table_force_vector
            ),
        )

        print(
            "cube-table force norm:",
            scalar(table_force),
            "N",
        )

        if scalar(table_force) < 0.01:
            print(
                "[WARN] cube-table forceも"
                "ほぼ0 Nです。"
            )
            print(
                "接触API、テーブルcollision、"
                "キューブ配置を先に確認してください。"
            )

        else:
            print(
                "[OK] pairwise contact APIは"
                "機能している可能性が高いです。"
            )

        # --------------------------------------------------
        # 2. Configured contact links
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("2. Finger Collision Geometry")
        print("=" * 72)

        links_map = (
            base_env.agent.robot.links_map
        )

        left_names = config[
            "gripper"
        ]["left_contact_links"]

        right_names = config[
            "gripper"
        ]["right_contact_links"]

        left_centers = []

        for name in left_names:

            if name not in links_map:
                print(
                    "[ERROR] link not found:",
                    name,
                )
                continue

            center = (
                inspect_link_collision(
                    name,
                    links_map[name],
                )
            )

            if center is not None:
                left_centers.append(
                    center
                )

        right_centers = []

        for name in right_names:

            if name not in links_map:
                print(
                    "[ERROR] link not found:",
                    name,
                )
                continue

            center = (
                inspect_link_collision(
                    name,
                    links_map[name],
                )
            )

            if center is not None:
                right_centers.append(
                    center
                )

        # --------------------------------------------------
        # 3. Collision geometryから
        #    把持中心候補を計算
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("3. Collision-based Grasp Center")
        print("=" * 72)

        if (
            len(left_centers) > 0
            and len(right_centers) > 0
        ):

            left_center = np.mean(
                np.stack(
                    left_centers
                ),
                axis=0,
            )

            right_center = np.mean(
                np.stack(
                    right_centers
                ),
                axis=0,
            )

            grasp_center = (
                left_center
                + right_center
            ) / 2.0

            finger_axis = (
                right_center
                - left_center
            )

            gap = np.linalg.norm(
                finger_axis
            )

            print(
                "left collision center :",
                left_center,
            )

            print(
                "right collision center:",
                right_center,
            )

            print(
                "grasp center candidate:",
                grasp_center,
            )

            print(
                "collision center gap   :",
                gap,
                "m",
            )

            # 現在のLink原点方式とも比較
            left_origin = np.mean(
                np.stack(
                    [
                        to_numpy(
                            links_map[name]
                            .pose.p
                        )[0]
                        for name
                        in left_names
                    ]
                ),
                axis=0,
            )

            right_origin = np.mean(
                np.stack(
                    [
                        to_numpy(
                            links_map[name]
                            .pose.p
                        )[0]
                        for name
                        in right_names
                    ]
                ),
                axis=0,
            )

            old_center = (
                left_origin
                + right_origin
            ) / 2.0

            print()
            print(
                "old link-origin center :",
                old_center,
            )

            print(
                "center difference      :",
                grasp_center
                - old_center,
            )

            print(
                "difference norm        :",
                np.linalg.norm(
                    grasp_center
                    - old_center
                ),
                "m",
            )

        else:

            print(
                "[ERROR] 左右どちらかの"
                "collision centerを取得できません"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()