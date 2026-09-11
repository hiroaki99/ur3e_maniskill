#!/usr/bin/env python3
"""UR3ePickLift-v0のDay 2 smoke test。

確認項目:
- 環境を生成できる
- resetできる
- UR3e＋EZGripperを読み込める
- テーブルと動的キューブが存在する
- 200 step後もキューブがテーブル上に留まる
- NaNや例外が発生しない
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-id",
        type=str,
        default="UR3ePickLift-v0",
    )

    parser.add_argument(
        "--robot-uid",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--control-mode",
        type=str,
        default=None,
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
        "--steps",
        type=int,
        default=200,
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
            "reports/day2_pick_lift_env.json"
        ),
    )

    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def first_env(value: Any) -> np.ndarray:
    array = to_numpy(value)

    if array.ndim >= 2 and array.shape[0] == 1:
        return array[0]

    return array


def get_active_joint_names(robot: Any) -> list[str]:
    if hasattr(robot, "active_joints"):
        return [
            joint.name
            for joint in robot.active_joints
        ]

    if hasattr(robot, "get_active_joints"):
        return [
            joint.name
            for joint in robot.get_active_joints()
        ]

    return []


def get_link_names(robot: Any) -> list[str]:
    if hasattr(robot, "links"):
        return [
            link.name
            for link in robot.links
        ]

    if hasattr(robot, "get_links"):
        return [
            link.name
            for link in robot.get_links()
        ]

    return []


def main() -> int:
    args = parse_args()

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"設定ファイルがありません: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    robot_uid = (
        args.robot_uid
        or os.environ.get("UR3E_EZGRIPPER_UID")
        or config["robot"].get("uid")
    )

    if not robot_uid:
        raise ValueError(
            "EZGripper AgentのUIDがありません。"
            "--robot-uidまたはrobot.uidを設定してください。"
        )

    control_mode = (
        args.control_mode
        or config["project"]["control_mode"]
    )

    # env moduleをimportすると@register_envが実行される
    import envs.ur3e_pick_lift  # noqa: F401

    render_mode = (
        "human"
        if args.render
        else None
    )

    env = gym.make(
        args.env_id,
        robot_uids=robot_uid,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend=args.sim_backend,
        render_mode=render_mode,
    )

    try:
        obs, reset_info = env.reset(
            seed=int(config["project"]["seed"])
        )

        base_env = env.unwrapped
        robot = base_env.agent.robot
        cube = base_env.cube

        joint_names = get_active_joint_names(robot)
        link_names = get_link_names(robot)

        initial_cube_position = first_env(
            cube.pose.p
        ).astype(float)

        initial_qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        action_low = np.asarray(
            env.action_space.low,
            dtype=float,
        )
        action_high = np.asarray(
            env.action_space.high,
            dtype=float,
        )

        print("=" * 72)
        print("Day 2 UR3ePickLift-v0 smoke test")
        print("=" * 72)
        print("env id       :", args.env_id)
        print("robot uid    :", robot_uid)
        print("control mode :", control_mode)
        print("sim backend  :", args.sim_backend)
        print("obs shape    :", getattr(obs, "shape", None))
        print("action shape :", env.action_space.shape)
        print("action low   :", action_low.tolist())
        print("action high  :", action_high.tolist())
        print("joint names  :", joint_names)
        print("link names   :", link_names)
        print("initial qpos :", initial_qpos.tolist())
        print(
            "cube initial :",
            initial_cube_position.tolist(),
        )

        rewards: list[float] = []

        for step in range(args.steps):
            # delta position制御におけるゼロ差分入力
            action = np.zeros(
                env.action_space.shape,
                dtype=np.float32,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            reward_array = to_numpy(reward)
            rewards.append(
                float(reward_array.reshape(-1)[0])
            )

            if args.render:
                env.render()
                time.sleep(args.sleep)

            if step % 50 == 0:
                cube_position = first_env(
                    cube.pose.p
                ).astype(float)

                print(
                    f"step={step:4d}, "
                    f"cube={cube_position.tolist()}, "
                    f"reward={rewards[-1]:.6f}"
                )

        final_cube_position = first_env(
            cube.pose.p
        ).astype(float)

        final_qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        cube_xy_drift = float(
            np.linalg.norm(
                final_cube_position[:2]
                - initial_cube_position[:2]
            )
        )

        expected_cube_z = float(
            config["cube"]["position"][2]
        )

        cube_z_error = abs(
            float(final_cube_position[2])
            - expected_cube_z
        )

        qpos_drift = np.abs(
            final_qpos - initial_qpos
        )

        warnings: list[str] = []
        errors: list[str] = []

        if not np.all(
            np.isfinite(final_cube_position)
        ):
            errors.append(
                "キューブ位置にNaNまたはinfがあります"
            )

        if not np.all(np.isfinite(final_qpos)):
            errors.append(
                "ロボット関節角にNaNまたはinfがあります"
            )

        if cube_xy_drift > 0.005:
            warnings.append(
                "キューブの水平移動が5 mmを超えました: "
                f"{cube_xy_drift:.6f} m"
            )

        if cube_z_error > 0.005:
            warnings.append(
                "キューブ中心高さの誤差が5 mmを超えました: "
                f"{cube_z_error:.6f} m"
            )

        # EZGripperらしいlink名が含まれているか簡易確認
        lower_link_names = [
            name.lower()
            for name in link_names
        ]

        has_gripper_link = any(
            (
                "gripper" in name
                or "finger" in name
                or "left_knuckle" in name
                or "right_knuckle" in name
            )
            for name in lower_link_names
        )

        if not has_gripper_link:
            warnings.append(
                "link名からEZGripperを確認できませんでした。"
                "GUIとurdf_pathを確認してください。"
            )

        report = {
            "env_id": args.env_id,
            "robot_uid": robot_uid,
            "control_mode": control_mode,
            "sim_backend": args.sim_backend,
            "observation_shape": list(
                getattr(obs, "shape", [])
            ),
            "action_shape": list(
                env.action_space.shape
            ),
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
            "joint_names": joint_names,
            "link_names": link_names,
            "initial_qpos": initial_qpos.tolist(),
            "final_qpos": final_qpos.tolist(),
            "max_qpos_drift": float(
                qpos_drift.max(initial=0.0)
            ),
            "initial_cube_position": (
                initial_cube_position.tolist()
            ),
            "final_cube_position": (
                final_cube_position.tolist()
            ),
            "cube_xy_drift_m": cube_xy_drift,
            "cube_z_error_m": cube_z_error,
            "mean_reward": float(
                np.mean(rewards)
            ),
            "warnings": warnings,
            "errors": errors,
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

        print("-" * 72)
        print(
            "final cube   :",
            final_cube_position.tolist(),
        )
        print(
            "cube XY drift:",
            f"{cube_xy_drift:.6f} m",
        )
        print(
            "cube Z error :",
            f"{cube_z_error:.6f} m",
        )
        print(
            "max q drift  :",
            f"{report['max_qpos_drift']:.6e} rad",
        )

        for warning in warnings:
            print("[WARN]", warning)

        for error in errors:
            print("[ERROR]", error)

        print("report       :", args.output)

        return 1 if errors else 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())