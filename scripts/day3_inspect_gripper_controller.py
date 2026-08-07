#!/usr/bin/env python3
"""Day 3: UR3e＋EZGripperのcontroller構成を検査する。

確認する内容:
- 環境全体のaction次元
- armとgripperのsub-controller
- 各controllerが担当するaction区間
- 各controllerが担当する関節
- active joint名、関節上限
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"


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
        choices=["physx_cpu", "physx_cuda"],
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/day3_gripper_controller.json"),
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


def get_active_joints(robot: Any) -> list[Any]:
    if hasattr(robot, "active_joints"):
        return list(robot.active_joints)

    if hasattr(robot, "get_active_joints"):
        return list(robot.get_active_joints())

    raise AttributeError( "robotからactive jointを取得できません")


def get_controller_space(controller: Any) -> Any | None:
    """Controller固有のaction spaceを取得する。"""

    if hasattr(controller, "single_action_space"):
        return controller.single_action_space

    if hasattr(controller, "action_space"):
        return controller.action_space

    return None


def get_controller_joint_names(controller: Any) -> list[str]:
    """Controllerが担当する関節名を可能な範囲で取得する。"""

    config = getattr(controller, "config", None)

    if config is not None:
        names = getattr(config, "joint_names", None)

        if names is not None:
            return [str(name) for name in names]

    joints = getattr(controller, "joints", None)

    if joints is not None:
        result = []

        for joint in joints:
            result.append(str(getattr(joint, "name", joint)))

        return result

    return []


def serialize_space(space: Any) -> dict[str, Any]:
    if space is None:
        return {
            "shape": None,
            "low": None,
            "high": None,
        }

    return {
        "shape": list(space.shape),
        "low": np.asarray(
            space.low,
            dtype=float,
        ).reshape(-1).tolist(),
        "high": np.asarray(
            space.high,
            dtype=float,
        ).reshape(-1).tolist(),
    }


def main() -> int:
    args = parse_args()

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {CONFIG_PATH}")

    with CONFIG_PATH.open("r",
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
            "robot.uidまたは--robot-uidを指定してください"
        )

    control_mode = (
        args.control_mode
        or config["project"]["control_mode"]
    )

    # import時にUR3ePickLift-v0を登録する
    import envs.ur3e_pick_lift  # noqa: F401

    env = gym.make(
        args.env_id,
        robot_uids=robot_uid,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend=args.sim_backend,
        render_mode=None,
    )

    try:
        obs, reset_info = env.reset(
            seed=int(config["project"]["seed"])
        )

        base_env = env.unwrapped
        agent = base_env.agent
        robot = agent.robot
        controller = agent.controller

        active_joints = get_active_joints(robot)
        active_joint_names = [
            joint.name for joint in active_joints
        ]

        qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        qlimits = first_env(
            robot.get_qlimits()
        ).astype(float)

        subcontrollers = getattr(
            controller,
            "controllers",
            None,
        )

        controller_reports: list[dict[str, Any]] = []
        warnings: list[str] = []
        errors: list[str] = []

        if not isinstance(subcontrollers, Mapping):
            errors.append(
                "agent.controller.controllersを取得できません。"
                "腕とグリッパがCombinedControllerとして"
                "構成されているか確認してください。"
            )
            subcontrollers = {}

        offset = 0

        for name, subcontroller in subcontrollers.items():
            space = get_controller_space(subcontroller)

            if space is None:
                action_dim = 0
            else:
                action_dim = int(
                    np.prod(space.shape)
                )

            action_start = offset
            action_stop = offset + action_dim
            offset = action_stop

            config_object = getattr(
                subcontroller,
                "config",
                None,
            )

            controller_reports.append(
                {
                    "name": str(name),
                    "class": type(
                        subcontroller
                    ).__name__,
                    "config_class": (
                        type(config_object).__name__
                        if config_object is not None
                        else None
                    ),
                    "joint_names": (
                        get_controller_joint_names(
                            subcontroller
                        )
                    ),
                    "action_slice": [
                        action_start,
                        action_stop,
                    ],
                    "action_space": (
                        serialize_space(space)
                    ),
                }
            )

        total_action_dim = int(
            np.prod(env.action_space.shape)
        )

        if offset != total_action_dim:
            warnings.append(
                "sub-controllerのaction次元合計と"
                "環境action次元が一致しません: "
                f"{offset} != {total_action_dim}"
            )

        lower_controller_names = [
            report["name"].lower()
            for report in controller_reports
        ]

        has_arm = any(
            "arm" in name
            for name in lower_controller_names
        )

        has_gripper = any(
            (
                "gripper" in name
                or "finger" in name
                or "hand" in name
            )
            for name in lower_controller_names
        )

        if not has_arm:
            warnings.append(
                "名前にarmを含むcontrollerが見つかりません"
            )

        if not has_gripper:
            errors.append(
                "グリッパ用controllerが見つかりません。"
                "agents/ur3e_ezgripper.pyの"
                "_controller_configsを修正する必要があります。"
            )

        if total_action_dim <= 6:
            errors.append(
                f"action次元が{total_action_dim}です。"
                "腕6関節だけがactionへ含まれ、"
                "グリッパ制御が含まれていない可能性があります。"
            )

        report = {
            "env_id": args.env_id,
            "robot_uid": robot_uid,
            "control_mode": control_mode,
            "sim_backend": args.sim_backend,
            "observation_shape": list(
                getattr(obs, "shape", [])
            ),
            "environment_action_space": (
                serialize_space(env.action_space)
            ),
            "active_joint_names": active_joint_names,
            "qpos_initial": qpos.tolist(),
            "qlimits": qlimits.tolist(),
            "controllers": controller_reports,
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

        print("=" * 72)
        print("Day 3 Controller Inspection")
        print("=" * 72)
        print("environment action shape:", env.action_space.shape)
        print("active joints:", active_joint_names)
        print()

        for item in controller_reports:
            print(f"controller : {item['name']}")
            print(f"class      : {item['class']}")
            print(f"config     : {item['config_class']}")
            print(f"joints     : {item['joint_names']}")
            print(f"slice      : {item['action_slice']}")
            print(
                "space      :",
                item["action_space"],
            )
            print("-" * 72)

        for warning in warnings:
            print("[WARN]", warning)

        for error in errors:
            print("[ERROR]", error)

        print("report:", args.output)

        return 1 if errors else 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())