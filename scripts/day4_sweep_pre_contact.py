#!/usr/bin/env python3
"""Day 4: EZGripper の pre-contact 行動値を実測する。

正規化グリッパ行動 ``[-1, 1]`` を掃引し、各行動値で十分に安定化した
後の左右指パッド位置からパッド間距離を測る。キューブは使用しないため、
接触・落下の影響を受けずに把持開口の幾何を決められる。

出力JSONの ``recommended_pre_contact_action`` は、キューブ幅に指定した
クリアランスを加えた目標開口に最も近い行動値である。
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
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EZGripperのpre-contact開口を実測する"
    )
    parser.add_argument("--env-id", default="UR3ePickLift-v0")
    parser.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=["physx_cpu", "physx_cuda"],
    )
    parser.add_argument(
        "--action-min",
        type=float,
        default=-1.0,
        help="掃引する正規化グリッパ行動の下限",
    )
    parser.add_argument(
        "--action-max",
        type=float,
        default=1.0,
        help="掃引する正規化グリッパ行動の上限",
    )
    parser.add_argument(
        "--action-step",
        type=float,
        default=0.1,
        help="正規化グリッパ行動の粗探索刻み",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help="各行動値で安定化させる制御ステップ数",
    )
    parser.add_argument(
        "--clearance-m",
        type=float,
        default=0.005,
        help="キューブ幅に加える目標クリアランス [m]",
    )
    parser.add_argument(
        "--target-gap-m",
        type=float,
        default=None,
        help="目標パッド間距離 [m]。省略時は cube.size + clearance-m",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/day4_pre_contact_sweep.json"),
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


def get_controller_space(controller: Any) -> Any:
    if hasattr(controller, "single_action_space"):
        return controller.single_action_space
    return controller.action_space


def build_controller_slices(controllers: Mapping) -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for name, controller in controllers.items():
        dimension = int(np.prod(get_controller_space(controller).shape))
        result[str(name)] = slice(offset, offset + dimension)
        offset += dimension
    return result


def find_controller(controllers: Mapping, keyword: str) -> str:
    for name in controllers:
        if keyword in str(name).lower():
            return str(name)
    raise RuntimeError(
        f"{keyword} controllerが見つかりません: {list(controllers.keys())}"
    )


def get_controller_joint_names(controller: Any) -> list[str]:
    names = getattr(controller.config, "joint_names", None)
    if names is None:
        return []
    return [str(name) for name in names]


def compute_arm_hold_action(
    current_qpos: np.ndarray,
    arm_joint_indices: list[int],
    target_qpos: np.ndarray,
    max_delta_rad: float,
) -> np.ndarray:
    """delta-position controller用の、現在姿勢を維持する腕行動を作る。"""
    error = target_qpos - current_qpos[arm_joint_indices]
    delta = np.clip(error, -max_delta_rad, max_delta_rad)
    return np.clip(delta / max_delta_rad, -1.0, 1.0).astype(np.float32)


def action_values(
    action_min: float, action_max: float, action_step: float
) -> np.ndarray:
    if not (-1.0 <= action_min <= 1.0 and -1.0 <= action_max <= 1.0):
        raise ValueError("action-minとaction-maxは[-1, 1]内で指定してください")
    if action_min >= action_max:
        raise ValueError("action-minはaction-maxより小さくしてください")
    if action_step <= 0:
        raise ValueError("action-stepは正である必要があります")

    count = int(round((action_max - action_min) / action_step))
    if count < 1 or not np.isclose(
        action_min + count * action_step, action_max, atol=1e-8
    ):
        raise ValueError(
            "(action-max - action-min) がaction-stepで割り切れる値を指定してください"
        )
    return np.linspace(action_min, action_max, count + 1, dtype=np.float32)


def pad_center(links: list[Any]) -> np.ndarray:
    positions = np.stack(
        [np.asarray(first_env(link.pose.p), dtype=np.float64) for link in links],
        axis=0,
    )
    return positions.mean(axis=0)


def main() -> int:
    args = parse_args()

    if args.settle_steps <= 0:
        raise ValueError("settle-stepsは1以上で指定してください")
    if args.clearance_m <= 0:
        raise ValueError("clearance-mは正である必要があります")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    robot_uid = os.environ.get("UR3E_EZGRIPPER_UID") or config["robot"]["uid"]
    control_mode = config["project"]["control_mode"]
    max_arm_delta = float(config["residual"]["max_joint_delta_rad"])
    cube_size = float(config["cube"]["size"])
    target_gap = (
        float(args.target_gap_m)
        if args.target_gap_m is not None
        else cube_size + args.clearance_m
    )
    if target_gap <= cube_size:
        raise ValueError("target-gap-mはcube.sizeより大きくしてください")

    # @register_env と @register_agent を実行する。
    import envs.ur3e_pick_lift  # noqa: F401

    env = gym.make(
        args.env_id,
        robot_uids=robot_uid,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend=args.sim_backend,
        render_mode="human" if args.render else None,
    )

    try:
        env.reset(seed=int(config["project"]["seed"]))
        base_env = env.unwrapped
        robot = base_env.agent.robot
        controllers = base_env.agent.controller.controllers
        if not isinstance(controllers, Mapping):
            raise RuntimeError("CombinedControllerではありません")

        slices = build_controller_slices(controllers)
        arm_name = find_controller(controllers, "arm")
        gripper_name = find_controller(controllers, "gripper")
        arm_slice = slices[arm_name]
        gripper_slice = slices[gripper_name]
        action_dimension = int(env.action_space.shape[0])

        active_joint_names = [joint.name for joint in robot.active_joints]
        name_to_index = {
            name: index for index, name in enumerate(active_joint_names)
        }
        arm_joint_names = get_controller_joint_names(controllers[arm_name])
        gripper_joint_names = get_controller_joint_names(controllers[gripper_name])
        arm_joint_indices = [name_to_index[name] for name in arm_joint_names]
        gripper_joint_indices = [
            name_to_index[name] for name in gripper_joint_names
        ]

        records: list[dict[str, Any]] = []
        print("=" * 72)
        print("Day 4 pre-contact action sweep")
        print("=" * 72)
        print("target gap     :", f"{target_gap:.6f} m")
        print("cube size      :", f"{cube_size:.6f} m")
        print("settle steps   :", args.settle_steps)
        print("action range   :", f"[{args.action_min}, {args.action_max}]")

        for index, gripper_action in enumerate(
            action_values(args.action_min, args.action_max, args.action_step)
        ):
            # 各候補を同一初期状態から測る。
            env.reset(seed=int(config["project"]["seed"]))
            initial_qpos = np.asarray(first_env(robot.get_qpos()), dtype=np.float64)
            arm_hold_qpos = initial_qpos[arm_joint_indices].copy()

            for _ in range(args.settle_steps):
                current_qpos = np.asarray(
                    first_env(robot.get_qpos()), dtype=np.float64
                )
                action = np.zeros(action_dimension, dtype=np.float32)
                action[arm_slice] = compute_arm_hold_action(
                    current_qpos=current_qpos,
                    arm_joint_indices=arm_joint_indices,
                    target_qpos=arm_hold_qpos,
                    max_delta_rad=max_arm_delta,
                )
                action[gripper_slice] = float(gripper_action)
                env.step(action)
                if args.render:
                    env.render()
                    time.sleep(args.sleep)

            left_center = pad_center(base_env.left_contact_links)
            right_center = pad_center(base_env.right_contact_links)
            pad_axis = right_center - left_center
            gap = float(np.linalg.norm(pad_axis))
            final_qpos = np.asarray(first_env(robot.get_qpos()), dtype=np.float64)
            error = abs(gap - target_gap)

            record = {
                "action": float(gripper_action),
                "gripper_qpos_rad": final_qpos[gripper_joint_indices].tolist(),
                "left_pad_center_m": left_center.tolist(),
                "right_pad_center_m": right_center.tolist(),
                "pad_center_m": ((left_center + right_center) / 2.0).tolist(),
                "pad_gap_m": gap,
                "target_gap_error_m": error,
            }
            records.append(record)
            print(
                f"{index + 1:2d}/{len(action_values(args.action_min, args.action_max, args.action_step)):2d} "
                f"action={record['action']:+.3f} "
                f"gap={gap:.6f} m "
                f"error={error:.6f} m "
                f"q={record['gripper_qpos_rad']}"
            )

        best = min(records, key=lambda record: record["target_gap_error_m"])
        report = {
            "env_id": args.env_id,
            "robot_uid": robot_uid,
            "control_mode": control_mode,
            "sim_backend": args.sim_backend,
            "cube_size_m": cube_size,
            "clearance_m": args.clearance_m,
            "target_gap_m": target_gap,
            "settle_steps": args.settle_steps,
            "recommended_pre_contact_action": best["action"],
            "recommended_pad_gap_m": best["pad_gap_m"],
            "recommended_gap_error_m": best["target_gap_error_m"],
            "arm_joint_names": arm_joint_names,
            "gripper_joint_names": gripper_joint_names,
            "records": records,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        print("-" * 72)
        print(
            "recommended action:",
            f"{report['recommended_pre_contact_action']:+.6f}",
        )
        print(
            "recommended gap   :",
            f"{report['recommended_pad_gap_m']:.6f} m",
        )
        print(
            "gap error         :",
            f"{report['recommended_gap_error_m']:.6f} m",
        )
        print("report            :", args.output)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
