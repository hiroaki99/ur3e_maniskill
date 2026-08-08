#!/usr/bin/env python3
"""Day 3: UR3eの腕を静止させ、EZGripperを繰り返し開閉する。

確認内容:
- グリッパcontrollerだけへ指令を送る
- 腕controllerへは0を送る
- 開閉を指定回数完走する
- NaN/infを検出する
- グリッパの移動量を記録する
- 腕関節のドリフトを記録する
- 開位置・閉位置の再現性を記録する
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
        "--gripper-controller",
        type=str,
        default=None,
        help=(
            "gripper sub-controller名。"
            "省略時はgripper/finger/handを含む名前を探索する"
        ),
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--phase-steps",
        type=int,
        default=20,
        help="開または閉の指令を維持するstep数",
    )

    parser.add_argument(
        "--open-action",
        type=float,
        default=-1.0,
    )

    parser.add_argument(
        "--close-action",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--arm-drift-limit",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--minimum-gripper-motion",
        type=float,
        default=1e-4,
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
        default=Path("reports/day3_gripper_cycles.json"),
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

    raise AttributeError(
        "robotからactive jointを取得できません"
    )


def get_controller_space(controller: Any) -> Any:
    if hasattr(controller, "single_action_space"):
        return controller.single_action_space

    if hasattr(controller, "action_space"):
        return controller.action_space

    raise AttributeError(
        f"{type(controller).__name__}から"
        "action spaceを取得できません"
    )


def get_controller_joint_names(controller: Any) -> list[str]:
    config = getattr(controller, "config", None)

    if config is not None:
        names = getattr(config, "joint_names", None)

        if names is not None:
            return [str(name) for name in names]

    joints = getattr(controller, "joints", None)

    if joints is not None:
        return [
            str(getattr(joint, "name", joint))
            for joint in joints
        ]

    return []


def find_gripper_controller_name(
    controllers: Mapping[str, Any],
    requested_name: str | None,
) -> str:
    if requested_name is not None:
        if requested_name not in controllers:
            raise KeyError(
                f"指定controllerがありません: {requested_name}. "
                f"候補={list(controllers.keys())}"
            )

        return requested_name

    for name in controllers:
        lower_name = str(name).lower()

        if (
            "gripper" in lower_name
            or "finger" in lower_name
            or "hand" in lower_name
        ):
            return str(name)

    raise KeyError(
        "グリッパcontrollerを自動検出できません。"
        f"候補={list(controllers.keys())}。"
        "--gripper-controllerで指定してください。"
    )


def build_controller_slices(
    controllers: Mapping[str, Any],
) -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0

    for name, controller in controllers.items():
        space = get_controller_space(controller)
        action_dim = int(np.prod(space.shape))

        result[str(name)] = slice(
            offset,
            offset + action_dim,
        )

        offset += action_dim

    return result


def run_phase(
    env,
    robot,
    action,
    steps,
    render,
    sleep_seconds,
    arm_slice,
    arm_joint_indices,
    arm_hold_qpos,
    max_arm_delta_rad,
):
    finite = True

    for _ in range(steps):

        current_qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        step_action = action.copy()

        # UR3e姿勢保持

        arm_hold_action = (
            compute_arm_hold_action(
                current_qpos=current_qpos,
                arm_joint_indices=(
                    arm_joint_indices
                ),
                arm_hold_qpos=(
                    arm_hold_qpos
                ),
                max_delta_rad=(
                    max_arm_delta_rad
                ),
            )
        )

        step_action[arm_slice] = (
            arm_hold_action
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(step_action)

        qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        qvel = first_env(
            robot.get_qvel()
        ).astype(float)

        if (
            not np.all(np.isfinite(qpos))
            or not np.all(np.isfinite(qvel))
        ):
            finite = False
            break

        if render:
            env.render()
            time.sleep(
                sleep_seconds
            )

    return qpos, qvel, finite

def compute_arm_hold_action(
    current_qpos: np.ndarray,
    arm_joint_indices: list[int],
    arm_hold_qpos: np.ndarray,
    max_delta_rad: float = 0.03,
) -> np.ndarray:
    """pd_joint_delta_pos用の姿勢保持actionを生成する。

    物理的な関節誤差を[-1, 1]の正規化actionへ変換する。
    """

    current_arm_qpos = current_qpos[
        arm_joint_indices
    ]

    error = (
        arm_hold_qpos
        - current_arm_qpos
    )

    # 1 stepで要求する補正量を制限
    delta = np.clip(
        error,
        -max_delta_rad,
        max_delta_rad,
    )

    # controller:
    # physical range [-0.03, +0.03]
    # normalized action [-1, +1]
    normalized_action = (
        delta / max_delta_rad
    )

    return np.clip(
        normalized_action,
        -1.0,
        1.0,
    ).astype(np.float32)


def main() -> int:
    args = parse_args()

    if args.cycles <= 0:
        raise ValueError("--cyclesは1以上にしてください")

    if args.phase_steps <= 0:
        raise ValueError(
            "--phase-stepsは1以上にしてください"
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
            "robot.uidまたは--robot-uidを指定してください"
        )

    control_mode = (
        args.control_mode
        or config["project"]["control_mode"]
    )

    import envs.ur3e_pick_lift  # noqa: F401

    render_mode = "human" if args.render else None

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
        agent = base_env.agent
        robot = agent.robot
        combined_controller = agent.controller

        controllers = getattr(
            combined_controller,
            "controllers",
            None,
        )

        if not isinstance(controllers, Mapping):
            raise RuntimeError(
                "CombinedControllerのsub-controllerを"
                "取得できません"
            )

        controller_slices = build_controller_slices(
            controllers
        )

        gripper_controller_name = (
            find_gripper_controller_name(
                controllers,
                args.gripper_controller,
            )
        )

        arm_controller_name = None

        for name in controllers:
            if "arm" in name.lower():
                arm_controller_name = name
                break

        if arm_controller_name is None:
            raise RuntimeError(
                "arm controllerが見つかりません"
            )

        arm_slice = controller_slices[
            arm_controller_name
        ]

        gripper_slice = controller_slices[
            gripper_controller_name
        ]

        gripper_controller = controllers[
            gripper_controller_name
        ]

        gripper_joint_names = (
            get_controller_joint_names(
                gripper_controller
            )
        )

        active_joints = get_active_joints(robot)
        active_joint_names = [
            joint.name for joint in active_joints
        ]

        joint_name_to_index = {
            name: index
            for index, name in enumerate(
                active_joint_names
            )
        }

        gripper_joint_indices = [
            joint_name_to_index[name]
            for name in gripper_joint_names
            if name in joint_name_to_index
        ]

        arm_joint_indices = [
            index
            for index in range(
                len(active_joint_names)
            )
            if index not in gripper_joint_indices
        ]

        action_shape = env.action_space.shape

        if len(action_shape) != 1:
            raise ValueError(
                f"1次元Box actionを想定していますが、"
                f"shape={action_shape}です"
            )

        total_action_dim = action_shape[0]

        initial_qpos = first_env(
            robot.get_qpos()
        ).astype(float)

        arm_hold_qpos = initial_qpos[
            arm_joint_indices
        ].copy()

        open_records: list[np.ndarray] = []
        close_records: list[np.ndarray] = []
        max_qvel_records: list[float] = []

        errors: list[str] = []
        warnings: list[str] = []

        print("=" * 72)
        print("Day 3 EZGripper cycle test")
        print("=" * 72)
        print("cycles              :", args.cycles)
        print("phase steps         :", args.phase_steps)
        print("gripper controller  :", gripper_controller_name)
        print("gripper slice       :", [
            gripper_slice.start,
            gripper_slice.stop,
        ])
        print("gripper joints      :", gripper_joint_names)
        print("gripper indices     :", gripper_joint_indices)
        print("open action         :", args.open_action)
        print("close action        :", args.close_action)

        finite = True

        for cycle in range(args.cycles):
            # 全controllerに対してゼロ指令を作る。
            # 腕がdelta controllerなら0で現在姿勢を維持する。
            open_action = np.zeros(
                total_action_dim,
                dtype=np.float32,
            )

            open_action[gripper_slice] = (
                args.open_action
            )

            open_qpos, open_qvel, phase_finite = (
                run_phase(
                    env=env,
                    robot=robot,
                    action=open_action,
                    steps=args.phase_steps,
                    render=args.render,
                    sleep_seconds=args.sleep,
                    arm_slice=arm_slice,
                    arm_joint_indices=arm_joint_indices,
                    arm_hold_qpos=arm_hold_qpos,
                    max_arm_delta_rad=0.03,
                )
            )

            finite = finite and phase_finite

            if not finite:
                errors.append(
                    f"cycle {cycle + 1}のopen phaseで"
                    "NaNまたはinfを検出しました"
                )
                break

            close_action = np.zeros(
                total_action_dim,
                dtype=np.float32,
            )

            close_action[gripper_slice] = (
                args.close_action
            )

            close_qpos, close_qvel, phase_finite = (
                run_phase(
                    env=env,
                    robot=robot,
                    action=close_action,
                    steps=args.phase_steps,
                    render=args.render,
                    sleep_seconds=args.sleep,
                    arm_slice=arm_slice,
                    arm_joint_indices=arm_joint_indices,
                    arm_hold_qpos=arm_hold_qpos,
                    max_arm_delta_rad=0.03,
                )
            )

            finite = finite and phase_finite

            if not finite:
                errors.append(
                    f"cycle {cycle + 1}のclose phaseで"
                    "NaNまたはinfを検出しました"
                )
                break

            open_records.append(open_qpos.copy())
            close_records.append(close_qpos.copy())

            max_qvel_records.append(
                float(
                    max(
                        np.abs(open_qvel).max(
                            initial=0.0
                        ),
                        np.abs(close_qvel).max(
                            initial=0.0
                        ),
                    )
                )
            )

            if (
                cycle == 0
                or (cycle + 1) % 10 == 0
                or cycle + 1 == args.cycles
            ):
                print(
                    f"cycle={cycle + 1:3d}/{args.cycles}, "
                    f"max|qvel|="
                    f"{max_qvel_records[-1]:.6f}"
                )

        completed_cycles = len(open_records)

        if completed_cycles == 0:
            errors.append(
                "1サイクルも完了していません"
            )

            open_array = np.empty(
                (0, len(initial_qpos))
            )
            close_array = np.empty(
                (0, len(initial_qpos))
            )
        else:
            open_array = np.stack(
                open_records,
                axis=0,
            )

            close_array = np.stack(
                close_records,
                axis=0,
            )

        if completed_cycles > 0:
            open_mean = open_array.mean(axis=0)
            close_mean = close_array.mean(axis=0)

            open_std = open_array.std(axis=0)
            close_std = close_array.std(axis=0)

            qpos_motion = np.abs(
                open_mean - close_mean
            )

            final_qpos = close_array[-1]

            if arm_joint_indices:
                arm_drift = np.abs(
                    final_qpos[arm_joint_indices]
                    - initial_qpos[arm_joint_indices]
                )

                max_arm_drift = float(
                    arm_drift.max(initial=0.0)
                )
            else:
                max_arm_drift = 0.0

            if gripper_joint_indices:
                gripper_motion = qpos_motion[
                    gripper_joint_indices
                ]

                max_gripper_motion = float(
                    gripper_motion.max(initial=0.0)
                )

                max_open_std = float(
                    open_std[
                        gripper_joint_indices
                    ].max(initial=0.0)
                )

                max_close_std = float(
                    close_std[
                        gripper_joint_indices
                    ].max(initial=0.0)
                )
            else:
                # Controllerのjoint名を取得できなかった場合、
                # 全active jointのopen-close差を参考値にする
                max_gripper_motion = float(
                    qpos_motion.max(initial=0.0)
                )

                max_open_std = float(
                    open_std.max(initial=0.0)
                )

                max_close_std = float(
                    close_std.max(initial=0.0)
                )

                warnings.append(
                    "グリッパ関節indexを特定できませんでした。"
                    "全関節の変化量を代用しています。"
                )

            if (
                max_gripper_motion
                < args.minimum_gripper_motion
            ):
                errors.append(
                    "openとcloseで十分な関節角変化がありません: "
                    f"{max_gripper_motion:.6e} rad"
                )

            if max_arm_drift > args.arm_drift_limit:
                warnings.append(
                    "腕の関節ドリフトが暫定上限を超えました: "
                    f"{max_arm_drift:.6f} rad > "
                    f"{args.arm_drift_limit:.6f} rad"
                )

            if max_open_std > 0.005:
                warnings.append(
                    "open位置のサイクル間標準偏差が"
                    "0.005 radを超えました: "
                    f"{max_open_std:.6f} rad"
                )

            if max_close_std > 0.005:
                warnings.append(
                    "close位置のサイクル間標準偏差が"
                    "0.005 radを超えました: "
                    f"{max_close_std:.6f} rad"
                )

        else:
            open_mean = np.full_like(
                initial_qpos,
                np.nan,
            )

            close_mean = np.full_like(
                initial_qpos,
                np.nan,
            )

            open_std = np.full_like(
                initial_qpos,
                np.nan,
            )

            close_std = np.full_like(
                initial_qpos,
                np.nan,
            )

            max_arm_drift = float("nan")
            max_gripper_motion = float("nan")
            max_open_std = float("nan")
            max_close_std = float("nan")

        report = {
            "env_id": args.env_id,
            "robot_uid": robot_uid,
            "control_mode": control_mode,
            "sim_backend": args.sim_backend,
            "requested_cycles": args.cycles,
            "completed_cycles": completed_cycles,
            "phase_steps": args.phase_steps,
            "open_action": args.open_action,
            "close_action": args.close_action,
            "active_joint_names": active_joint_names,
            "gripper_controller": (
                gripper_controller_name
            ),
            "gripper_action_slice": [
                gripper_slice.start,
                gripper_slice.stop,
            ],
            "gripper_joint_names": (
                gripper_joint_names
            ),
            "gripper_joint_indices": (
                gripper_joint_indices
            ),
            "arm_joint_indices": (
                arm_joint_indices
            ),
            "initial_qpos": (
                initial_qpos.tolist()
            ),
            "mean_open_qpos": (
                open_mean.tolist()
            ),
            "mean_close_qpos": (
                close_mean.tolist()
            ),
            "std_open_qpos": (
                open_std.tolist()
            ),
            "std_close_qpos": (
                close_std.tolist()
            ),
            "max_gripper_motion_rad": (
                max_gripper_motion
            ),
            "max_arm_drift_rad": (
                max_arm_drift
            ),
            "max_open_std_rad": (
                max_open_std
            ),
            "max_close_std_rad": (
                max_close_std
            ),
            "max_observed_qvel_rad_s": (
                max(max_qvel_records)
                if max_qvel_records
                else None
            ),
            "finite": finite,
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
            "completed cycles     :",
            completed_cycles,
        )
        print(
            "max gripper motion   :",
            f"{max_gripper_motion:.6f} rad",
        )
        print(
            "max arm drift        :",
            f"{max_arm_drift:.6f} rad",
        )
        print(
            "max open std         :",
            f"{max_open_std:.6f} rad",
        )
        print(
            "max close std        :",
            f"{max_close_std:.6f} rad",
        )

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