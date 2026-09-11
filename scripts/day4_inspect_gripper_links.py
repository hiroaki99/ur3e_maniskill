#!/usr/bin/env python3
"""
Day 4:
UR3e + EZGripper のリンク一覧を取得し、
キューブとの接触判定に使う左右指リンク候補を調査する。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
        "--output",
        type=Path,
        default=Path(
            "reports/day4_gripper_links.json"
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


def is_gripper_candidate(name: str) -> bool:
    """
    グリッパに関係しそうなリンクを簡易抽出する。
    あくまで候補表示用。
    """

    lower = name.lower()

    keywords = [
        "gripper",
        "finger",
        "knuckle",
        "pad",
        "tip",
        "ezgripper",
    ]

    return any(
        keyword in lower
        for keyword in keywords
    )


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
            "robot.uidまたは--robot-uidを指定してください"
        )

    control_mode = (
        args.control_mode
        or config["project"]["control_mode"]
    )

    # @register_env を実行
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

        env.reset(
            seed=int(
                config["project"]["seed"]
            )
        )

        base_env = env.unwrapped
        robot = base_env.agent.robot

        # ManiSkill articulation のリンクMap
        links_map = robot.links_map

        results = []
        candidates = []

        print("=" * 88)
        print("Day 4 - UR3e + EZGripper Link Inspection")
        print("=" * 88)

        for index, (name, link) in enumerate(
            links_map.items()
        ):

            position = first_env(
                link.pose.p
            ).astype(float)

            candidate = is_gripper_candidate(
                name
            )

            item = {
                "index": index,
                "name": name,
                "position_m": (
                    position.tolist()
                ),
                "gripper_candidate": (
                    candidate
                ),
            }

            results.append(item)

            if candidate:
                candidates.append(item)

            mark = "*" if candidate else " "

            print(
                f"{mark} "
                f"{index:2d} "
                f"{name:55s} "
                f"p={position.tolist()}"
            )

        print()
        print(
            "* = グリッパ接触リンク候補"
        )

        print()
        print("=" * 88)
        print("Gripper candidates")
        print("=" * 88)

        for item in candidates:

            print(
                f"{item['name']:55s} "
                f"{item['position_m']}"
            )

        report = {
            "env_id": args.env_id,
            "robot_uid": robot_uid,
            "control_mode": control_mode,
            "all_links": results,
            "gripper_candidates": candidates,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True,)

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
        print(
            "report:",
            args.output,
        )

        return 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())