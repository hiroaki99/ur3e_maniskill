#!/usr/bin/env python3

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts_sim2real.ur3e_ros2_backend import (
    UR3eROS2Backend,
)


TARGET_JOINT = "wrist_3_joint"

# 0.01 rad ≈ 0.57 deg
DELTA_RAD = 0.01

DURATION_SEC = 2.0


def print_joints(title, joints):
    print(f"\n=== {title} ===")

    for name, value in joints.items():
        print(
            f"{name}: {value:.9f} rad"
        )


def main():

    backend = UR3eROS2Backend(
        timeout=15.0,
    )

    print("=== Bridge ===")
    print(backend.ping())


    # ========================================================
    # 1. 現在姿勢取得
    # ========================================================

    current = (
        backend.get_joint_positions()
    )

    print_joints(
        "Current Joint Positions",
        current,
    )


    # ========================================================
    # 2. +0.01 rad の目標姿勢を作る
    # ========================================================

    target = current.copy()

    target[TARGET_JOINT] += DELTA_RAD


    print(
        "\n=== Requested Motion ==="
    )

    print(
        f"joint: {TARGET_JOINT}"
    )

    print(
        f"delta: {DELTA_RAD:.6f} rad"
    )

    print(
        f"before: "
        f"{current[TARGET_JOINT]:.9f}"
    )

    print(
        f"target: "
        f"{target[TARGET_JOINT]:.9f}"
    )


    # ========================================================
    # 3. 微小関節指令
    # ========================================================

    print(
        "\n=== Command Small Delta ==="
    )

    result = (
        backend.command_joint_positions(
            target,
            duration_sec=DURATION_SEC,
        )
    )

    print(result)


    # 最新状態待ち
    time.sleep(0.2)


    # ========================================================
    # 4. 実行後確認
    # ========================================================

    after = (
        backend.get_joint_positions()
    )

    print_joints(
        "Joint Positions After Command",
        after,
    )


    actual_delta = (
        after[TARGET_JOINT]
        - current[TARGET_JOINT]
    )

    target_error = (
        after[TARGET_JOINT]
        - target[TARGET_JOINT]
    )


    print(
        "\n=== Motion Result ==="
    )

    print(
        f"requested delta: "
        f"{DELTA_RAD:.9f} rad"
    )

    print(
        f"actual delta: "
        f"{actual_delta:.9f} rad"
    )

    print(
        f"target error: "
        f"{target_error:.9f} rad"
    )


    # ========================================================
    # 5. TCPも確認
    # ========================================================

    tcp_pose = (
        backend.get_tcp_pose()
    )

    print(
        "\n=== TCP Pose After Command ==="
    )

    print(tcp_pose)


    backend.close()


if __name__ == "__main__":
    main()