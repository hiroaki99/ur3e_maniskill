#!/usr/bin/env python3

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts_sim2real.ur3e_ros2_backend import (
    UR3eROS2Backend,
)


TARGET_JOINT = "wrist_3_joint"

DELTA_RAD = 0.4
DURATION_SEC = 2.0

# 往復後の許容誤差
RETURN_TOLERANCE_RAD = 0.001


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
    # 1. 往復試験開始時の姿勢
    # ========================================================

    original = (
        backend.get_joint_positions()
    )

    print_joints(
        "Original Joint Positions",
        original,
    )


    # ========================================================
    # 2. +0.01 rad
    # ========================================================

    forward_target = original.copy()

    forward_target[TARGET_JOINT] += (
        DELTA_RAD
    )

    print(
        "\n=== Forward Command ==="
    )

    print(
        f"{TARGET_JOINT}: "
        f"{original[TARGET_JOINT]:.9f}"
        " -> "
        f"{forward_target[TARGET_JOINT]:.9f}"
    )

    forward_result = (
        backend.command_joint_positions(
            forward_target,
            duration_sec=DURATION_SEC,
        )
    )

    print(forward_result)

    time.sleep(0.2)

    forward_actual = (
        backend.get_joint_positions()
    )

    print_joints(
        "After Forward Command",
        forward_actual,
    )


    # ========================================================
    # 3. 元の姿勢へ戻す
    #
    # -0.01を現在値から計算するのではなく、
    # 最初に保存したoriginalを目標にする。
    # ========================================================

    print(
        "\n=== Return Command ==="
    )

    return_result = (
        backend.command_joint_positions(
            original,
            duration_sec=DURATION_SEC,
        )
    )

    print(return_result)

    time.sleep(0.2)


    # ========================================================
    # 4. 往復後
    # ========================================================

    final = (
        backend.get_joint_positions()
    )

    print_joints(
        "Final Joint Positions",
        final,
    )


    # ========================================================
    # 5. 評価
    # ========================================================

    forward_delta = (
        forward_actual[TARGET_JOINT]
        - original[TARGET_JOINT]
    )

    return_error = (
        final[TARGET_JOINT]
        - original[TARGET_JOINT]
    )


    print(
        "\n=== Round Trip Result ==="
    )

    print(
        f"requested forward delta: "
        f"{DELTA_RAD:.9f} rad"
    )

    print(
        f"actual forward delta: "
        f"{forward_delta:.9f} rad"
    )

    print(
        f"return error: "
        f"{return_error:.9f} rad"
    )


    passed = (
        abs(return_error)
        < RETURN_TOLERANCE_RAD
    )

    print(
        "round trip:",
        "PASS" if passed else "FAIL",
    )


    tcp_pose = (
        backend.get_tcp_pose()
    )

    print(
        "\n=== Final TCP Pose ==="
    )

    print(tcp_pose)

    backend.close()


if __name__ == "__main__":
    main()