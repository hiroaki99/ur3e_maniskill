#!/usr/bin/env python3

import math
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


TARGET_JOINT = "shoulder_pan_joint"

DELTA_DEG = 20.0
DELTA_RAD = math.radians(DELTA_DEG)

DURATION_SEC = 4.0

RETURN_TOLERANCE_RAD = 0.002


def print_joints(title, joints):
    print(f"\n=== {title} ===")

    for name, value in joints.items():
        print(
            f"{name}: {value:.9f} rad"
        )


def main():

    backend = UR3eROS2Backend(
        timeout=20.0,
    )

    print("=== Bridge ===")
    print(backend.ping())


    # ========================================================
    # 1. 開始姿勢
    # ========================================================

    original = (
        backend.get_joint_positions()
    )

    original_tcp = (
        backend.get_tcp_pose()
    )

    print_joints(
        "Original Joint Positions",
        original,
    )

    print(
        "\n=== Original TCP ==="
    )
    print(original_tcp)


    # ========================================================
    # 2. shoulder_pan +20 deg
    # ========================================================

    forward_target = original.copy()

    forward_target[TARGET_JOINT] += (
        DELTA_RAD
    )

    print(
        "\n=== Requested Forward Motion ==="
    )

    print(
        f"joint: {TARGET_JOINT}"
    )

    print(
        f"delta: +{DELTA_DEG:.1f} deg"
    )

    print(
        f"delta: +{DELTA_RAD:.9f} rad"
    )

    print(
        f"before: "
        f"{original[TARGET_JOINT]:.9f} rad"
    )

    print(
        f"target: "
        f"{forward_target[TARGET_JOINT]:.9f} rad"
    )


    # ========================================================
    # 3. 前進
    # ========================================================

    print(
        "\n=== Forward Command ==="
    )

    forward_result = (
        backend.command_joint_positions(
            forward_target,
            duration_sec=DURATION_SEC,
        )
    )

    print(forward_result)

    time.sleep(0.3)

    forward_actual = (
        backend.get_joint_positions()
    )

    forward_tcp = (
        backend.get_tcp_pose()
    )

    print_joints(
        "After Forward Command",
        forward_actual,
    )

    print(
        "\n=== TCP After Forward Command ==="
    )
    print(forward_tcp)


    # ========================================================
    # 4. 元姿勢へ復帰
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

    time.sleep(0.3)

    final = (
        backend.get_joint_positions()
    )

    final_tcp = (
        backend.get_tcp_pose()
    )


    # ========================================================
    # 5. 結果
    # ========================================================

    print_joints(
        "Final Joint Positions",
        final,
    )

    actual_forward_delta = (
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
        f"{DELTA_RAD:.9f} rad "
        f"({DELTA_DEG:.1f} deg)"
    )

    print(
        f"actual forward delta: "
        f"{actual_forward_delta:.9f} rad"
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

    print(
        "\n=== Final TCP ==="
    )
    print(final_tcp)

    backend.close()


if __name__ == "__main__":
    main()