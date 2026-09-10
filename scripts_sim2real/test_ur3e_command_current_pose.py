#!/usr/bin/env python3

import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts_sim2real.ur3e_ros2_backend import (
    UR3eROS2Backend,
)


def main():

    backend = UR3eROS2Backend(
        timeout=15.0,
    )


    print("=== Bridge ===")
    print(backend.ping())


    # --------------------------------------------
    # 現在姿勢取得
    # --------------------------------------------

    current = (
        backend.get_joint_positions()
    )

    print(
        "\n=== Current Joint Positions ==="
    )

    for name, value in current.items():
        print(
            f"{name}: {value:.9f}"
        )


    # --------------------------------------------
    # 現在姿勢をそのまま送信
    #
    # 意図的な移動は発生させない。
    # --------------------------------------------

    print(
        "\n=== Command Current Pose ==="
    )

    result = (
        backend.command_joint_positions(
            current,
            duration_sec=2.0,
        )
    )

    print(result)


    # --------------------------------------------
    # 実行後姿勢
    # --------------------------------------------

    after = (
        backend.get_joint_positions()
    )

    print(
        "\n=== Joint Positions After Command ==="
    )

    for name, value in after.items():
        print(
            f"{name}: {value:.9f}"
        )


    backend.close()


if __name__ == "__main__":
    main()