#!/usr/bin/env python3

import sys
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


def main():
    backend = UR3eROS2Backend()

    print("=== Bridge Ping ===")
    print(
        backend.ping()
    )

    print("\n=== Joint Positions ===")
    joint_positions = (
        backend.get_joint_positions()
    )

    for name, value in joint_positions.items():
        print(
            f"{name}: {value:.9f} rad"
        )

    print("\n=== TCP Pose ===")
    tcp_pose = backend.get_tcp_pose()

    print(
        f"frame_id: {tcp_pose.frame_id}"
    )

    print(
        f"position: "
        f"{tcp_pose.position}"
    )

    print(
        f"orientation_xyzw: "
        f"{tcp_pose.orientation_xyzw}"
    )

    backend.close()


if __name__ == "__main__":
    main()