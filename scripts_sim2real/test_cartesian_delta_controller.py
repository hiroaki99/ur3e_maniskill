#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import gymnasium as gym
import mani_skill.envs  # noqa: F401

import envs.ur3e_reach  # noqa: F401


from scripts_sim2real.maniskill_backend import (
    ManiSkillBackend,
)

from scripts_sim2real.ur3e_ros2_backend import (
    UR3eROS2Backend,
)

from scripts_sim2real.cartesian_delta_controller import (
    CartesianDeltaController,
    JOINT_NAMES,
)


def main():

    real_backend = (
        UR3eROS2Backend(
            timeout=15.0
        )
    )

    print(
        "=== Bridge ==="
    )

    print(
        real_backend.ping()
    )


    real_backend.wait_until_ready(
        timeout_sec=10.0
    )


    env = gym.make(
        "UR3eReach-v0",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )

    env.reset()


    sim_backend = (
        ManiSkillBackend(env)
    )


    controller = (
        CartesianDeltaController(
            kinematics_backend=(
                sim_backend
            ),
            robot_backend=(
                real_backend
            ),
            command_duration_sec=0.5,
        )
    )


    # --------------------------------------------------------
    # dry-run
    # --------------------------------------------------------

    result = (
        controller.command_tcp_delta(
            np.array(
                [
                    0.001,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
            execute=False,
        )
    )


    plan = (
        result["plan"]
    )


    print(
        "\n=== Cartesian Delta Plan ==="
    )

    print(
        "requested delta [m]:",
        plan.requested_delta_m.tolist(),
    )

    print(
        "predicted delta [m]:",
        plan.predicted_delta_m.tolist(),
    )

    print(
        "predicted error [mm]:",
        plan.predicted_position_error_m
        * 1000.0,
    )

    print(
        "orientation change [deg]:",
        plan.predicted_orientation_change_deg,
    )

    print(
        "condition number:",
        plan.condition_number,
    )


    print(
        "\n=== dq ==="
    )

    for name, dq in zip(
        JOINT_NAMES,
        plan.dq,
    ):
        print(
            f"{name}: "
            f"{dq:+.9f} rad"
        )


    print(
        "\n=== q target ==="
    )

    for name in JOINT_NAMES:
        print(
            f"{name}: "
            f"{plan.q_target[name]:+.9f} rad"
        )


    print(
        "\nDRY RUN COMPLETE."
    )

    print(
        "The real robot was NOT moved."
    )


    sim_backend.close()
    real_backend.close()


if __name__ == "__main__":
    main()