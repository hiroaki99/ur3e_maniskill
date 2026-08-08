#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from mani_skill.utils.structs.pose import Pose


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def first_env(value):
    a = to_numpy(value)

    if a.ndim >= 2 and a.shape[0] == 1:
        return a[0]

    return a


def main():

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    import envs.ur3e_pick_lift  # noqa

    env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend="physx_cpu",
        render_mode="human",
    )

    try:

        env.reset(seed=0)

        base_env = env.unwrapped

        # --------------------------------------------------
        # 現在コードで求めている中心
        # --------------------------------------------------

        left_positions = np.stack(
            [
                first_env(link.pose.p)
                for link
                in base_env.left_contact_links
            ]
        )

        right_positions = np.stack(
            [
                first_env(link.pose.p)
                for link
                in base_env.right_contact_links
            ]
        )

        left_center = left_positions.mean(axis=0)
        right_center = right_positions.mean(axis=0)

        calculated_center = (
            left_center + right_center
        ) / 2.0

        print(
            "calculated center:",
            calculated_center,
        )

        # --------------------------------------------------
        # ±15 cmを1 cm刻みで探索
        #
        # まずX方向だけ。
        # --------------------------------------------------

        offsets = np.arange(
            -0.15,
            0.151,
            0.01,
        )

        zero_velocity = torch.zeros(
            (1, 3),
            dtype=torch.float32,
            device=base_env.device,
        )

        for dx in offsets:

            target = (
                calculated_center
                + np.array(
                    [dx, 0.0, 0.0],
                    dtype=np.float32,
                )
            )

            p = torch.tensor(
                target,
                dtype=torch.float32,
                device=base_env.device,
            ).unsqueeze(0)

            q = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=base_env.device,
            )

            base_env.cube.set_pose(
                Pose.create_from_pq(
                    p=p,
                    q=q,
                )
            )

            base_env.cube.set_linear_velocity(
                zero_velocity
            )

            base_env.cube.set_angular_velocity(
                zero_velocity
            )

            # 数step進めてcontactを更新
            for _ in range(3):

                action = np.zeros(
                    env.action_space.shape,
                    dtype=np.float32,
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                env.render()
                time.sleep(0.015)

            left_force = float(
                first_env(
                    info["left_contact_force"]
                )
            )

            right_force = float(
                first_env(
                    info["right_contact_force"]
                )
            )

            print(
                f"dx={dx:+.3f} "
                f"position={target} "
                f"L={left_force:.5f} N "
                f"R={right_force:.5f} N"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()