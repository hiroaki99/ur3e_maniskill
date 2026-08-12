#!/usr/bin/env python3
"""
Small task perturbations for UR3e Residual RL.

Day12ではReference trajectoryそのものは変更せず、
Cubeの初期XY位置だけを数mmずらす。

これにより

    nominal reference
        +
    small environment mismatch

を作り、Residual RLが補正すべき課題を構成する。

Zは変更しない。
Cubeは常にTable上へ配置する。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.structs.pose import Pose


def to_numpy(
    value: Any,
) -> np.ndarray:

    if hasattr(
        value,
        "detach",
    ):
        value = value.detach()

    if hasattr(
        value,
        "cpu",
    ):
        value = value.cpu()

    if hasattr(
        value,
        "numpy",
    ):
        value = value.numpy()

    return np.asarray(
        value
    )


def first_env(
    value: Any,
) -> np.ndarray:

    array = to_numpy(
        value
    )

    if (
        array.ndim >= 2
        and array.shape[0] == 1
    ):
        return array[0]

    return array


class CubePositionPerturbationWrapper(
    gym.Wrapper
):
    """
    Cube初期位置をXY方向へ微小変更するWrapper。

    Parameters
    ----------
    radius_min_m:
        最小移動距離[m]

    radius_max_m:
        最大移動距離[m]

    direction_mode:
        random_angle
        x_positive
        x_negative
        y_positive
        y_negative

    Day12 calibrationでは

        radius_min_m == radius_max_m

    として固定半径を使う。

    Day13では

        0.005 <= radius <= 0.010

    のような分布へ拡張できる。
    """

    VALID_DIRECTION_MODES = {
        "random_angle",
        "x_positive",
        "x_negative",
        "y_positive",
        "y_negative",
    }

    def __init__(
        self,
        env: gym.Env,
        *,
        radius_min_m: float,
        radius_max_m: float,
        direction_mode: str = (
            "random_angle"
        ),
        fixed_angle_rad: float | None = None,
        seed: int = 0,
    ):

        super().__init__(
            env
        )

        self.radius_min_m = float(
            radius_min_m
        )

        self.radius_max_m = float(
            radius_max_m
        )

        if (
            self.radius_min_m < 0.0
        ):

            raise ValueError(
                "radius_min_m must be >= 0"
            )

        if (
            self.radius_max_m
            < self.radius_min_m
        ):

            raise ValueError(
                "radius_max_m must be "
                ">= radius_min_m"
            )

        self.direction_mode = str(
            direction_mode
        )

        if (
            self.direction_mode
            not in self.VALID_DIRECTION_MODES
        ):

            raise ValueError(
                "Unknown direction_mode: "
                f"{self.direction_mode}"
            )

        self.fixed_angle_rad = (
                    None
                    if fixed_angle_rad is None
                    else float(fixed_angle_rad)
                    )

        self.seed_value = int(
            seed
        )

        self.rng = (
            np.random.default_rng(
                self.seed_value
            )
        )

        self.last_offset = np.zeros(
            3,
            dtype=np.float64,
        )

        self.last_nominal_position = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

        self.last_actual_position = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

    # ======================================================
    # Sampling
    # ======================================================

    def _sample_radius(
        self,
    ) -> float:

        if np.isclose(
            self.radius_min_m,
            self.radius_max_m,
        ):

            return float(
                self.radius_min_m
            )

        return float(
            self.rng.uniform(
                self.radius_min_m,
                self.radius_max_m,
            )
        )

    def _sample_offset(
        self,
    ) -> np.ndarray:

        radius = (
            self._sample_radius()
        )

        if radius == 0.0:

            return np.zeros(
                3,
                dtype=np.float64,
            )

        if (
            self.direction_mode
            == "random_angle"
        ):

            if self.fixed_angle_rad is not None:

                theta = float(
                    self.fixed_angle_rad
                )

            else:

                theta = float(
                    self.rng.uniform(
                        0.0,
                        2.0 * np.pi,
                    )
                )

            return np.asarray(
                [
                    radius
                    * np.cos(
                        theta
                    ),

                    radius
                    * np.sin(
                        theta
                    ),

                    0.0,
                ],
                dtype=np.float64,
            )

        if (
            self.direction_mode
            == "x_positive"
        ):

            return np.asarray(
                [
                    radius,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        if (
            self.direction_mode
            == "x_negative"
        ):

            return np.asarray(
                [
                    -radius,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        if (
            self.direction_mode
            == "y_positive"
        ):

            return np.asarray(
                [
                    0.0,
                    radius,
                    0.0,
                ],
                dtype=np.float64,
            )

        if (
            self.direction_mode
            == "y_negative"
        ):

            return np.asarray(
                [
                    0.0,
                    -radius,
                    0.0,
                ],
                dtype=np.float64,
            )

        raise RuntimeError(
            self.direction_mode
        )

    # ======================================================
    # Gym API
    # ======================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        (
            observation,
            info,
        ) = self.env.reset(
            seed=seed,
            options=options,
        )

        # episode seedごとに
        # perturbation方向を再現可能にする
        if seed is not None:

            self.rng = (
                np.random.default_rng(
                    int(seed)
                )
            )

        base_env = (
            self.env.unwrapped
        )

        cube = (
            base_env.cube
        )

        nominal_position = (
            first_env(
                cube.pose.p
            )
            .astype(
                np.float64
            )
            .copy()
        )

        nominal_orientation = (
            first_env(
                cube.pose.q
            )
            .astype(
                np.float64
            )
            .copy()
        )

        offset = (
            self._sample_offset()
        )

        target_position = (
            nominal_position
            + offset
        )

        # Zは変更しない
        target_position[
            2
        ] = nominal_position[
            2
        ]

        device = (
            base_env.device
        )

        position_tensor = (
            torch.as_tensor(
                target_position,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
        )

        orientation_tensor = (
            torch.as_tensor(
                nominal_orientation,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
        )

        cube.set_pose(
            Pose.create_from_pq(
                p=position_tensor,
                q=orientation_tensor,
            )
        )

        zero_velocity = torch.zeros(
            (
                1,
                3,
            ),
            dtype=torch.float32,
            device=device,
        )

        cube.set_linear_velocity(
            zero_velocity
        )

        cube.set_angular_velocity(
            zero_velocity
        )

        actual_position = (
            first_env(
                cube.pose.p
            )
            .astype(
                np.float64
            )
            .copy()
        )

        self.last_offset = (
            actual_position
            - nominal_position
        )

        self.last_nominal_position = (
            nominal_position.copy()
        )

        self.last_actual_position = (
            actual_position.copy()
        )

        info = dict(
            info
        )

        info.update(
            {
                "cube_nominal_position_m":
                    nominal_position.tolist(),

                "cube_perturbed_position_m":
                    actual_position.tolist(),

                "cube_offset_m":
                    self.last_offset.tolist(),

                "cube_offset_xy_m":
                    self.last_offset[
                        :2
                    ].tolist(),

                "cube_offset_radius_m":
                    float(
                        np.linalg.norm(
                            self.last_offset[
                                :2
                            ]
                        )
                    ),

                "cube_perturbation_direction":
                    self.direction_mode,
            }
        )

        return (
            observation,
            info,
        )