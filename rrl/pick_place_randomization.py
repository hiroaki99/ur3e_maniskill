#!/usr/bin/env python3
"""Cube/Goal XY randomization for UR3e Pick-and-Place.

Wrapper order for Week2+:
    UR3ePickPlace-v0
      -> PickPlaceRandomizationWrapper
      -> ResidualPickPlaceEnv
      -> PickPlaceObservationScaleWrapper

Randomization happens after the base reset but before ResidualPickPlaceEnv
captures its episode initial state and produces the first TD3 observation.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.structs.pose import Pose


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def first_env(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr[0]
    return arr


class PickPlaceRandomizationWrapper(gym.Wrapper):
    VALID_MODES = {"none", "cube_only", "goal_only", "both"}
    VALID_SAMPLING = {"area_uniform_disk", "fixed_ring"}

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: str = "both",
        cube_radius_min_m: float = 0.0,
        cube_radius_max_m: float = 0.01,
        goal_radius_min_m: float = 0.0,
        goal_radius_max_m: float = 0.01,
        sampling: str = "area_uniform_disk",
        fixed_cube_angle_rad: float | None = None,
        fixed_goal_angle_rad: float | None = None,
        seed: int = 0,
    ):
        super().__init__(env)
        self.mode = str(mode)
        self.sampling = str(sampling)
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.sampling not in self.VALID_SAMPLING:
            raise ValueError(f"unknown sampling: {self.sampling}")

        self.cube_radius_min_m = float(cube_radius_min_m)
        self.cube_radius_max_m = float(cube_radius_max_m)
        self.goal_radius_min_m = float(goal_radius_min_m)
        self.goal_radius_max_m = float(goal_radius_max_m)
        for lo, hi, name in [
            (self.cube_radius_min_m, self.cube_radius_max_m, "cube"),
            (self.goal_radius_min_m, self.goal_radius_max_m, "goal"),
        ]:
            if lo < 0 or hi < lo:
                raise ValueError(f"invalid {name} radius range: {lo}, {hi}")

        self.fixed_cube_angle_rad = (
            None if fixed_cube_angle_rad is None else float(fixed_cube_angle_rad)
        )
        self.fixed_goal_angle_rad = (
            None if fixed_goal_angle_rad is None else float(fixed_goal_angle_rad)
        )
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)

        base = self.env.unwrapped
        self.nominal_cube_position = np.asarray(
            base.cube_initial_position, dtype=np.float64
        ).copy()
        self.nominal_goal_position = np.asarray(
            base.goal_position, dtype=np.float64
        ).copy()

        self.last_cube_offset = np.zeros(3, dtype=np.float64)
        self.last_goal_offset = np.zeros(3, dtype=np.float64)
        self.last_cube_position = self.nominal_cube_position.copy()
        self.last_goal_position = self.nominal_goal_position.copy()

    def _sample_radius(self, lo: float, hi: float) -> float:
        if np.isclose(lo, hi):
            return float(lo)
        if self.sampling == "area_uniform_disk":
            u = float(self.rng.uniform(0.0, 1.0))
            return float(np.sqrt(lo * lo + u * (hi * hi - lo * lo)))
        return float(self.rng.uniform(lo, hi))

    def _sample_offset(
        self,
        lo: float,
        hi: float,
        fixed_angle_rad: float | None,
    ) -> np.ndarray:
        r = self._sample_radius(lo, hi)
        if np.isclose(r, 0.0):
            return np.zeros(3, dtype=np.float64)
        theta = (
            float(fixed_angle_rad)
            if fixed_angle_rad is not None
            else float(self.rng.uniform(0.0, 2.0 * np.pi))
        )
        return np.asarray([r * np.cos(theta), r * np.sin(theta), 0.0], dtype=np.float64)

    def _set_cube(self, position: np.ndarray):
        base = self.env.unwrapped
        cube = base.cube
        orientation = first_env(cube.pose.q).astype(np.float64).copy()
        p = torch.as_tensor(position, dtype=torch.float32, device=base.device).unsqueeze(0)
        q = torch.as_tensor(orientation, dtype=torch.float32, device=base.device).unsqueeze(0)
        cube.set_pose(Pose.create_from_pq(p=p, q=q))
        zero = torch.zeros((1, 3), dtype=torch.float32, device=base.device)
        cube.set_linear_velocity(zero)
        cube.set_angular_velocity(zero)

    def _set_goal(self, position: np.ndarray):
        base = self.env.unwrapped
        base.goal_position = np.asarray(position, dtype=np.float32).copy()
        marker_p = np.asarray(
            [position[0], position[1], 0.001], dtype=np.float64
        )
        p = torch.as_tensor(marker_p, dtype=torch.float32, device=base.device).unsqueeze(0)
        q = torch.as_tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=base.device)
        base.goal_marker.set_pose(Pose.create_from_pq(p=p, q=q))

    def current_randomization_info(self) -> dict:
        return {
            "randomization_mode": self.mode,
            "randomization_sampling": self.sampling,
            "cube_nominal_position_m": self.nominal_cube_position.tolist(),
            "cube_position_m": self.last_cube_position.tolist(),
            "cube_offset_m": self.last_cube_offset.tolist(),
            "cube_offset_radius_m": float(np.linalg.norm(self.last_cube_offset[:2])),
            "goal_nominal_position_m": self.nominal_goal_position.tolist(),
            "goal_position_m": self.last_goal_position.tolist(),
            "goal_offset_m": self.last_goal_offset.tolist(),
            "goal_offset_radius_m": float(np.linalg.norm(self.last_goal_offset[:2])),
        }

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        cube_offset = np.zeros(3, dtype=np.float64)
        goal_offset = np.zeros(3, dtype=np.float64)

        if self.mode in {"cube_only", "both"}:
            cube_offset = self._sample_offset(
                self.cube_radius_min_m,
                self.cube_radius_max_m,
                self.fixed_cube_angle_rad,
            )
        if self.mode in {"goal_only", "both"}:
            goal_offset = self._sample_offset(
                self.goal_radius_min_m,
                self.goal_radius_max_m,
                self.fixed_goal_angle_rad,
            )

        cube_position = self.nominal_cube_position + cube_offset
        goal_position = self.nominal_goal_position + goal_offset
        cube_position[2] = self.nominal_cube_position[2]
        goal_position[2] = self.nominal_goal_position[2]

        self._set_cube(cube_position)
        self._set_goal(goal_position)

        self.last_cube_position = first_env(self.env.unwrapped.cube.pose.p).astype(np.float64).copy()
        self.last_goal_position = np.asarray(self.env.unwrapped.goal_position, dtype=np.float64).copy()
        self.last_cube_offset = self.last_cube_position - self.nominal_cube_position
        self.last_goal_offset = self.last_goal_position - self.nominal_goal_position

        info = dict(info)
        info.update(self.current_randomization_info())
        return observation, info
