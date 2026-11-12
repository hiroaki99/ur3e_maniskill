#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reusable loader / mapper / resampler for Yaginuma expert trajectories.

This module is robot-independent:
- no ManiSkill import
- no ROS2 import
- no robot command

Input:
    reports/expert_gp/expert_XX.npz
        key: "mean", shape (N, 2)

Coordinate convention:
    Yaginuma logical [axis0, axis1] == [x, z]

Relative ManiSkill TCP mapping:
    logical +x -> TCP +X
    logical +z -> TCP -Y
    TCP Z       -> unchanged

The absolute robot-space anchor is deliberately NOT chosen here.
The caller (ExpertExecutor) anchors the relative trajectory to the
current TCP pose at execution/planning time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expert_gp.expert_coordinate_mapper import ExpertCoordinateMapper


NUM_EXPERTS = 12


@dataclass(frozen=True)
class ExpertTrajectory:
    expert_id: int
    logical: np.ndarray
    tcp_relative: np.ndarray
    tcp_sampled_relative: np.ndarray
    logical_path_length_m: float
    tcp_path_length_m: float
    sampled_path_length_m: float
    max_sampled_step_m: float


def _validate_expert_id(expert_id: int) -> int:
    expert_id = int(expert_id)

    if not 0 <= expert_id < NUM_EXPERTS:
        raise ValueError(
            f"expert_id must be in [0, {NUM_EXPERTS - 1}], "
            f"got {expert_id}."
        )

    return expert_id


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(points, axis=0),
                axis=1,
            )
        )
    )


def _max_adjacent_step(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0

    return float(
        np.max(
            np.linalg.norm(
                np.diff(points, axis=0),
                axis=1,
            )
        )
    )


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2:
        raise ValueError(
            f"points must be 2-D, got shape {points.shape}."
        )

    if points.shape[0] < 1:
        raise ValueError(
            "points must contain at least one point."
        )

    delta = np.diff(
        points,
        axis=0,
    )

    segment_length = np.linalg.norm(
        delta,
        axis=1,
    )

    s = np.empty(
        points.shape[0],
        dtype=np.float64,
    )

    s[0] = 0.0

    if points.shape[0] >= 2:
        s[1:] = np.cumsum(
            segment_length
        )

    return s


def resample_by_max_step(
    points: np.ndarray,
    max_step_m: float,
) -> np.ndarray:
    """
    Resample by path arc length.

    The nominal arc-length interval is <= max_step_m.
    The original first and last points are preserved.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2:
        raise ValueError(
            f"points must be 2-D, got shape {points.shape}."
        )

    if points.shape[0] < 2:
        raise ValueError(
            "points must contain at least two points."
        )

    if not np.all(
        np.isfinite(points)
    ):
        raise ValueError(
            "points contains NaN/Inf."
        )

    max_step_m = float(
        max_step_m
    )

    if (
        not np.isfinite(max_step_m)
        or max_step_m <= 0.0
    ):
        raise ValueError(
            "max_step_m must be finite and > 0."
        )

    s = cumulative_arc_length(
        points
    )

    total_length = float(
        s[-1]
    )

    if total_length == 0.0:
        return points[[0]].copy()

    num_segments = max(
        1,
        int(
            np.ceil(
                total_length
                / max_step_m
            )
        ),
    )

    target_s = np.linspace(
        0.0,
        total_length,
        num_segments + 1,
    )

    sampled = np.empty(
        (
            target_s.size,
            points.shape[1],
        ),
        dtype=np.float64,
    )

    for dim in range(
        points.shape[1]
    ):
        sampled[:, dim] = np.interp(
            target_s,
            s,
            points[:, dim],
        )

    # Make endpoint preservation explicit.
    sampled[0] = points[0]
    sampled[-1] = points[-1]

    return sampled


class ExpertLibrary:
    """
    Robot-independent Yaginuma expert trajectory library.
    """

    def __init__(
        self,
        expert_dir: str | Path,
    ):
        self.expert_dir = Path(
            expert_dir
        ).expanduser().resolve()

        if not self.expert_dir.is_dir():
            raise FileNotFoundError(
                f"Expert directory not found: "
                f"{self.expert_dir}"
            )

    def expert_path(
        self,
        expert_id: int,
    ) -> Path:
        expert_id = _validate_expert_id(
            expert_id
        )

        return (
            self.expert_dir
            / f"expert_{expert_id:02d}.npz"
        )

    def load_mean(
        self,
        expert_id: int,
    ) -> np.ndarray:
        path = self.expert_path(
            expert_id
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Expert file not found: {path}"
            )

        with np.load(path) as data:
            if "mean" not in data:
                raise KeyError(
                    f"{path} does not contain "
                    f"key 'mean'."
                )

            mean = np.asarray(
                data["mean"],
                dtype=np.float64,
            )

        if (
            mean.ndim != 2
            or mean.shape[1] != 2
        ):
            raise RuntimeError(
                "Expected expert mean shape "
                f"(N,2), got {mean.shape}."
            )

        if mean.shape[0] < 2:
            raise RuntimeError(
                "Expert trajectory contains "
                "fewer than two points."
            )

        if not np.all(
            np.isfinite(mean)
        ):
            raise RuntimeError(
                "Expert trajectory contains "
                "NaN/Inf."
            )

        return mean

    def prepare_relative(
        self,
        expert_id: int,
        max_step_m: float,
    ) -> ExpertTrajectory:
        """
        Load one expert and create a RELATIVE TCP trajectory.

        The expert start is mapped to TCP [0, 0, 0].
        Absolute anchoring is intentionally deferred to the executor.
        """
        expert_id = _validate_expert_id(
            expert_id
        )

        logical = self.load_mean(
            expert_id
        )

        mapper = ExpertCoordinateMapper(
            logical_reference=(
                logical[0].copy()
            ),
            tcp_reference=np.zeros(
                3,
                dtype=np.float64,
            ),
        )

        tcp_relative = (
            mapper.trajectory_to_tcp(
                logical
            )
        )

        tcp_sampled_relative = (
            resample_by_max_step(
                tcp_relative,
                max_step_m=max_step_m,
            )
        )

        return ExpertTrajectory(
            expert_id=expert_id,
            logical=logical,
            tcp_relative=tcp_relative,
            tcp_sampled_relative=(
                tcp_sampled_relative
            ),
            logical_path_length_m=(
                _path_length(
                    logical
                )
            ),
            tcp_path_length_m=(
                _path_length(
                    tcp_relative
                )
            ),
            sampled_path_length_m=(
                _path_length(
                    tcp_sampled_relative
                )
            ),
            max_sampled_step_m=(
                _max_adjacent_step(
                    tcp_sampled_relative
                )
            ),
        )
