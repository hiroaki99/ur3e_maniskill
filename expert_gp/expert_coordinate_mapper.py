#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coordinate mapping for Yaginuma expert trajectories -> ManiSkill UR3e TCP.

Yaginuma logical coordinates:
    [axis0, axis1] == [x, z]

Original Gazebo execution transform:
    a = 32.3 / 41.5
    gazebo_x = a * x + 0.15
    gazebo_y = -a * z + 0.165
    gazebo_z = fixed (0.12 m in the original simulator)

For ManiSkill we preserve the original scale and axis directions, but we do
NOT reuse the Gazebo absolute offsets (0.15, 0.165), because the UR3e base
frame/workspace origin differs.

Instead, an anchor pair is used:
    logical_reference = [x_ref, z_ref]
    tcp_reference     = [X_ref, Y_ref, Z_ref]

Mapping:
    TCP_X = X_ref + a * (x - x_ref)
    TCP_Y = Y_ref - a * (z - z_ref)
    TCP_Z = Z_ref

This preserves the original Yaginuma/Gazebo trajectory geometry while
placing it at an explicit reachable/safe location in ManiSkill.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


YAGINUMA_SCALE = 32.3 / 41.5


@dataclass(frozen=True)
class ExpertCoordinateMapper:
    """
    Map Yaginuma logical [x, z] trajectories into ManiSkill TCP [X, Y, Z].

    Parameters
    ----------
    logical_reference:
        Yaginuma logical coordinate [x_ref, z_ref] that corresponds to
        tcp_reference.

    tcp_reference:
        ManiSkill TCP base-frame position [X_ref, Y_ref, Z_ref] in metres.

    scale:
        Physical scale inherited from Yaginuma's Gazebo transform.
        Default = 32.3 / 41.5.

    Notes
    -----
    - axis0 == Yaginuma logical x
    - axis1 == Yaginuma logical z
    - logical +x -> ManiSkill TCP +X
    - logical +z -> ManiSkill TCP -Y
    - ManiSkill TCP Z is held fixed
    """

    logical_reference: np.ndarray
    tcp_reference: np.ndarray
    scale: float = YAGINUMA_SCALE

    def __post_init__(self):
        logical_reference = np.asarray(
            self.logical_reference,
            dtype=np.float64,
        )
        tcp_reference = np.asarray(
            self.tcp_reference,
            dtype=np.float64,
        )

        if logical_reference.shape != (2,):
            raise ValueError(
                "logical_reference must have shape (2,), "
                f"got {logical_reference.shape}."
            )

        if tcp_reference.shape != (3,):
            raise ValueError(
                "tcp_reference must have shape (3,), "
                f"got {tcp_reference.shape}."
            )

        if not np.all(np.isfinite(logical_reference)):
            raise ValueError("logical_reference contains NaN/Inf.")

        if not np.all(np.isfinite(tcp_reference)):
            raise ValueError("tcp_reference contains NaN/Inf.")

        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(
                f"scale must be finite and positive, got {self.scale}."
            )

        object.__setattr__(
            self,
            "logical_reference",
            logical_reference,
        )
        object.__setattr__(
            self,
            "tcp_reference",
            tcp_reference,
        )

    def logical_to_tcp(self, logical_xz) -> np.ndarray:
        """
        Convert one Yaginuma logical point [x, z] to TCP [X, Y, Z].
        """
        logical_xz = np.asarray(
            logical_xz,
            dtype=np.float64,
        )

        if logical_xz.shape != (2,):
            raise ValueError(
                "logical_xz must have shape (2,), "
                f"got {logical_xz.shape}."
            )

        delta = logical_xz - self.logical_reference

        tcp = self.tcp_reference.copy()
        tcp[0] += self.scale * delta[0]
        tcp[1] -= self.scale * delta[1]
        # tcp[2] intentionally remains fixed.

        return tcp

    def trajectory_to_tcp(self, logical_trajectory) -> np.ndarray:
        """
        Convert an N x 2 Yaginuma logical trajectory to N x 3 TCP positions.
        """
        logical_trajectory = np.asarray(
            logical_trajectory,
            dtype=np.float64,
        )

        if (
            logical_trajectory.ndim != 2
            or logical_trajectory.shape[1] != 2
        ):
            raise ValueError(
                "logical_trajectory must have shape (N, 2), "
                f"got {logical_trajectory.shape}."
            )

        if not np.all(np.isfinite(logical_trajectory)):
            raise ValueError(
                "logical_trajectory contains NaN/Inf."
            )

        delta = (
            logical_trajectory
            - self.logical_reference[None, :]
        )

        tcp = np.empty(
            (logical_trajectory.shape[0], 3),
            dtype=np.float64,
        )

        tcp[:, 0] = (
            self.tcp_reference[0]
            + self.scale * delta[:, 0]
        )

        tcp[:, 1] = (
            self.tcp_reference[1]
            - self.scale * delta[:, 1]
        )

        tcp[:, 2] = self.tcp_reference[2]

        return tcp

    def logical_delta_to_tcp_delta(self, logical_delta_xz) -> np.ndarray:
        """
        Convert a logical displacement [dx, dz] to TCP displacement [dX,dY,dZ].

        Useful later for residual actions:
            [Delta x, Delta z]
              -> [a*Delta x, -a*Delta z, 0]
        """
        logical_delta_xz = np.asarray(
            logical_delta_xz,
            dtype=np.float64,
        )

        if logical_delta_xz.shape != (2,):
            raise ValueError(
                "logical_delta_xz must have shape (2,), "
                f"got {logical_delta_xz.shape}."
            )

        return np.array(
            [
                self.scale * logical_delta_xz[0],
                -self.scale * logical_delta_xz[1],
                0.0,
            ],
            dtype=np.float64,
        )

    def tcp_to_logical(self, tcp_xyz) -> np.ndarray:
        """
        Inverse position mapping for X/Y. TCP Z is ignored because the
        Yaginuma logical task is two-dimensional.
        """
        tcp_xyz = np.asarray(
            tcp_xyz,
            dtype=np.float64,
        )

        if tcp_xyz.shape != (3,):
            raise ValueError(
                "tcp_xyz must have shape (3,), "
                f"got {tcp_xyz.shape}."
            )

        logical_x = (
            self.logical_reference[0]
            + (tcp_xyz[0] - self.tcp_reference[0]) / self.scale
        )

        logical_z = (
            self.logical_reference[1]
            - (tcp_xyz[1] - self.tcp_reference[1]) / self.scale
        )

        return np.array(
            [logical_x, logical_z],
            dtype=np.float64,
        )
