#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Physical ManiSkill joint-control backend for UR3eReach-v0.

Purpose
-------
Use ManiSkill's controller + env.step(action) for robot motion instead of
directly overwriting qpos.

This backend intentionally reuses ManiSkillBackend ONLY for read-only state/FK:
    - get_joint_positions()
    - get_tcp_pose()

It NEVER calls ManiSkillBackend.command_joint_positions().

Controller contract confirmed for UR3eReach-v0 / pd_joint_delta_pos:
    action_space      : Box(-1, 1, (6,), float32)
    normalize_action  : True
    use_delta         : True
    lower             : -0.03 rad
    upper             : +0.03 rad

Therefore, for the current symmetric configuration:
    normalized_action = desired_joint_delta_rad / 0.03

The implementation below derives the inverse normalization from the runtime
controller configuration rather than hard-coding 0.03.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from scripts_sim2real.maniskill_backend import ManiSkillBackend
from scripts_sim2real.cartesian_delta_controller import JOINT_NAMES


class ManiSkillControlBackend:
    """
    RobotBackend-compatible physical-control adapter for ManiSkill.

    Notes
    -----
    - Motion uses env.step(action).
    - State/TCP reads delegate to ManiSkillBackend for consistency with the
      already validated virtual TCP definition (tool0 + local Z 0.150 m).
    - `max_requested_joint_delta_rad` protects the same per-command joint-delta
      envelope already used by CartesianDeltaController.
    """

    def __init__(
        self,
        env,
        *,
        joint_tolerance_rad: float = 2.0e-4,
        max_requested_joint_delta_rad: float = 1.0e-2,
        default_duration_sec: float = 0.5,
    ):
        self.env = env
        self.unwrapped = env.unwrapped

        self.joint_tolerance_rad = float(
            joint_tolerance_rad
        )
        self.max_requested_joint_delta_rad = float(
            max_requested_joint_delta_rad
        )
        self.default_duration_sec = float(
            default_duration_sec
        )

        if self.joint_tolerance_rad <= 0.0:
            raise ValueError(
                "joint_tolerance_rad must be > 0."
            )

        if self.max_requested_joint_delta_rad <= 0.0:
            raise ValueError(
                "max_requested_joint_delta_rad must be > 0."
            )

        if self.default_duration_sec <= 0.0:
            raise ValueError(
                "default_duration_sec must be > 0."
            )

        # Existing validated reader/FK implementation.
        # Do not call its command_joint_positions() here.
        self._state_reader = ManiSkillBackend(
            env
        )

        controller = (
            self.unwrapped.agent.controller
        )

        if not hasattr(
            controller,
            "controllers",
        ):
            raise RuntimeError(
                "Expected CombinedController with "
                "sub-controller mapping."
            )

        if "arm" not in controller.controllers:
            raise RuntimeError(
                "CombinedController does not contain "
                "'arm' controller."
            )

        self.arm_controller = (
            controller.controllers["arm"]
        )
        self.arm_config = (
            self.arm_controller.config
        )

        self._validate_controller_contract()

        self.lower = self._broadcast_limit(
            self.arm_config.lower,
            "lower",
        )
        self.upper = self._broadcast_limit(
            self.arm_config.upper,
            "upper",
        )

        if np.any(
            self.upper <= self.lower
        ):
            raise RuntimeError(
                "Controller upper limits must be "
                "greater than lower limits."
            )

        self.action_low = np.asarray(
            self.env.action_space.low,
            dtype=np.float64,
        ).reshape(-1)

        self.action_high = np.asarray(
            self.env.action_space.high,
            dtype=np.float64,
        ).reshape(-1)

        if self.action_low.shape != (6,):
            raise RuntimeError(
                "Expected action_space shape (6,), "
                f"got {self.env.action_space.shape}."
            )

        self.control_freq = self._resolve_control_freq()

        self.last_command_result = None

    def _broadcast_limit(
        self,
        value,
        name: str,
    ) -> np.ndarray:
        arr = np.asarray(
            value,
            dtype=np.float64,
        )

        if arr.ndim == 0:
            return np.full(
                6,
                float(arr),
                dtype=np.float64,
            )

        arr = arr.reshape(-1)

        if arr.shape != (6,):
            raise RuntimeError(
                f"Controller {name} must be scalar "
                f"or shape (6,), got {arr.shape}."
            )

        return arr

    def _validate_controller_contract(
        self,
    ):
        cfg = self.arm_config

        joint_names = list(
            cfg.joint_names
        )

        if joint_names != JOINT_NAMES:
            raise RuntimeError(
                "Arm joint order mismatch.\n"
                f"expected={JOINT_NAMES}\n"
                f"actual={joint_names}"
            )

        if not bool(
            getattr(
                cfg,
                "normalize_action",
                False,
            )
        ):
            raise RuntimeError(
                "This backend expects "
                "normalize_action=True."
            )

        if not bool(
            getattr(
                cfg,
                "use_delta",
                False,
            )
        ):
            raise RuntimeError(
                "This backend expects "
                "use_delta=True."
            )

        if bool(
            getattr(
                cfg,
                "use_target",
                False,
            )
        ):
            raise RuntimeError(
                "This backend is for "
                "pd_joint_delta_pos with "
                "use_target=False."
            )

        if cfg.lower is None or cfg.upper is None:
            raise RuntimeError(
                "Normalized delta controller requires "
                "finite lower/upper bounds."
            )

    def _resolve_control_freq(
        self,
    ) -> float:
        for obj in (
            self.unwrapped,
            getattr(
                self.unwrapped,
                "scene",
                None,
            ),
        ):
            if obj is None:
                continue

            for name in (
                "control_freq",
                "_control_freq",
            ):
                if hasattr(
                    obj,
                    name,
                ):
                    value = float(
                        getattr(
                            obj,
                            name,
                        )
                    )

                    if value > 0.0:
                        return value

        # ManiSkill normally exposes control_freq on BaseEnv.
        # Refuse to invent a timing conversion if it is unavailable.
        raise RuntimeError(
            "Could not determine ManiSkill control_freq."
        )

    def get_joint_positions(
        self,
    ) -> Dict[str, float]:
        return (
            self._state_reader
            .get_joint_positions()
        )

    def get_tcp_pose(
        self,
    ):
        return (
            self._state_reader
            .get_tcp_pose()
        )

    def _joint_dict_to_array(
        self,
        joint_positions,
    ) -> np.ndarray:
        missing = [
            name
            for name in JOINT_NAMES
            if name not in joint_positions
        ]

        if missing:
            raise ValueError(
                "joint_positions is missing: "
                + ", ".join(missing)
            )

        values = np.asarray(
            [
                joint_positions[name]
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "joint_positions contains NaN/Inf."
            )

        return values

    def _current_q_array(
        self,
    ) -> np.ndarray:
        q = self.get_joint_positions()

        return self._joint_dict_to_array(
            q
        )

    def joint_delta_to_normalized_action(
        self,
        delta_q_rad,
    ) -> np.ndarray:
        """
        Inverse of normalized action mapping.

        Normalized [-1,1] -> physical controller [lower, upper]:

            physical =
                lower
                + (normalized + 1) * (upper-lower)/2

        Therefore:

            normalized =
                2*(physical-lower)/(upper-lower) - 1
        """
        delta_q_rad = np.asarray(
            delta_q_rad,
            dtype=np.float64,
        ).reshape(-1)

        if delta_q_rad.shape != (6,):
            raise ValueError(
                "delta_q_rad must have shape (6,)."
            )

        normalized = (
            2.0
            * (
                delta_q_rad
                - self.lower
            )
            / (
                self.upper
                - self.lower
            )
            - 1.0
        )

        return np.clip(
            normalized,
            self.action_low,
            self.action_high,
        ).astype(
            np.float32
        )

    def command_joint_positions(
        self,
        joint_positions,
        duration_sec=None,
    ):
        """
        Track a six-joint target using repeated ManiSkill env.step(action).

        `joint_positions` follows the existing RobotBackend contract:
            Dict[str, float]

        The initial requested target delta must remain within
        `max_requested_joint_delta_rad`.  The backend does not relax the
        Cartesian controller's 0.01 rad safety envelope.
        """
        if duration_sec is None:
            duration_sec = (
                self.default_duration_sec
            )

        duration_sec = float(
            duration_sec
        )

        if duration_sec <= 0.0:
            raise ValueError(
                "duration_sec must be > 0 "
                "for physical env.step control."
            )

        q_target = (
            self._joint_dict_to_array(
                joint_positions
            )
        )

        q_initial = (
            self._current_q_array()
        )

        initial_error = (
            q_target
            - q_initial
        )

        initial_max_error = float(
            np.max(
                np.abs(
                    initial_error
                )
            )
        )

        if (
            initial_max_error
            > self.max_requested_joint_delta_rad
        ):
            raise RuntimeError(
                "Requested joint target exceeds "
                "physical-control safety limit: "
                f"{initial_max_error:.6f} rad > "
                f"{self.max_requested_joint_delta_rad:.6f} rad"
            )

        max_env_steps = max(
            1,
            int(
                math.ceil(
                    duration_sec
                    * self.control_freq
                )
            ),
        )

        max_action_abs = 0.0
        terminated_seen = False
        truncated_seen = False

        for env_step_id in range(
            1,
            max_env_steps + 1,
        ):
            q_current = (
                self._current_q_array()
            )

            error = (
                q_target
                - q_current
            )

            max_error = float(
                np.max(
                    np.abs(
                        error
                    )
                )
            )

            if (
                max_error
                <= self.joint_tolerance_rad
            ):
                result = {
                    "success": True,
                    "env_steps": (
                        env_step_id - 1
                    ),
                    "control_freq_hz": (
                        self.control_freq
                    ),
                    "initial_max_joint_error_rad": (
                        initial_max_error
                    ),
                    "final_max_joint_error_rad": (
                        max_error
                    ),
                    "max_abs_normalized_action": (
                        max_action_abs
                    ),
                    "terminated_seen": (
                        terminated_seen
                    ),
                    "truncated_seen": (
                        truncated_seen
                    ),
                    "target_qpos": (
                        q_target.copy()
                    ),
                    "final_qpos": (
                        q_current.copy()
                    ),
                }

                self.last_command_result = (
                    result
                )

                return result

            # use_delta=True/use_target=False:
            # ask the controller for the remaining joint error.
            physical_delta = np.clip(
                error,
                self.lower,
                self.upper,
            )

            action = (
                self
                .joint_delta_to_normalized_action(
                    physical_delta
                )
            )

            max_action_abs = max(
                max_action_abs,
                float(
                    np.max(
                        np.abs(
                            action
                        )
                    )
                ),
            )

            (
                _obs,
                _reward,
                terminated,
                truncated,
                _info,
            ) = self.env.step(
                action
            )

            terminated_seen = (
                terminated_seen
                or bool(
                    np.asarray(
                        terminated
                    ).reshape(-1)[0]
                )
            )

            truncated_seen = (
                truncated_seen
                or bool(
                    np.asarray(
                        truncated
                    ).reshape(-1)[0]
                )
            )

            if (
                terminated_seen
                or truncated_seen
            ):
                break

        q_final = (
            self._current_q_array()
        )

        final_error = (
            q_target
            - q_final
        )

        final_max_error = float(
            np.max(
                np.abs(
                    final_error
                )
            )
        )

        success = (
            final_max_error
            <= self.joint_tolerance_rad
        )

        result = {
            "success": bool(
                success
            ),
            "env_steps": int(
                min(
                    max_env_steps,
                    env_step_id,
                )
            ),
            "control_freq_hz": (
                self.control_freq
            ),
            "initial_max_joint_error_rad": (
                initial_max_error
            ),
            "final_max_joint_error_rad": (
                final_max_error
            ),
            "max_abs_normalized_action": (
                max_action_abs
            ),
            "terminated_seen": (
                terminated_seen
            ),
            "truncated_seen": (
                truncated_seen
            ),
            "target_qpos": (
                q_target.copy()
            ),
            "final_qpos": (
                q_final.copy()
            ),
        }

        self.last_command_result = (
            result
        )

        if not success:
            reason = (
                "ManiSkill physical joint tracking "
                "did not converge: "
                f"final max error="
                f"{final_max_error:.6f} rad, "
                f"tolerance="
                f"{self.joint_tolerance_rad:.6f} rad, "
                f"env_steps="
                f"{result['env_steps']}"
            )

            if terminated_seen:
                reason += ", terminated=True"

            if truncated_seen:
                reason += ", truncated=True"

            raise RuntimeError(
                reason
            )

        return result

    def close(
        self,
    ):
        """
        No-op.

        The caller owns the Gym/ManiSkill environment and should close it.
        """
        return None
