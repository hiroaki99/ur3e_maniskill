#!/usr/bin/env python3
"""
Phase-dependent residual action scaling.

Day16:
Lift phaseだけResidual actionを縮小する。

External action spaceは従来通り [-1, 1]^6。
TD3 Actor自体は変更しない。

Example
-------
lift_scale = 0.5

non-lift:
    a_applied = a_raw

lift:
    a_applied = 0.5 * a_raw

その後ResidualPickLiftEnv内部で

    a_exec = a_ref + alpha * a_applied

が計算される。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


PHASE_NAMES = [
    "to_pregrasp",
    "descend",
    "grasp",
    "stable_hold",
    "lift",
    "final_hold",
    "terminal",
]


class LiftResidualScaleWrapper(
    gym.Wrapper
):
    """
    Lift phaseだけResidualを縮小するWrapper。

    Day9 observation contract:
        phase one-hot = observation[33:40]

    lift phase index:
        4
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        lift_residual_scale: float = 1.0,
        lift_phase_index: int = 4,
    ):

        super().__init__(
            env
        )

        if not (
            0.0
            <= lift_residual_scale
            <= 1.0
        ):

            raise ValueError(
                "lift_residual_scale must "
                "be in [0, 1]"
            )

        self.lift_residual_scale = float(
            lift_residual_scale
        )

        self.lift_phase_index = int(
            lift_phase_index
        )

        self._last_observation = None

        # Day15等の既存コードとの互換性
        self.alpha = float(
            getattr(
                env,
                "alpha",
                1.0,
            )
        )

    # ======================================================
    # Phase
    # ======================================================

    @staticmethod
    def phase_index_from_observation(
        observation,
    ) -> int:

        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).reshape(-1)

        if observation.size < 40:

            raise ValueError(
                "Observation must contain "
                "phase one-hot at [33:40]"
            )

        phase_onehot = observation[
            33:40
        ]

        return int(
            np.argmax(
                phase_onehot
            )
        )

    # ======================================================
    # Reset
    # ======================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        observation, info = (
            self.env.reset(
                seed=seed,
                options=options,
            )
        )

        self._last_observation = (
            observation
        )

        info = dict(
            info
        )

        info[
            "lift_residual_scale"
        ] = self.lift_residual_scale

        return (
            observation,
            info,
        )

    # ======================================================
    # Step
    # ======================================================

    def step(
        self,
        action,
    ):

        if self._last_observation is None:

            raise RuntimeError(
                "reset() must be called "
                "before step()"
            )

        raw_action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(
            self.action_space.shape
        )

        # 外部action boundを保証
        raw_action = np.clip(
            raw_action,
            self.action_space.low,
            self.action_space.high,
        ).astype(
            np.float32
        )

        phase_index = (
            self.phase_index_from_observation(
                self._last_observation
            )
        )

        if (
            phase_index
            == self.lift_phase_index
        ):

            phase_scale = (
                self.lift_residual_scale
            )

        else:

            phase_scale = 1.0

        applied_action = (
            raw_action
            * phase_scale
        ).astype(
            np.float32
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            applied_action
        )

        info = dict(
            info
        )

        info[
            "raw_residual_action"
        ] = raw_action.copy()

        info[
            "applied_residual_action"
        ] = applied_action.copy()

        info[
            "residual_phase_index"
        ] = int(
            phase_index
        )

        info[
            "residual_phase_name"
        ] = (
            PHASE_NAMES[
                phase_index
            ]
            if 0
            <= phase_index
            < len(
                PHASE_NAMES
            )
            else "unknown"
        )

        info[
            "residual_phase_scale"
        ] = float(
            phase_scale
        )

        info[
            "lift_residual_scale"
        ] = float(
            self.lift_residual_scale
        )

        self._last_observation = (
            observation
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )