#!/usr/bin/env python3
"""Observation-only scaling utilities for Day17.

Important design choice:
    This wrapper must be OUTSIDE ResidualPickLiftEnv.

Stack:
    base ManiSkill env
      -> CubePositionPerturbationWrapper
      -> ResidualPickLiftEnv
      -> CubeToGraspObservationScaleWrapper

This keeps Day10 reward computation on the original unscaled observation while
only changing the observation seen by TD3 / ReplayBuffer.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class CubeToGraspObservationScaleWrapper(gym.ObservationWrapper):
    """Scale only the ``cube_to_grasp`` observation feature.

    Parameters
    ----------
    env:
        A ResidualPickLiftEnv (or a compatible wrapper exposing
        ``observation_layout`` and ``alpha``).
    scale:
        Multiplicative scale. Day17 candidate is 10.0.
    """

    def __init__(self, env: gym.Env, *, scale: float = 10.0):
        super().__init__(env)

        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"scale must be finite and > 0, got {self.scale}")

        layout = getattr(env, "observation_layout", None)
        if layout is None or "cube_to_grasp" not in layout:
            raise RuntimeError(
                "Wrapped env must expose observation_layout['cube_to_grasp']"
            )

        start, end = layout["cube_to_grasp"]
        self.start = int(start)
        self.end = int(end)

        if (self.start, self.end) != (30, 33):
            raise RuntimeError(
                "Day17 expects cube_to_grasp at [30:33], "
                f"got [{self.start}:{self.end}]"
            )

        # Shape/range remain valid because the original observation space is
        # Box(-inf, inf, (44,), float32).
        self.observation_space = env.observation_space

    @property
    def alpha(self) -> float:
        """Compatibility with rrl.evaluation.evaluate_residual_policy."""
        return float(self.env.alpha)

    @property
    def observation_layout(self):
        return self.env.observation_layout

    @property
    def cube_to_grasp_scale(self) -> float:
        return self.scale

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32).copy()

        expected_shape = self.observation_space.shape
        if obs.shape != expected_shape:
            raise RuntimeError(
                f"Observation shape mismatch: expected {expected_shape}, got {obs.shape}"
            )

        obs[self.start : self.end] *= self.scale
        return obs
