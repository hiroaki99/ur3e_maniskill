#!/usr/bin/env python3
"""TD3-only scaling for relative Pick-and-Place observation features."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class PickPlaceObservationScaleWrapper(gym.ObservationWrapper):
    """Scale selected fields while preserving the 56-D observation contract.

    This wrapper must be outside ResidualPickPlaceEnv so the reward continues to
    use raw SI-unit observations.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        cube_to_grasp_scale: float = 10.0,
        cube_to_goal_scale: float = 10.0,
        grasp_to_goal_scale: float = 10.0,
    ):
        super().__init__(env)
        if not hasattr(env, "observation_layout"):
            raise AttributeError("wrapped env must expose observation_layout")
        self.layout = dict(env.observation_layout)
        self.scales = {
            "cube_to_grasp": float(cube_to_grasp_scale),
            "cube_to_goal": float(cube_to_goal_scale),
            "grasp_to_goal": float(grasp_to_goal_scale),
        }
        for name, scale in self.scales.items():
            if scale <= 0 or not np.isfinite(scale):
                raise ValueError(f"invalid {name} scale: {scale}")
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        out = np.asarray(observation, dtype=np.float32).copy()
        for name, scale in self.scales.items():
            start, stop = self.layout[name]
            out[start:stop] *= np.float32(scale)
        return out

    def scaling_contract(self):
        return {
            "scaled_fields": dict(self.scales),
            "observation_dim": int(self.observation_space.shape[0]),
            "reward_observation": "raw inside ResidualPickPlaceEnv",
        }
