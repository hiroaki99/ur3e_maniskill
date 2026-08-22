#!/usr/bin/env python3
"""Reward model for UR3e Residual Pick-and-Place.

The reward is computed inside ResidualPickPlaceEnv from *raw* observations.
Any TD3-only observation scaling wrapper must sit outside that environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ResidualPickPlaceReward:
    APPROACH_PHASES = {"to_pregrasp", "descend"}
    GRASP_PHASES = {"grasp", "stable_hold"}
    HOLD_PHASES = {"lift", "transport", "descend_place"}

    def __init__(self, config: dict, observation_layout: dict, alpha: float):
        self.config = dict(config)
        self.layout = dict(observation_layout)
        self.alpha = float(alpha)

        self.approach_scale = float(self.config.get("approach_progress_scale_m", 0.01))
        self.approach_weight = float(self.config.get("approach_progress_weight", 0.20))
        self.bilateral_contact_bonus = float(self.config.get("bilateral_contact_bonus", 0.05))
        self.hold_contact_bonus = float(self.config.get("hold_contact_bonus", 0.02))

        self.lift_weight = float(self.config.get("lift_progress_weight", 50.0))
        self.max_lift_delta = float(self.config.get("max_lift_delta_per_step_m", 0.01))

        self.transport_scale = float(self.config.get("transport_progress_scale_m", 0.01))
        self.transport_weight = float(self.config.get("transport_progress_weight", 0.30))
        self.place_scale = float(self.config.get("place_progress_scale_m", 0.01))
        self.place_weight = float(self.config.get("place_progress_weight", 0.30))

        self.release_bonus = float(self.config.get("release_bonus", 0.50))
        self.stable_step_bonus = float(self.config.get("stable_step_bonus", 0.02))
        self.success_bonus = float(self.config.get("success_bonus", 10.0))
        self.failure_penalty = float(self.config.get("failure_penalty", 5.0))
        self.residual_penalty_weight = float(self.config.get("residual_penalty_weight", 0.05))

        self.enable_force_penalty = bool(
            self.config.get("enable_excessive_force_penalty", False)
        )
        self.max_safe_force = float(self.config.get("max_safe_gripper_force_n", 100.0))
        self.excessive_force_weight = float(
            self.config.get("excessive_force_penalty_weight", 0.20)
        )
        self.enable_collision_penalty = bool(
            self.config.get("enable_unsafe_collision_penalty", False)
        )
        self.unsafe_collision_penalty = float(
            self.config.get("unsafe_collision_penalty", 5.0)
        )

        for value, name in [
            (self.approach_scale, "approach_progress_scale_m"),
            (self.transport_scale, "transport_progress_scale_m"),
            (self.place_scale, "place_progress_scale_m"),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be > 0")

    def _slice(self, observation, name: str) -> np.ndarray:
        start, stop = self.layout[name]
        return np.asarray(observation, dtype=np.float64)[start:stop]

    def _scalar(self, observation, name: str) -> float:
        value = self._slice(observation, name)
        if value.size != 1:
            raise ValueError(f"{name} is not scalar")
        return float(value.reshape(-1)[0])

    @staticmethod
    def _progress_reward(previous_distance, current_distance, scale, weight):
        progress = float(previous_distance - current_distance)
        normalized = float(np.clip(progress / scale, -1.0, 1.0))
        return weight * normalized

    def compute(
        self,
        previous_observation,
        observation,
        residual_action,
        info: dict[str, Any],
        terminated: bool,
    ):
        prev = np.asarray(previous_observation, dtype=np.float64)
        obs = np.asarray(observation, dtype=np.float64)
        residual = np.clip(np.asarray(residual_action, dtype=np.float64).reshape(-1), -1, 1)
        phase = str(info.get("phase_name_before", ""))

        reward_approach = 0.0
        if phase in self.APPROACH_PHASES:
            reward_approach = self._progress_reward(
                np.linalg.norm(self._slice(prev, "cube_to_grasp")),
                np.linalg.norm(self._slice(obs, "cube_to_grasp")),
                self.approach_scale,
                self.approach_weight,
            )

        left = bool(info.get("left_contact", False))
        right = bool(info.get("right_contact", False))
        both = left and right
        reward_grasp = self.bilateral_contact_bonus if both and phase in self.GRASP_PHASES else 0.0
        reward_hold = self.hold_contact_bonus if both and phase in self.HOLD_PHASES else 0.0

        reward_lift = 0.0
        if phase in {"stable_hold", "lift"}:
            delta = self._scalar(obs, "cube_lift") - self._scalar(prev, "cube_lift")
            delta = float(np.clip(delta, -self.max_lift_delta, self.max_lift_delta))
            reward_lift = self.lift_weight * delta

        reward_transport = 0.0
        if phase == "transport":
            reward_transport = self._progress_reward(
                np.linalg.norm(self._slice(prev, "cube_to_goal")[:2]),
                np.linalg.norm(self._slice(obs, "cube_to_goal")[:2]),
                self.transport_scale,
                self.transport_weight,
            )

        reward_place = 0.0
        if phase == "descend_place":
            reward_place = self._progress_reward(
                np.linalg.norm(self._slice(prev, "cube_to_goal")),
                np.linalg.norm(self._slice(obs, "cube_to_goal")),
                self.place_scale,
                self.place_weight,
            )

        near_goal = bool(info.get("within_goal_xy", False)) and bool(
            info.get("within_goal_z", False)
        )
        released = bool(info.get("released", False))
        reward_release = self.release_bonus if phase == "release" and near_goal and released else 0.0

        stable_steps = int(info.get("stable_place_steps", 0))
        reward_stable = self.stable_step_bonus if near_goal and released and stable_steps > 0 else 0.0

        # Penalize only the correction that can actually reach the low-level controller.
        effective_residual = self.alpha * residual
        reward_residual = -self.residual_penalty_weight * float(
            np.mean(np.square(effective_residual))
        )

        reward_force = 0.0
        if self.enable_force_penalty:
            max_force = max(
                float(info.get("left_contact_force", 0.0)),
                float(info.get("right_contact_force", 0.0)),
            )
            excess_ratio = max(0.0, (max_force - self.max_safe_force) / self.max_safe_force)
            reward_force = -self.excessive_force_weight * min(excess_ratio, 1.0)

        reward_collision = 0.0
        if self.enable_collision_penalty and bool(info.get("unsafe_collision", False)):
            reward_collision = -self.unsafe_collision_penalty

        reward_terminal = 0.0
        if terminated:
            reward_terminal = self.success_bonus if bool(info.get("success", False)) else -self.failure_penalty

        terms = {
            "approach": float(reward_approach),
            "grasp": float(reward_grasp),
            "hold": float(reward_hold),
            "lift": float(reward_lift),
            "transport": float(reward_transport),
            "place": float(reward_place),
            "release": float(reward_release),
            "stable_place": float(reward_stable),
            "residual_penalty": float(reward_residual),
            "excessive_force": float(reward_force),
            "unsafe_collision": float(reward_collision),
            "terminal": float(reward_terminal),
        }
        total = float(sum(terms.values()))
        if not np.isfinite(total):
            raise RuntimeError(f"non-finite reward: {terms}")
        return total, terms
