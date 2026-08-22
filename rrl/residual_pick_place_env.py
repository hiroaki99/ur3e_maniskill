#!/usr/bin/env python3
"""Residual RL environment for UR3e + EZGripper Pick-and-Place.

External action:
    6-D normalized arm residual in [-1, 1]

Executed arm action:
    clip(reference_action + alpha * residual_action, -1, 1)

Gripper:
    scripted from the validated Week1 reference procedure.

Observation:
    56-D raw state. Observation-only scaling is applied by an outer wrapper.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from rrl.pick_place_reward import ResidualPickPlaceReward


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


def scalar_first(value: Any) -> float:
    arr = np.asarray(first_env(value))
    if arr.size != 1:
        raise ValueError(f"scalar expected, shape={arr.shape}")
    return float(arr.reshape(-1)[0])


def get_controller_space(controller):
    if hasattr(controller, "single_action_space"):
        return controller.single_action_space
    return controller.action_space


def build_controller_slices(controllers: Mapping):
    result = {}
    offset = 0
    for name, controller in controllers.items():
        dim = int(np.prod(get_controller_space(controller).shape))
        result[str(name)] = slice(offset, offset + dim)
        offset += dim
    return result


def find_controller(controllers, keyword: str):
    for name in controllers:
        if keyword in str(name).lower():
            return str(name)
    raise RuntimeError(f"{keyword} controller not found: {list(controllers.keys())}")


def get_controller_joint_names(controller):
    names = getattr(controller.config, "joint_names", None)
    return [] if names is None else [str(x) for x in names]


class ResidualPickPlaceEnv(gym.Wrapper):
    PHASE_TO_PREGRASP = 0
    PHASE_DESCEND = 1
    PHASE_GRASP = 2
    PHASE_STABLE_HOLD = 3
    PHASE_LIFT = 4
    PHASE_TRANSPORT = 5
    PHASE_DESCEND_PLACE = 6
    PHASE_RELEASE = 7
    PHASE_FINAL_SETTLE = 8
    PHASE_TERMINAL = 9

    PHASE_NAMES = [
        "to_pregrasp",
        "descend",
        "grasp",
        "stable_hold",
        "lift",
        "transport",
        "descend_place",
        "release",
        "final_settle",
        "terminal",
    ]
    PHASE_COUNT = len(PHASE_NAMES)
    OBSERVATION_DIM = 56

    def __init__(
        self,
        env: gym.Env,
        trajectory_path: str | Path,
        config_path: str | Path,
        alpha: float | None = None,
    ):
        super().__init__(env)
        self.trajectory_path = Path(trajectory_path)
        self.config_path = Path(config_path)

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with self.trajectory_path.open("r", encoding="utf-8") as f:
            self.trajectory = json.load(f)

        self.pp = self.config["pick_place"]
        self.gripper_cfg = self.config["gripper"]
        self.residual_cfg = self.config.get("residual_pick_place", {})
        self.alpha = float(
            alpha if alpha is not None else self.residual_cfg.get("alpha", 0.20)
        )
        self.max_joint_delta = float(self.config["control"]["max_joint_delta_rad"])
        if self.max_joint_delta <= 0:
            raise ValueError("control.max_joint_delta_rad must be > 0")

        self.base_env = self.env.unwrapped
        self.robot = self.base_env.agent.robot
        controllers = self.base_env.agent.controller.controllers
        if not isinstance(controllers, Mapping):
            raise RuntimeError("CombinedController expected")
        slices = build_controller_slices(controllers)
        self.arm_name = find_controller(controllers, "arm")
        self.gripper_name = find_controller(controllers, "gripper")
        self.arm_controller = controllers[self.arm_name]
        self.arm_slice = slices[self.arm_name]
        self.gripper_slice = slices[self.gripper_name]
        self.arm_joint_names = get_controller_joint_names(self.arm_controller)
        self.arm_dof = len(self.arm_joint_names)
        if self.arm_dof != 6:
            raise RuntimeError(f"UR3e arm DoF=6 expected, got {self.arm_dof}")
        if self.trajectory["arm_joint_names"] != self.arm_joint_names:
            raise RuntimeError("trajectory and current arm joint order differ")

        active_names = [joint.name for joint in self.robot.active_joints]
        name_to_index = {name: i for i, name in enumerate(active_names)}
        self.arm_joint_indices = [name_to_index[name] for name in self.arm_joint_names]
        self.total_action_dim = int(np.prod(self.env.action_space.shape))

        segments = self.trajectory["segments"]
        self.to_pregrasp_qpos = np.asarray(segments["to_pregrasp"]["arm_qpos"], dtype=np.float64)
        self.descend_qpos = np.asarray(segments["descend"]["arm_qpos"], dtype=np.float64)
        self.lift_qpos = np.asarray(segments["lift"]["arm_qpos"], dtype=np.float64)
        self.transport_qpos = np.asarray(segments["transport"]["arm_qpos"], dtype=np.float64)
        self.descend_place_qpos = np.asarray(segments["descend_place"]["arm_qpos"], dtype=np.float64)
        self.retreat_qpos = np.asarray(segments["retreat"]["arm_qpos"], dtype=np.float64)

        self.home_arm_qpos = np.asarray(self.trajectory["home_arm_qpos"], dtype=np.float64)
        self.grasp_arm_qpos = self.descend_qpos[-1].copy()
        self.place_arm_qpos = self.descend_place_qpos[-1].copy()
        self.retreat_arm_qpos = self.retreat_qpos[-1].copy()

        self.control_steps = int(self.pp["control_steps_per_waypoint"])
        self.pregrasp_hold_steps = int(self.pp["pregrasp_hold_steps"])
        self.grasp_ramp_steps = int(self.pp["grasp_ramp_steps"])
        self.bilateral_required = int(self.pp["bilateral_contact_streak"])
        self.squeeze_margin = float(self.pp["squeeze_margin_action"])
        self.stable_grasp_force = float(self.gripper_cfg["stable_grasp_force_n"])
        self.pre_lift_hold_steps = int(self.pp["pre_lift_hold_steps"])
        self.pre_lift_required_ratio = float(self.pp["pre_lift_required_contact_ratio"])
        self.release_ramp_steps = int(self.pp["release_ramp_steps"])
        self.release_hold_steps = int(self.pp["release_hold_steps"])
        self.final_settle_steps = int(self.pp["final_settle_steps"])
        self.reset_precontact_steps = int(self.pp["reset_precontact_steps"])
        self.success_lift_height = float(self.pp["success_lift_height_m"])
        self.minimum_lift_after_lift = float(
            self.residual_cfg.get("minimum_lift_after_lift_m", 0.040)
        )
        self.transport_drop_threshold = float(
            self.residual_cfg.get("transport_drop_threshold_m", 0.020)
        )

        self.pre_contact_action = float(self.gripper_cfg["pre_contact_action"])
        self.close_action = float(self.gripper_cfg["close_action"])
        self.open_action = float(self.gripper_cfg["open_action"])
        self.contact_threshold = float(self.gripper_cfg["contact_force_threshold_n"])

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.arm_dof,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBSERVATION_DIM,), dtype=np.float32
        )

        self.observation_layout = {
            "arm_qpos": [0, 6],
            "arm_qvel": [6, 12],
            "grasp_center": [12, 15],
            "cube_position": [15, 18],
            "goal_position": [18, 21],
            "reference_qpos": [21, 27],
            "reference_error": [27, 33],
            "cube_to_grasp": [33, 36],
            "cube_to_goal": [36, 39],
            "grasp_to_goal": [39, 42],
            "phase_onehot": [42, 52],
            "contact_flags": [52, 54],
            "cube_lift": [54, 55],
            "phase_progress": [55, 56],
        }

        self.reward_model = ResidualPickPlaceReward(
            config=self.config.get("reward_pick_place", {}),
            observation_layout=self.observation_layout,
            alpha=self.alpha,
        )

        self.phase = self.PHASE_TO_PREGRASP
        self.phase_step = 0
        self.episode_step = 0
        self.bilateral_streak = 0
        self.stable_both_steps = 0
        self.hold_gripper_action = self.pre_contact_action
        self.cube_initial_position = np.zeros(3, dtype=np.float64)
        self.goal_episode_position = np.zeros(3, dtype=np.float64)
        self.max_cube_lift_m = 0.0
        self.min_cube_lift_transport_m = float("inf")
        self.terminal_success = False
        self.terminal_reason = None
        self.last_metrics = {
            "left_force": 0.0,
            "right_force": 0.0,
            "left_contact": False,
            "right_contact": False,
        }

    def _arm_qpos(self):
        return first_env(self.robot.get_qpos()).astype(np.float64)[self.arm_joint_indices]

    def _arm_qvel(self):
        return first_env(self.robot.get_qvel()).astype(np.float64)[self.arm_joint_indices]

    def _cube_position(self):
        return first_env(self.base_env.cube.pose.p).astype(np.float64)

    def _goal_position(self):
        return np.asarray(self.base_env.goal_position, dtype=np.float64).copy()

    def _grasp_center(self):
        left = np.stack(
            [first_env(link.pose.p) for link in self.base_env.left_contact_links], axis=0
        ).astype(np.float64).mean(axis=0)
        right = np.stack(
            [first_env(link.pose.p) for link in self.base_env.right_contact_links], axis=0
        ).astype(np.float64).mean(axis=0)
        return (left + right) / 2.0

    def _extract_metrics(self, info=None):
        data = {} if info is None else dict(info)
        required = {"left_contact_force", "right_contact_force"}
        if not required.issubset(data):
            data.update(self.base_env.evaluate())
        lf = scalar_first(data["left_contact_force"])
        rf = scalar_first(data["right_contact_force"])
        metrics = {
            "left_force": lf,
            "right_force": rf,
            "left_contact": lf >= self.contact_threshold,
            "right_contact": rf >= self.contact_threshold,
        }
        return metrics

    def _movement_target(self, path: np.ndarray, phase_step: int):
        index = min(phase_step // self.control_steps, len(path) - 1)
        return path[index]

    def _phase_total_steps(self, phase: int):
        if phase == self.PHASE_TO_PREGRASP:
            return len(self.to_pregrasp_qpos) * self.control_steps + self.pregrasp_hold_steps
        if phase == self.PHASE_DESCEND:
            return len(self.descend_qpos) * self.control_steps
        if phase == self.PHASE_GRASP:
            return self.grasp_ramp_steps
        if phase == self.PHASE_STABLE_HOLD:
            return self.pre_lift_hold_steps
        if phase == self.PHASE_LIFT:
            return len(self.lift_qpos) * self.control_steps
        if phase == self.PHASE_TRANSPORT:
            return len(self.transport_qpos) * self.control_steps
        if phase == self.PHASE_DESCEND_PLACE:
            return len(self.descend_place_qpos) * self.control_steps
        if phase == self.PHASE_RELEASE:
            return self.release_ramp_steps + self.release_hold_steps
        if phase == self.PHASE_FINAL_SETTLE:
            return len(self.retreat_qpos) * self.control_steps + self.final_settle_steps
        return 1

    def _phase_progress(self):
        if self.phase == self.PHASE_TERMINAL:
            return 1.0
        total = max(self._phase_total_steps(self.phase), 1)
        return float(np.clip(self.phase_step / max(total - 1, 1), 0.0, 1.0))

    def _current_reference_qpos(self):
        if self.phase == self.PHASE_TO_PREGRASP:
            movement = len(self.to_pregrasp_qpos) * self.control_steps
            if self.phase_step >= movement:
                return self.to_pregrasp_qpos[-1]
            return self._movement_target(self.to_pregrasp_qpos, self.phase_step)
        if self.phase == self.PHASE_DESCEND:
            return self._movement_target(self.descend_qpos, self.phase_step)
        if self.phase in {self.PHASE_GRASP, self.PHASE_STABLE_HOLD}:
            return self.grasp_arm_qpos
        if self.phase == self.PHASE_LIFT:
            return self._movement_target(self.lift_qpos, self.phase_step)
        if self.phase == self.PHASE_TRANSPORT:
            return self._movement_target(self.transport_qpos, self.phase_step)
        if self.phase == self.PHASE_DESCEND_PLACE:
            return self._movement_target(self.descend_place_qpos, self.phase_step)
        if self.phase == self.PHASE_RELEASE:
            return self.place_arm_qpos
        if self.phase == self.PHASE_FINAL_SETTLE:
            movement = len(self.retreat_qpos) * self.control_steps
            if self.phase_step >= movement:
                return self.retreat_arm_qpos
            return self._movement_target(self.retreat_qpos, self.phase_step)
        return self.retreat_arm_qpos

    def _current_gripper_action(self):
        if self.phase in {self.PHASE_TO_PREGRASP, self.PHASE_DESCEND}:
            return self.pre_contact_action
        if self.phase == self.PHASE_GRASP:
            ratio = float(
                np.clip(
                    self.phase_step / max(self.grasp_ramp_steps - 1, 1), 0.0, 1.0
                )
            )
            return float(
                self.pre_contact_action
                + ratio * (self.close_action - self.pre_contact_action)
            )
        if self.phase in {
            self.PHASE_STABLE_HOLD,
            self.PHASE_LIFT,
            self.PHASE_TRANSPORT,
            self.PHASE_DESCEND_PLACE,
        }:
            return float(self.hold_gripper_action)
        if self.phase == self.PHASE_RELEASE:
            if self.phase_step >= self.release_ramp_steps:
                return self.open_action
            ratio = float(
                np.clip(
                    self.phase_step / max(self.release_ramp_steps - 1, 1), 0.0, 1.0
                )
            )
            return float(
                self.hold_gripper_action
                + ratio * (self.open_action - self.hold_gripper_action)
            )
        return self.open_action

    def _compose_arm_action(self, reference_qpos, residual_action):
        current = self._arm_qpos()
        reference_action = np.clip(
            (np.asarray(reference_qpos, dtype=np.float64) - current) / self.max_joint_delta,
            -1.0,
            1.0,
        )
        residual = np.clip(np.asarray(residual_action, dtype=np.float64), -1.0, 1.0)
        combined = np.clip(reference_action + self.alpha * residual, -1.0, 1.0)
        return reference_action, residual, combined

    def _execute_low_level(self, reference_qpos, residual_action, gripper_action):
        reference_action, residual, combined = self._compose_arm_action(
            reference_qpos, residual_action
        )
        full_action = np.zeros(self.total_action_dim, dtype=np.float32)
        full_action[self.arm_slice] = combined.astype(np.float32)
        full_action[self.gripper_slice] = float(gripper_action)
        _, _, base_terminated, base_truncated, info = self.env.step(full_action)
        metrics = self._extract_metrics(info)
        info = dict(info)
        info.update(
            {
                "reference_qpos": np.asarray(reference_qpos, dtype=np.float64).copy(),
                "reference_action": reference_action.copy(),
                "residual_action": residual.copy(),
                "executed_arm_action": combined.copy(),
                "gripper_action": float(gripper_action),
                "alpha": float(self.alpha),
                "base_terminated": bool(base_terminated),
                "base_truncated": bool(base_truncated),
            }
        )
        return info, metrics, bool(base_terminated), bool(base_truncated)

    def _set_terminal(self, reason: str, success: bool):
        self.phase = self.PHASE_TERMINAL
        self.phase_step = 0
        self.terminal_reason = str(reason)
        self.terminal_success = bool(success)

    def _advance_phase(self, metrics, gripper_command, latest_info):
        if self.phase == self.PHASE_TO_PREGRASP:
            self.phase_step += 1
            if self.phase_step >= self._phase_total_steps(self.PHASE_TO_PREGRASP):
                self.phase, self.phase_step = self.PHASE_DESCEND, 0
            return

        if self.phase == self.PHASE_DESCEND:
            self.phase_step += 1
            if self.phase_step >= self._phase_total_steps(self.PHASE_DESCEND):
                self.phase, self.phase_step = self.PHASE_GRASP, 0
                self.bilateral_streak = 0
            return

        if self.phase == self.PHASE_GRASP:
            if metrics["left_contact"] and metrics["right_contact"]:
                self.bilateral_streak += 1
            else:
                self.bilateral_streak = 0
            if self.bilateral_streak >= self.bilateral_required:
                direction = np.sign(self.close_action - self.pre_contact_action)
                self.hold_gripper_action = float(
                    np.clip(gripper_command + direction * self.squeeze_margin, -1.0, 1.0)
                )
                self.phase, self.phase_step = self.PHASE_STABLE_HOLD, 0
                self.stable_both_steps = 0
                return
            self.phase_step += 1
            if self.phase_step >= self.grasp_ramp_steps:
                self._set_terminal("grasp_timeout", False)
            return

        if self.phase == self.PHASE_STABLE_HOLD:
            if (
                metrics["left_force"] >= self.stable_grasp_force
                and metrics["right_force"] >= self.stable_grasp_force
            ):
                self.stable_both_steps += 1
            self.phase_step += 1
            if self.phase_step >= self.pre_lift_hold_steps:
                required = int(math.ceil(self.pre_lift_hold_steps * self.pre_lift_required_ratio))
                if self.stable_both_steps < required:
                    self._set_terminal("unstable_grasp", False)
                else:
                    self.phase, self.phase_step = self.PHASE_LIFT, 0
            return

        if self.phase == self.PHASE_LIFT:
            self.phase_step += 1
            if self.phase_step >= self._phase_total_steps(self.PHASE_LIFT):
                if self.max_cube_lift_m < self.minimum_lift_after_lift:
                    self._set_terminal("insufficient_lift", False)
                else:
                    self.phase, self.phase_step = self.PHASE_TRANSPORT, 0
                    self.min_cube_lift_transport_m = float("inf")
            return

        if self.phase == self.PHASE_TRANSPORT:
            cube_lift = float(self._cube_position()[2] - self.cube_initial_position[2])
            self.min_cube_lift_transport_m = min(self.min_cube_lift_transport_m, cube_lift)
            self.phase_step += 1
            if cube_lift < self.transport_drop_threshold:
                self._set_terminal("drop_during_transport", False)
            elif self.phase_step >= self._phase_total_steps(self.PHASE_TRANSPORT):
                self.phase, self.phase_step = self.PHASE_DESCEND_PLACE, 0
            return

        if self.phase == self.PHASE_DESCEND_PLACE:
            self.phase_step += 1
            if self.phase_step >= self._phase_total_steps(self.PHASE_DESCEND_PLACE):
                self.phase, self.phase_step = self.PHASE_RELEASE, 0
            return

        if self.phase == self.PHASE_RELEASE:
            self.phase_step += 1
            if self.phase_step >= self._phase_total_steps(self.PHASE_RELEASE):
                self.phase, self.phase_step = self.PHASE_FINAL_SETTLE, 0
            return

        if self.phase == self.PHASE_FINAL_SETTLE:
            self.phase_step += 1
            if self.phase_step >= self._phase_total_steps(self.PHASE_FINAL_SETTLE):
                success = bool(latest_info.get("success", False))
                if success:
                    reason = "success"
                elif not bool(latest_info.get("released", False)):
                    reason = "release_failure"
                elif not bool(latest_info.get("within_goal_xy", False)):
                    reason = "place_alignment_failure"
                elif not bool(latest_info.get("within_goal_z", False)):
                    reason = "place_height_failure"
                elif not bool(latest_info.get("is_stable", False)):
                    reason = "unstable_placement"
                else:
                    reason = "insufficient_stable_steps"
                self._set_terminal(reason, success)

    def _get_observation(self, metrics=None):
        metrics = self.last_metrics if metrics is None else metrics
        arm_qpos = self._arm_qpos()
        arm_qvel = self._arm_qvel()
        grasp = self._grasp_center()
        cube = self._cube_position()
        goal = self._goal_position()
        reference = self._current_reference_qpos().copy()
        reference_error = reference - arm_qpos
        cube_to_grasp = cube - grasp
        cube_to_goal = goal - cube
        grasp_to_goal = goal - grasp
        phase_onehot = np.zeros(self.PHASE_COUNT, dtype=np.float64)
        phase_onehot[self.phase] = 1.0
        contact_flags = np.asarray(
            [float(metrics["left_contact"]), float(metrics["right_contact"])],
            dtype=np.float64,
        )
        cube_lift = np.asarray([cube[2] - self.cube_initial_position[2]], dtype=np.float64)
        phase_progress = np.asarray([self._phase_progress()], dtype=np.float64)
        observation = np.concatenate(
            [
                arm_qpos,
                arm_qvel,
                grasp,
                cube,
                goal,
                reference,
                reference_error,
                cube_to_grasp,
                cube_to_goal,
                grasp_to_goal,
                phase_onehot,
                contact_flags,
                cube_lift,
                phase_progress,
            ]
        ).astype(np.float32)
        if observation.shape != (self.OBSERVATION_DIM,):
            raise RuntimeError(f"observation shape mismatch: {observation.shape}")
        return observation

    def _randomization_info(self):
        obj = self.env
        visited = set()
        while obj is not None and id(obj) not in visited:
            visited.add(id(obj))
            if hasattr(obj, "current_randomization_info"):
                return dict(obj.current_randomization_info())
            obj = getattr(obj, "env", None)
        return {}

    def reset(self, *, seed=None, options=None):
        _, base_info = self.env.reset(seed=seed, options=options)
        self.phase = self.PHASE_TO_PREGRASP
        self.phase_step = 0
        self.episode_step = 0
        self.bilateral_streak = 0
        self.stable_both_steps = 0
        self.hold_gripper_action = self.pre_contact_action
        self.terminal_success = False
        self.terminal_reason = None
        self.cube_initial_position = self._cube_position().copy()
        self.goal_episode_position = self._goal_position().copy()
        self.max_cube_lift_m = 0.0
        self.min_cube_lift_transport_m = float("inf")

        zero = np.zeros(self.arm_dof, dtype=np.float32)
        for _ in range(self.reset_precontact_steps):
            _, metrics, _, _ = self._execute_low_level(
                self.home_arm_qpos, zero, self.pre_contact_action
            )
            self.last_metrics = metrics

        observation = self._get_observation(self.last_metrics)
        info = dict(base_info)
        info.update(self._randomization_info())
        info.update(
            {
                "phase": int(self.phase),
                "phase_name": self.PHASE_NAMES[self.phase],
                "reference_trajectory": str(self.trajectory_path),
                "alpha": float(self.alpha),
                "observation_layout": self.observation_layout,
                "observation_dim": self.OBSERVATION_DIM,
                "cube_episode_initial_position_m": self.cube_initial_position.tolist(),
                "goal_episode_position_m": self.goal_episode_position.tolist(),
            }
        )
        return observation, info

    def step(self, residual_action):
        if self.phase == self.PHASE_TERMINAL:
            raise RuntimeError("episode finished; call reset()")
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(-1)
        if residual_action.shape != (self.arm_dof,):
            raise ValueError(f"residual action shape mismatch: {residual_action.shape}")
        if not np.all(np.isfinite(residual_action)):
            raise ValueError("residual action contains NaN/inf")
        residual_action = np.clip(residual_action, -1.0, 1.0)

        previous_observation = self._get_observation(self.last_metrics).copy()
        phase_before = int(self.phase)
        phase_name_before = self.PHASE_NAMES[self.phase]
        reference_qpos = self._current_reference_qpos().copy()
        gripper_action = self._current_gripper_action()

        info, metrics, base_terminated, base_truncated = self._execute_low_level(
            reference_qpos, residual_action, gripper_action
        )
        self.last_metrics = metrics
        self.episode_step += 1
        cube_lift = float(self._cube_position()[2] - self.cube_initial_position[2])
        self.max_cube_lift_m = max(self.max_cube_lift_m, cube_lift)

        # Preserve the base task signals produced at this low-level step.
        task_info = dict(info)
        self._advance_phase(metrics, gripper_action, task_info)

        if base_truncated and self.phase != self.PHASE_TERMINAL:
            self._set_terminal("base_env_truncated", False)

        observation = self._get_observation(metrics)
        terminated = self.phase == self.PHASE_TERMINAL
        truncated = False

        info.update(self._randomization_info())
        info.update(
            {
                "phase_before": phase_before,
                "phase_name_before": phase_name_before,
                "phase": int(self.phase),
                "phase_name": self.PHASE_NAMES[self.phase],
                "phase_step": int(self.phase_step),
                "episode_step": int(self.episode_step),
                "left_contact_force": float(metrics["left_force"]),
                "right_contact_force": float(metrics["right_force"]),
                "left_contact": bool(metrics["left_contact"]),
                "right_contact": bool(metrics["right_contact"]),
                "stable_both_steps": int(self.stable_both_steps),
                "cube_lift_m": cube_lift,
                "max_cube_lift_m": float(self.max_cube_lift_m),
                "min_cube_lift_transport_m": (
                    None
                    if np.isinf(self.min_cube_lift_transport_m)
                    else float(self.min_cube_lift_transport_m)
                ),
                "success": bool(self.terminal_success),
                "terminal_reason": self.terminal_reason,
                "cube_episode_initial_position_m": self.cube_initial_position.tolist(),
                "goal_episode_position_m": self.goal_episode_position.tolist(),
                "goal_xy_error_m": scalar_first(task_info.get("goal_xy_error_m", np.nan)),
                "goal_z_error_m": scalar_first(task_info.get("goal_z_error_m", np.nan)),
                "within_goal_xy": bool(scalar_first(task_info.get("within_goal_xy", False))),
                "within_goal_z": bool(scalar_first(task_info.get("within_goal_z", False))),
                "released": bool(scalar_first(task_info.get("released", False))),
                "is_stable": bool(scalar_first(task_info.get("is_stable", False))),
                "stable_place_steps": int(scalar_first(task_info.get("stable_place_steps", 0))),
            }
        )

        reward, reward_terms = self.reward_model.compute(
            previous_observation=previous_observation,
            observation=observation,
            residual_action=residual_action,
            info=info,
            terminated=terminated,
        )
        info["reward_terms"] = reward_terms
        return observation, reward, terminated, truncated, info

    def observation_contract(self):
        return {
            "dimension": self.OBSERVATION_DIM,
            "layout": self.observation_layout,
            "phase_names": list(self.PHASE_NAMES),
            "raw_units": {
                "positions": "m",
                "joint_positions": "rad",
                "joint_velocities": "rad/s",
            },
            "note": "Reward uses raw observation. TD3-only scaling must be an outer wrapper.",
        }
