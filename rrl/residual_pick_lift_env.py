#!/usr/bin/env python3
"""
UR3e + EZGripper Residual Pick-and-Lift Environment.

外部Action:
    residual_action [6], normalized [-1, 1]

内部:
    Day6 Reference qpos
        ↓
    reference action
        ↓
    Day8 ResidualActionComposer
        ↓
    a_exec = clip(a_ref + alpha * a_res)
        ↓
    scripted gripper action
        ↓
    ManiSkill UR3ePickLift-v0

Day9ではrewardはまだ実装しない。
常に0.0を返し、Day10で報酬を追加する。
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

from rrl.residual_action import (
    ResidualActionComposer,
)


# ==========================================================
# Utilities
# ==========================================================

def to_numpy(value: Any) -> np.ndarray:

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def first_env(value: Any) -> np.ndarray:

    array = to_numpy(value)

    if (
        array.ndim >= 2
        and array.shape[0] == 1
    ):
        return array[0]

    return array


def scalar_first(value: Any) -> float:

    array = np.asarray(
        first_env(value)
    )

    if array.size != 1:
        raise ValueError(
            "scalarを想定しました: "
            f"shape={array.shape}"
        )

    return float(
        array.reshape(-1)[0]
    )


def get_controller_space(controller):

    if hasattr(
        controller,
        "single_action_space",
    ):
        return controller.single_action_space

    return controller.action_space


def build_controller_slices(
    controllers: Mapping,
):

    result = {}

    offset = 0

    for name, controller in (
        controllers.items()
    ):

        space = get_controller_space(
            controller
        )

        dim = int(
            np.prod(
                space.shape
            )
        )

        result[str(name)] = slice(
            offset,
            offset + dim,
        )

        offset += dim

    return result


def find_controller(
    controllers,
    keyword: str,
):

    for name in controllers:

        if keyword in str(
            name
        ).lower():

            return str(name)

    raise RuntimeError(
        f"{keyword} controllerがありません: "
        f"{list(controllers.keys())}"
    )


def get_controller_joint_names(
    controller,
):

    names = getattr(
        controller.config,
        "joint_names",
        None,
    )

    if names is None:
        return []

    return [
        str(name)
        for name in names
    ]


# ==========================================================
# Residual Environment
# ==========================================================

class ResidualPickLiftEnv(
    gym.Wrapper
):

    """
    TD3から見る環境。

    Action:
        residual action [6]

    Observation:
        44 dimensions

    Reward:
        Day9では0
        Day10で実装
    """

    PHASE_TO_PREGRASP = 0
    PHASE_DESCEND = 1
    PHASE_GRASP = 2
    PHASE_STABLE_HOLD = 3
    PHASE_LIFT = 4
    PHASE_FINAL_HOLD = 5
    PHASE_TERMINAL = 6

    PHASE_NAMES = [
        "to_pregrasp",
        "descend",
        "grasp",
        "stable_hold",
        "lift",
        "final_hold",
        "terminal",
    ]

    PHASE_COUNT = len(
        PHASE_NAMES
    )

    OBSERVATION_DIM = 44

    def __init__(
        self,
        env: gym.Env,
        trajectory_path: str | Path,
        config_path: str | Path,
        alpha: float | None = None,
    ):

        super().__init__(
            env
        )

        self.trajectory_path = Path(
            trajectory_path
        )

        self.config_path = Path(
            config_path
        )

        # --------------------------------------------------
        # Config
        # --------------------------------------------------

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.config = yaml.safe_load(
                file
            )

        with self.trajectory_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.trajectory = json.load(
                file
            )

        self.day6 = self.config[
            "day6"
        ]

        self.day9 = self.config.get(
            "day9",
            {},
        )

        self.gripper_cfg = (
            self.config[
                "gripper"
            ]
        )

        self.alpha = (
            float(alpha)
            if alpha is not None
            else float(
                self.config[
                    "residual"
                ][
                    "alpha"
                ]
            )
        )

        # --------------------------------------------------
        # ManiSkill
        # --------------------------------------------------

        self.base_env = (
            self.env.unwrapped
        )

        self.robot = (
            self.base_env.agent.robot
        )

        combined_controller = (
            self.base_env
            .agent
            .controller
        )

        controllers = (
            combined_controller.controllers
        )

        if not isinstance(
            controllers,
            Mapping,
        ):

            raise RuntimeError(
                "CombinedControllerではありません"
            )

        slices = build_controller_slices(
            controllers
        )

        self.arm_name = (
            find_controller(
                controllers,
                "arm",
            )
        )

        self.gripper_name = (
            find_controller(
                controllers,
                "gripper",
            )
        )

        self.arm_controller = (
            controllers[
                self.arm_name
            ]
        )

        self.arm_slice = slices[
            self.arm_name
        ]

        self.gripper_slice = slices[
            self.gripper_name
        ]

        self.arm_joint_names = (
            get_controller_joint_names(
                self.arm_controller
            )
        )

        self.arm_dof = len(
            self.arm_joint_names
        )

        if self.arm_dof != 6:

            raise RuntimeError(
                "UR3e arm DoF=6を想定しています。"
                f" actual={self.arm_dof}"
            )

        trajectory_joint_names = (
            self.trajectory[
                "arm_joint_names"
            ]
        )

        if (
            trajectory_joint_names
            != self.arm_joint_names
        ):

            raise RuntimeError(
                "Reference trajectoryと"
                "現在のArm joint順序が一致しません"
            )

        active_joint_names = [
            joint.name
            for joint
            in self.robot.active_joints
        ]

        name_to_index = {
            name: index
            for index, name
            in enumerate(
                active_joint_names
            )
        }

        self.arm_joint_indices = [
            name_to_index[name]
            for name
            in self.arm_joint_names
        ]

        self.total_action_dim = int(
            np.prod(
                self.env.action_space.shape
            )
        )

        # --------------------------------------------------
        # Day8 Residual Composer
        # --------------------------------------------------

        controller_cfg = (
            self.arm_controller.config
        )

        self.composer = (
            ResidualActionComposer(
                physical_lower=(
                    controller_cfg.lower
                ),
                physical_upper=(
                    controller_cfg.upper
                ),
                alpha=self.alpha,
                dof=self.arm_dof,
            )
        )

        # --------------------------------------------------
        # Reference trajectory
        # --------------------------------------------------

        segments = self.trajectory[
            "segments"
        ]

        self.to_pregrasp_qpos = np.asarray(
            segments[
                "to_pregrasp"
            ][
                "arm_qpos"
            ],
            dtype=np.float64,
        )

        self.descend_qpos = np.asarray(
            segments[
                "descend"
            ][
                "arm_qpos"
            ],
            dtype=np.float64,
        )

        self.lift_qpos = np.asarray(
            segments[
                "lift"
            ][
                "arm_qpos"
            ],
            dtype=np.float64,
        )

        self.home_arm_qpos = np.asarray(
            self.trajectory[
                "home_arm_qpos"
            ],
            dtype=np.float64,
        )

        self.grasp_arm_qpos = (
            self.descend_qpos[
                -1
            ].copy()
        )

        self.lift_arm_qpos = (
            self.lift_qpos[
                -1
            ].copy()
        )

        # --------------------------------------------------
        # Script control parameters
        # --------------------------------------------------

        self.control_steps = int(
            self.day6[
                "control_steps_per_waypoint"
            ]
        )

        self.pregrasp_hold_steps = int(
            self.day6.get(
                "pregrasp_hold_steps",
                0,
            )
        )

        self.grasp_ramp_steps = int(
            self.day6[
                "grasp_ramp_steps"
            ]
        )

        self.bilateral_required = int(
            self.day6[
                "bilateral_contact_streak"
            ]
        )

        self.squeeze_margin = float(
            self.day6.get(
                "squeeze_margin_action",
                0.0,
            )
        )

        self.stable_grasp_force = float(
            self.day6.get(
                "stable_grasp_force_n",
                5.0,
            )
        )

        self.pre_lift_hold_steps = int(
            self.day6.get(
                "pre_lift_hold_steps",
                30,
            )
        )

        self.pre_lift_required_ratio = float(
            self.day6.get(
                "pre_lift_required_contact_ratio",
                0.8,
            )
        )

        self.final_hold_steps = int(
            self.day6[
                "final_hold_steps"
            ]
        )

        self.success_lift_height = float(
            self.day6[
                "success_lift_height_m"
            ]
        )

        self.success_hold_steps = int(
            self.day6[
                "success_hold_steps"
            ]
        )

        self.pre_contact_action = float(
            self.gripper_cfg[
                "pre_contact_action"
            ]
        )

        self.close_action = float(
            self.gripper_cfg[
                "close_action"
            ]
        )

        self.contact_threshold = float(
            self.gripper_cfg[
                "contact_force_threshold_n"
            ]
        )

        self.reset_precontact_steps = int(
            self.day9.get(
                "reset_precontact_steps",
                60,
            )
        )

        # --------------------------------------------------
        # RL spaces
        # --------------------------------------------------

        # TD3はUR3e 6軸Residualのみ出力。
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                self.arm_dof,
            ),
            dtype=np.float32,
        )

        self.observation_space = (
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.OBSERVATION_DIM,
                ),
                dtype=np.float32,
            )
        )

        # --------------------------------------------------
        # Observation contract
        # --------------------------------------------------

        self.observation_layout = {
            "arm_qpos": [0, 6],
            "arm_qvel": [6, 12],
            "grasp_center": [12, 15],
            "cube_position": [15, 18],
            "reference_qpos": [18, 24],
            "reference_error": [24, 30],
            "cube_to_grasp": [30, 33],
            "phase_onehot": [33, 40],
            "contact_flags": [40, 42],
            "cube_lift": [42, 43],
            "phase_progress": [43, 44],
        }

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------

        self.phase = (
            self.PHASE_TO_PREGRASP
        )

        self.phase_step = 0

        self.episode_step = 0

        self.bilateral_streak = 0

        self.stable_both_steps = 0

        self.final_both_contact_steps = 0

        self.hold_gripper_action = (
            self.pre_contact_action
        )

        self.cube_initial_position = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

        self.terminal_success = False
        self.terminal_reason = None

        self.last_metrics = {
            "left_force": 0.0,
            "right_force": 0.0,
            "left_contact": False,
            "right_contact": False,
        }

    # ======================================================
    # Robot state
    # ======================================================

    def _arm_qpos(
        self,
    ) -> np.ndarray:

        qpos = first_env(
            self.robot.get_qpos()
        ).astype(
            np.float64
        )

        return qpos[
            self.arm_joint_indices
        ]

    def _arm_qvel(
        self,
    ) -> np.ndarray:

        qvel = first_env(
            self.robot.get_qvel()
        ).astype(
            np.float64
        )

        return qvel[
            self.arm_joint_indices
        ]

    def _cube_position(
        self,
    ) -> np.ndarray:

        return first_env(
            self.base_env.cube.pose.p
        ).astype(
            np.float64
        )

    def _grasp_center(
        self,
    ) -> np.ndarray:

        left_positions = np.stack(
            [
                first_env(
                    link.pose.p
                )
                for link
                in self.base_env.left_contact_links
            ],
            axis=0,
        ).astype(
            np.float64
        )

        right_positions = np.stack(
            [
                first_env(
                    link.pose.p
                )
                for link
                in self.base_env.right_contact_links
            ],
            axis=0,
        ).astype(
            np.float64
        )

        left_center = (
            left_positions.mean(
                axis=0
            )
        )

        right_center = (
            right_positions.mean(
                axis=0
            )
        )

        return (
            left_center
            + right_center
        ) / 2.0

    # ======================================================
    # Contact
    # ======================================================

    def _extract_metrics(
        self,
        info=None,
    ):

        data = {}

        if info is not None:
            data.update(
                dict(info)
            )

        required = (
            "left_contact_force",
            "right_contact_force",
        )

        if not all(
            key in data
            for key in required
        ):

            evaluate_info = (
                self.base_env.evaluate()
            )

            data.update(
                evaluate_info
            )

        left_force = scalar_first(
            data[
                "left_contact_force"
            ]
        )

        right_force = scalar_first(
            data[
                "right_contact_force"
            ]
        )

        return {
            "left_force":
                left_force,

            "right_force":
                right_force,

            "left_contact":
                (
                    left_force
                    >= self.contact_threshold
                ),

            "right_contact":
                (
                    right_force
                    >= self.contact_threshold
                ),
        }

    # ======================================================
    # Reference
    # ======================================================

    def _movement_target(
        self,
        path: np.ndarray,
        phase_step: int,
    ) -> np.ndarray:

        waypoint_index = min(
            phase_step
            // self.control_steps,
            len(path) - 1,
        )

        return path[
            waypoint_index
        ]

    def _current_reference_qpos(
        self,
    ) -> np.ndarray:

        if (
            self.phase
            == self.PHASE_TO_PREGRASP
        ):

            movement_steps = (
                len(
                    self.to_pregrasp_qpos
                )
                * self.control_steps
            )

            if (
                self.phase_step
                >= movement_steps
            ):

                return (
                    self.to_pregrasp_qpos[
                        -1
                    ]
                )

            return self._movement_target(
                self.to_pregrasp_qpos,
                self.phase_step,
            )

        if (
            self.phase
            == self.PHASE_DESCEND
        ):

            return self._movement_target(
                self.descend_qpos,
                self.phase_step,
            )

        if self.phase in (
            self.PHASE_GRASP,
            self.PHASE_STABLE_HOLD,
        ):

            return (
                self.grasp_arm_qpos
            )

        if (
            self.phase
            == self.PHASE_LIFT
        ):

            return self._movement_target(
                self.lift_qpos,
                self.phase_step,
            )

        return self.lift_arm_qpos

    def _phase_total_steps(
        self,
        phase: int,
    ) -> int:

        if (
            phase
            == self.PHASE_TO_PREGRASP
        ):

            return (
                len(
                    self.to_pregrasp_qpos
                )
                * self.control_steps
                + self.pregrasp_hold_steps
            )

        if (
            phase
            == self.PHASE_DESCEND
        ):

            return (
                len(
                    self.descend_qpos
                )
                * self.control_steps
            )

        if (
            phase
            == self.PHASE_GRASP
        ):
            return self.grasp_ramp_steps

        if (
            phase
            == self.PHASE_STABLE_HOLD
        ):
            return self.pre_lift_hold_steps

        if (
            phase
            == self.PHASE_LIFT
        ):

            return (
                len(
                    self.lift_qpos
                )
                * self.control_steps
            )

        if (
            phase
            == self.PHASE_FINAL_HOLD
        ):
            return self.final_hold_steps

        return 1

    def _phase_progress(
        self,
    ) -> float:

        if (
            self.phase
            == self.PHASE_TERMINAL
        ):
            return 1.0

        total = max(
            self._phase_total_steps(
                self.phase
            ),
            1,
        )

        return float(
            np.clip(
                self.phase_step
                / max(
                    total - 1,
                    1,
                ),
                0.0,
                1.0,
            )
        )

    # ======================================================
    # Gripper script
    # ======================================================

    def _current_gripper_action(
        self,
    ) -> float:

        if self.phase in (
            self.PHASE_TO_PREGRASP,
            self.PHASE_DESCEND,
        ):

            return (
                self.pre_contact_action
            )

        if (
            self.phase
            == self.PHASE_GRASP
        ):

            ratio = float(
                np.clip(
                    self.phase_step
                    / max(
                        self.grasp_ramp_steps
                        - 1,
                        1,
                    ),
                    0.0,
                    1.0,
                )
            )

            return float(
                self.pre_contact_action
                + ratio
                * (
                    self.close_action
                    - self.pre_contact_action
                )
            )

        return float(
            self.hold_gripper_action
        )

    # ======================================================
    # Low-level action
    # ======================================================

    def _execute_low_level(
        self,
        reference_qpos,
        residual_action,
        gripper_action,
    ):

        current_qpos = (
            self._arm_qpos()
        )

        composed = (
            self.composer.compose_from_qpos(
                current_qpos=current_qpos,
                target_qpos=reference_qpos,
                residual_action=residual_action,
            )
        )

        full_action = np.zeros(
            self.total_action_dim,
            dtype=np.float32,
        )

        full_action[
            self.arm_slice
        ] = composed[
            "combined_action"
        ]

        full_action[
            self.gripper_slice
        ] = gripper_action

        (
            _base_obs,
            _base_reward,
            base_terminated,
            base_truncated,
            info,
        ) = self.env.step(
            full_action
        )

        metrics = (
            self._extract_metrics(
                info
            )
        )

        info = dict(
            info
        )

        info.update(
            {
                "reference_qpos":
                    np.asarray(
                        reference_qpos
                    ).copy(),

                "reference_action":
                    composed[
                        "reference_action"
                    ].copy(),

                "residual_action":
                    composed[
                        "residual_action"
                    ].copy(),

                "executed_arm_action":
                    composed[
                        "combined_action"
                    ].copy(),

                "gripper_action":
                    float(
                        gripper_action
                    ),

                "alpha":
                    float(
                        self.alpha
                    ),

                "base_terminated":
                    bool(
                        base_terminated
                    ),

                "base_truncated":
                    bool(
                        base_truncated
                    ),
            }
        )

        return (
            info,
            metrics,
            bool(base_terminated),
            bool(base_truncated),
        )

    # ======================================================
    # Phase transition
    # ======================================================

    def _set_terminal(
        self,
        reason: str,
        success: bool,
    ):

        self.phase = (
            self.PHASE_TERMINAL
        )

        self.phase_step = 0

        self.terminal_reason = str(
            reason
        )

        self.terminal_success = bool(
            success
        )

    def _advance_phase(
        self,
        metrics,
        gripper_command,
    ):

        # --------------------------------------------------
        # To pre-grasp
        # --------------------------------------------------

        if (
            self.phase
            == self.PHASE_TO_PREGRASP
        ):

            self.phase_step += 1

            if (
                self.phase_step
                >= self._phase_total_steps(
                    self.PHASE_TO_PREGRASP
                )
            ):

                self.phase = (
                    self.PHASE_DESCEND
                )

                self.phase_step = 0

            return

        # --------------------------------------------------
        # Descend
        # --------------------------------------------------

        if (
            self.phase
            == self.PHASE_DESCEND
        ):

            self.phase_step += 1

            if (
                self.phase_step
                >= self._phase_total_steps(
                    self.PHASE_DESCEND
                )
            ):

                self.phase = (
                    self.PHASE_GRASP
                )

                self.phase_step = 0
                self.bilateral_streak = 0

            return

        # --------------------------------------------------
        # Grasp
        # --------------------------------------------------

        if (
            self.phase
            == self.PHASE_GRASP
        ):

            if (
                metrics[
                    "left_contact"
                ]
                and metrics[
                    "right_contact"
                ]
            ):

                self.bilateral_streak += 1

            else:

                self.bilateral_streak = 0

            if (
                self.bilateral_streak
                >= self.bilateral_required
            ):

                direction = np.sign(
                    self.close_action
                    - self.pre_contact_action
                )

                self.hold_gripper_action = float(
                    np.clip(
                        gripper_command
                        + direction
                        * self.squeeze_margin,
                        -1.0,
                        1.0,
                    )
                )

                self.phase = (
                    self.PHASE_STABLE_HOLD
                )

                self.phase_step = 0
                self.stable_both_steps = 0

                return

            self.phase_step += 1

            if (
                self.phase_step
                >= self.grasp_ramp_steps
            ):

                self._set_terminal(
                    "grasp_timeout",
                    False,
                )

            return

        # --------------------------------------------------
        # Stable Hold
        # --------------------------------------------------

        if (
            self.phase
            == self.PHASE_STABLE_HOLD
        ):

            left_stable = (
                metrics[
                    "left_force"
                ]
                >= self.stable_grasp_force
            )

            right_stable = (
                metrics[
                    "right_force"
                ]
                >= self.stable_grasp_force
            )

            if (
                left_stable
                and right_stable
            ):

                self.stable_both_steps += 1

            self.phase_step += 1

            if (
                self.phase_step
                >= self.pre_lift_hold_steps
            ):

                required = int(
                    math.ceil(
                        self.pre_lift_hold_steps
                        * self.pre_lift_required_ratio
                    )
                )

                if (
                    self.stable_both_steps
                    < required
                ):

                    self._set_terminal(
                        "unstable_grasp",
                        False,
                    )

                    return

                self.phase = (
                    self.PHASE_LIFT
                )

                self.phase_step = 0

            return

        # --------------------------------------------------
        # Lift
        # --------------------------------------------------

        if (
            self.phase
            == self.PHASE_LIFT
        ):

            self.phase_step += 1

            if (
                self.phase_step
                >= self._phase_total_steps(
                    self.PHASE_LIFT
                )
            ):

                self.phase = (
                    self.PHASE_FINAL_HOLD
                )

                self.phase_step = 0

                self.final_both_contact_steps = 0

            return

        # --------------------------------------------------
        # Final Hold
        # --------------------------------------------------

        if (
            self.phase
            == self.PHASE_FINAL_HOLD
        ):

            if (
                metrics[
                    "left_contact"
                ]
                and metrics[
                    "right_contact"
                ]
            ):

                self.final_both_contact_steps += 1

            self.phase_step += 1

            if (
                self.phase_step
                >= self.final_hold_steps
            ):

                cube_position = (
                    self._cube_position()
                )

                cube_lift = float(
                    cube_position[2]
                    - self.cube_initial_position[2]
                )

                success = (
                    cube_lift
                    >= self.success_lift_height
                    and
                    self.final_both_contact_steps
                    >= self.success_hold_steps
                )

                self._set_terminal(
                    "schedule_complete",
                    success,
                )

    # ======================================================
    # Observation
    # ======================================================

    def _get_observation(
        self,
        metrics=None,
    ) -> np.ndarray:

        if metrics is None:
            metrics = (
                self.last_metrics
            )

        arm_qpos = (
            self._arm_qpos()
        )

        arm_qvel = (
            self._arm_qvel()
        )

        grasp_center = (
            self._grasp_center()
        )

        cube_position = (
            self._cube_position()
        )

        reference_qpos = (
            self._current_reference_qpos()
        )

        reference_error = (
            reference_qpos
            - arm_qpos
        )

        cube_to_grasp = (
            cube_position
            - grasp_center
        )

        phase_onehot = np.zeros(
            self.PHASE_COUNT,
            dtype=np.float64,
        )

        phase_onehot[
            self.phase
        ] = 1.0

        contact_flags = np.asarray(
            [
                float(
                    metrics[
                        "left_contact"
                    ]
                ),
                float(
                    metrics[
                        "right_contact"
                    ]
                ),
            ],
            dtype=np.float64,
        )

        cube_lift = np.asarray(
            [
                cube_position[2]
                - self.cube_initial_position[2]
            ],
            dtype=np.float64,
        )

        phase_progress = np.asarray(
            [
                self._phase_progress()
            ],
            dtype=np.float64,
        )

        observation = np.concatenate(
            [
                arm_qpos,
                arm_qvel,
                grasp_center,
                cube_position,
                reference_qpos,
                reference_error,
                cube_to_grasp,
                phase_onehot,
                contact_flags,
                cube_lift,
                phase_progress,
            ]
        ).astype(
            np.float32
        )

        if (
            observation.shape
            != (
                self.OBSERVATION_DIM,
            )
        ):

            raise RuntimeError(
                "Observation shape mismatch: "
                f"{observation.shape}"
            )

        return observation

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
            _base_observation,
            base_info,
        ) = self.env.reset(
            seed=seed,
            options=options,
        )

        self.phase = (
            self.PHASE_TO_PREGRASP
        )

        self.phase_step = 0
        self.episode_step = 0

        self.bilateral_streak = 0
        self.stable_both_steps = 0
        self.final_both_contact_steps = 0

        self.hold_gripper_action = (
            self.pre_contact_action
        )

        self.terminal_success = False
        self.terminal_reason = None

        self.cube_initial_position = (
            self._cube_position().copy()
        )

        # --------------------------------------------------
        # Hidden setup:
        # Day6と同様にgripperをpre-contactへ
        # --------------------------------------------------

        zero_residual = np.zeros(
            self.arm_dof,
            dtype=np.float32,
        )

        for _ in range(
            self.reset_precontact_steps
        ):

            (
                _,
                metrics,
                _,
                _,
            ) = self._execute_low_level(
                reference_qpos=(
                    self.home_arm_qpos
                ),
                residual_action=(
                    zero_residual
                ),
                gripper_action=(
                    self.pre_contact_action
                ),
            )

            self.last_metrics = (
                metrics
            )

        observation = (
            self._get_observation(
                self.last_metrics
            )
        )

        info = dict(
            base_info
        )

        info.update(
            {
                "phase":
                    self.phase,

                "phase_name":
                    self.PHASE_NAMES[
                        self.phase
                    ],

                "reference_trajectory":
                    str(
                        self.trajectory_path
                    ),

                "alpha":
                    self.alpha,

                "observation_layout":
                    self.observation_layout,

                "migration_contract":
                    self.migration_contract(),
            }
        )

        return (
            observation,
            info,
        )

    def step(
        self,
        residual_action,
    ):

        if (
            self.phase
            == self.PHASE_TERMINAL
        ):

            raise RuntimeError(
                "Episodeは終了しています。"
                "reset()してください。"
            )

        residual_action = np.asarray(
            residual_action,
            dtype=np.float32,
        ).reshape(-1)

        if (
            residual_action.shape
            != (
                self.arm_dof,
            )
        ):

            raise ValueError(
                "Residual action shape mismatch: "
                f"{residual_action.shape}"
            )

        if not np.all(
            np.isfinite(
                residual_action
            )
        ):

            raise ValueError(
                "Residual actionにNaN/infがあります"
            )

        residual_action = np.clip(
            residual_action,
            -1.0,
            1.0,
        )

        phase_before = int(
            self.phase
        )

        phase_name_before = (
            self.PHASE_NAMES[
                self.phase
            ]
        )

        reference_qpos = (
            self._current_reference_qpos()
            .copy()
        )

        gripper_action = (
            self._current_gripper_action()
        )

        (
            info,
            metrics,
            base_terminated,
            base_truncated,
        ) = self._execute_low_level(
            reference_qpos=(
                reference_qpos
            ),
            residual_action=(
                residual_action
            ),
            gripper_action=(
                gripper_action
            ),
        )

        self.last_metrics = metrics

        self.episode_step += 1

        self._advance_phase(
            metrics,
            gripper_action,
        )

        # base TimeLimitは1000を想定。
        if (
            base_truncated
            and self.phase
            != self.PHASE_TERMINAL
        ):

            self._set_terminal(
                "base_env_truncated",
                False,
            )

        observation = (
            self._get_observation(
                metrics
            )
        )

        cube_position = (
            self._cube_position()
        )

        cube_lift = float(
            cube_position[2]
            - self.cube_initial_position[2]
        )

        info.update(
            {
                "phase_before":
                    phase_before,

                "phase_name_before":
                    phase_name_before,

                "phase":
                    int(
                        self.phase
                    ),

                "phase_name":
                    self.PHASE_NAMES[
                        self.phase
                    ],

                "phase_step":
                    int(
                        self.phase_step
                    ),

                "episode_step":
                    int(
                        self.episode_step
                    ),

                "left_contact_force":
                    metrics[
                        "left_force"
                    ],

                "right_contact_force":
                    metrics[
                        "right_force"
                    ],

                "left_contact":
                    metrics[
                        "left_contact"
                    ],

                "right_contact":
                    metrics[
                        "right_contact"
                    ],

                "stable_both_steps":
                    int(
                        self.stable_both_steps
                    ),

                "final_both_contact_steps":
                    int(
                        self.final_both_contact_steps
                    ),

                "cube_lift_m":
                    cube_lift,

                "success":
                    bool(
                        self.terminal_success
                    ),

                "terminal_reason":
                    self.terminal_reason,
            }
        )

        # Day9ではRewardを入れない。
        reward = 0.0

        terminated = (
            self.phase
            == self.PHASE_TERMINAL
        )

        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ======================================================
    # Reproduction record
    # ======================================================

    def migration_contract(
        self,
    ) -> dict:

        return {
            "source":
                "yaginuma-tracker",

            "source_residual_target":
                "expert trajectory endpoint [dx, dz]",

            "source_action_dim":
                2,

            "source_observation":
                "pose9 + expert onehot13",

            "ported_robot":
                "UR3e + EZGripper",

            "ported_simulator":
                "ManiSkill",

            "ported_reference":
                "Day6 joint-space reference trajectory",

            "ported_residual_target":
                (
                    "6-DoF normalized "
                    "UR3e joint residual action"
                ),

            "ported_action_dim":
                6,

            "ported_phase_encoding":
                "phase onehot7",

            "gripper_control":
                "scripted",

            "alpha":
                float(
                    self.alpha
                ),
        }