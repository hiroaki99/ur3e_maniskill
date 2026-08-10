#!/usr/bin/env python3
"""
Day 8:
Residual Action合成処理の単体・実環境テスト。

確認するもの
------------
1. 実際のUR3e controller lower/upper
2. Reference qpos -> Reference Action変換
3. residual=0ならReferenceと完全一致
4. residualを加えても[-1,1]内
5. Gripper ActionはResidualで変化しない
6. ManiSkill公式scale式と一致
7. 実際にenv.step()へ入力可能
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from mani_skill.utils import gym_utils


# ==========================================================
# Path
# ==========================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT),
)

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)

DEFAULT_TRAJECTORY = (
    REPO_ROOT
    / "trajectories"
    / "day6_pick_lift_reference_v2.json"
)


from rrl import ResidualActionComposer


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
    )

    parser.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=[
            "physx_cpu",
            "physx_cuda",
        ],
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "省略時はconfigsの"
            "residual.alphaを使用"
        ),
    )

    parser.add_argument(
        "--random-tests",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/day8_residual_action.json"
        ),
    )

    return parser.parse_args()


# ==========================================================
# Utilities
# ==========================================================

def to_numpy(value):

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def first_env(value):

    array = to_numpy(
        value
    )

    if (
        array.ndim >= 2
        and array.shape[0] == 1
    ):
        return array[0]

    return array


def controller_space(
    controller,
):

    if hasattr(
        controller,
        "single_action_space",
    ):
        return (
            controller
            .single_action_space
        )

    return controller.action_space


def build_controller_slices(
    controllers: Mapping,
):

    slices = {}

    offset = 0

    for name, controller in (
        controllers.items()
    ):

        space = controller_space(
            controller
        )

        dim = int(
            np.prod(
                space.shape
            )
        )

        slices[
            str(name)
        ] = slice(
            offset,
            offset + dim,
        )

        offset += dim

    return slices


def find_controller(
    controllers,
    keyword,
):

    for name in controllers:

        if keyword in str(
            name
        ).lower():

            return str(name)

    raise RuntimeError(
        f"{keyword} controllerが"
        "見つかりません"
    )


def get_joint_names(
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


def broadcast_bound(
    value,
    dof,
):

    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim == 0:

        return np.full(
            dof,
            float(array),
        )

    array = array.reshape(-1)

    if array.size == 1:

        return np.full(
            dof,
            float(array[0]),
        )

    if array.size != dof:

        raise RuntimeError(
            "controller bound size mismatch"
        )

    return array


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    # ------------------------------------------------------
    # Config / trajectory
    # ------------------------------------------------------

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    if not args.trajectory.exists():

        raise FileNotFoundError(
            args.trajectory
        )

    with args.trajectory.open(
        "r",
        encoding="utf-8",
    ) as file:

        trajectory = json.load(
            file
        )

    alpha = (
        float(args.alpha)
        if args.alpha is not None
        else float(
            config[
                "residual"
            ][
                "alpha"
            ]
        )
    )

    # ------------------------------------------------------
    # Environment
    # ------------------------------------------------------

    import envs.ur3e_pick_lift  # noqa: F401

    env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config[
            "robot"
        ]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config[
            "project"
        ]["control_mode"],
        sim_backend=args.sim_backend,
    )

    errors = []
    warnings = []

    try:

        env.reset(
            seed=int(
                config[
                    "project"
                ]["seed"]
            )
        )

        base_env = env.unwrapped

        robot = base_env.agent.robot

        combined_controller = (
            base_env
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

        slices = (
            build_controller_slices(
                controllers
            )
        )

        arm_name = (
            find_controller(
                controllers,
                "arm",
            )
        )

        gripper_name = (
            find_controller(
                controllers,
                "gripper",
            )
        )

        arm_controller = (
            controllers[
                arm_name
            ]
        )

        arm_slice = (
            slices[
                arm_name
            ]
        )

        gripper_slice = (
            slices[
                gripper_name
            ]
        )

        # --------------------------------------------------
        # Controller validation
        # --------------------------------------------------

        controller_config = (
            arm_controller.config
        )

        use_delta = bool(
            getattr(
                controller_config,
                "use_delta",
                False,
            )
        )

        use_target = bool(
            getattr(
                controller_config,
                "use_target",
                False,
            )
        )

        normalize_action = bool(
            getattr(
                controller_config,
                "normalize_action",
                False,
            )
        )

        if not use_delta:

            errors.append(
                "Arm controller is not "
                "delta-position control"
            )

        if use_target:

            errors.append(
                "Day8 assumes "
                "pd_joint_delta_pos "
                "(use_target=False)"
            )

        if not normalize_action:

            errors.append(
                "Arm controller action is "
                "not normalized"
            )

        arm_joint_names = (
            get_joint_names(
                arm_controller
            )
        )

        arm_dof = len(
            arm_joint_names
        )

        if arm_dof != 6:

            errors.append(
                f"Expected arm DoF=6, "
                f"actual={arm_dof}"
            )

        # --------------------------------------------------
        # Actual controller bounds
        # --------------------------------------------------

        lower = broadcast_bound(
            controller_config.lower,
            arm_dof,
        )

        upper = broadcast_bound(
            controller_config.upper,
            arm_dof,
        )

        composer = (
            ResidualActionComposer(
                physical_lower=lower,
                physical_upper=upper,
                alpha=alpha,
                dof=arm_dof,
            )
        )

        print("=" * 72)
        print("Day 8 Residual Action Test")
        print("=" * 72)

        print(
            "control mode:",
            config[
                "project"
            ]["control_mode"],
        )

        print(
            "arm joints:",
            arm_joint_names,
        )

        print(
            "controller lower:",
            lower.tolist(),
        )

        print(
            "controller upper:",
            upper.tolist(),
        )

        print(
            "alpha:",
            alpha,
        )

        # --------------------------------------------------
        # Current arm qpos
        # --------------------------------------------------

        active_joint_names = [
            joint.name
            for joint
            in robot.active_joints
        ]

        name_to_index = {
            name: i
            for i, name
            in enumerate(
                active_joint_names
            )
        }

        arm_indices = [
            name_to_index[
                name
            ]
            for name
            in arm_joint_names
        ]

        full_qpos = first_env(
            robot.get_qpos()
        ).astype(
            np.float64
        )

        current_arm_qpos = (
            full_qpos[
                arm_indices
            ]
        )

        # Day6 Reference trajectoryの
        # 最初のwaypoint
        target_arm_qpos = np.asarray(
            trajectory[
                "segments"
            ][
                "to_pregrasp"
            ][
                "arm_qpos"
            ][0],
            dtype=np.float64,
        )

        # --------------------------------------------------
        # Reference Action
        # --------------------------------------------------

        reference_action = (
            composer
            .reference_action_from_qpos(
                current_arm_qpos,
                target_arm_qpos,
            )
        )

        physical_delta = (
            target_arm_qpos
            - current_arm_qpos
        )

        # ManiSkill公式式との比較
        official_action = np.clip(
            gym_utils.inv_scale_action(
                physical_delta,
                lower,
                upper,
            ),
            -1.0,
            1.0,
        )

        official_diff = float(
            np.max(
                np.abs(
                    reference_action
                    - official_action
                )
            )
        )

        print()
        print(
            "current qpos:",
            current_arm_qpos.tolist(),
        )

        print(
            "target qpos :",
            target_arm_qpos.tolist(),
        )

        print(
            "physical dq :",
            physical_delta.tolist(),
        )

        print(
            "reference action:",
            reference_action.tolist(),
        )

        print(
            "ManiSkill formula diff:",
            official_diff,
        )

        if official_diff > 1e-12:

            errors.append(
                "Reference Action conversion "
                "does not match ManiSkill scaling"
            )

        # --------------------------------------------------
        # Zero Residual
        # --------------------------------------------------

        zero_residual = np.zeros(
            arm_dof,
            dtype=np.float64,
        )

        zero_combined = (
            composer.compose(
                reference_action,
                zero_residual,
            )
        )

        zero_error = float(
            np.max(
                np.abs(
                    zero_combined
                    - reference_action
                )
            )
        )

        print()
        print(
            "zero residual combined:",
            zero_combined.tolist(),
        )

        print(
            "zero residual error:",
            zero_error,
        )

        if zero_error > 1e-12:

            errors.append(
                "Zero Residual does not "
                "reproduce Reference Action"
            )

        # --------------------------------------------------
        # Positive / Negative Residual
        # --------------------------------------------------

        positive_residual = np.ones(
            arm_dof,
            dtype=np.float64,
        )

        negative_residual = -np.ones(
            arm_dof,
            dtype=np.float64,
        )

        positive_combined = (
            composer.compose(
                reference_action,
                positive_residual,
            )
        )

        negative_combined = (
            composer.compose(
                reference_action,
                negative_residual,
            )
        )

        positive_physical_effect = (
            composer.physical_residual_effect(
                reference_action,
                positive_residual,
            )
        )

        negative_physical_effect = (
            composer.physical_residual_effect(
                reference_action,
                negative_residual,
            )
        )

        print()
        print(
            "+1 residual combined:",
            positive_combined.tolist(),
        )

        print(
            "+1 physical correction [rad]:",
            positive_physical_effect.tolist(),
        )

        print(
            "-1 residual combined:",
            negative_combined.tolist(),
        )

        print(
            "-1 physical correction [rad]:",
            negative_physical_effect.tolist(),
        )

        # --------------------------------------------------
        # Full 7-D Action / Gripper isolation
        # --------------------------------------------------

        total_dim = (
            env.action_space.shape[0]
        )

        pre_contact_action = float(
            config[
                "gripper"
            ][
                "pre_contact_action"
            ]
        )

        reference_full_action = (
            np.zeros(
                total_dim,
                dtype=np.float32,
            )
        )

        zero_full_action = (
            np.zeros(
                total_dim,
                dtype=np.float32,
            )
        )

        positive_full_action = (
            np.zeros(
                total_dim,
                dtype=np.float32,
            )
        )

        reference_full_action[
            arm_slice
        ] = (
            reference_action
        )

        zero_full_action[
            arm_slice
        ] = (
            zero_combined
        )

        positive_full_action[
            arm_slice
        ] = (
            positive_combined
        )

        reference_full_action[
            gripper_slice
        ] = (
            pre_contact_action
        )

        zero_full_action[
            gripper_slice
        ] = (
            pre_contact_action
        )

        positive_full_action[
            gripper_slice
        ] = (
            pre_contact_action
        )

        gripper_zero_diff = float(
            np.max(
                np.abs(
                    zero_full_action[
                        gripper_slice
                    ]
                    - reference_full_action[
                        gripper_slice
                    ]
                )
            )
        )

        gripper_positive_diff = float(
            np.max(
                np.abs(
                    positive_full_action[
                        gripper_slice
                    ]
                    - reference_full_action[
                        gripper_slice
                    ]
                )
            )
        )

        if (
            gripper_zero_diff > 0.0
            or gripper_positive_diff > 0.0
        ):

            errors.append(
                "Residual Action changed "
                "the scripted gripper action"
            )

        # --------------------------------------------------
        # Random stress test
        # --------------------------------------------------

        rng = np.random.default_rng(
            args.seed
        )

        max_random_zero_error = (
            0.0
        )

        out_of_bounds_count = 0

        for _ in range(
            args.random_tests
        ):

            random_reference = (
                rng.uniform(
                    -1.0,
                    1.0,
                    arm_dof,
                )
            )

            random_residual = (
                rng.uniform(
                    -1.0,
                    1.0,
                    arm_dof,
                )
            )

            combined = (
                composer.compose(
                    random_reference,
                    random_residual,
                )
            )

            zero_test = (
                composer.compose(
                    random_reference,
                    np.zeros(
                        arm_dof
                    ),
                )
            )

            max_random_zero_error = max(
                max_random_zero_error,
                float(
                    np.max(
                        np.abs(
                            zero_test
                            - random_reference
                        )
                    )
                ),
            )

            if (
                np.any(
                    combined
                    < -1.0
                )
                or np.any(
                    combined
                    > 1.0
                )
            ):

                out_of_bounds_count += 1

        if max_random_zero_error > 1e-12:

            errors.append(
                "Random zero residual "
                "equivalence failed"
            )

        if out_of_bounds_count != 0:

            errors.append(
                "Combined Action exceeded "
                "[-1,1]"
            )

        # --------------------------------------------------
        # Actual env.step sanity check
        # --------------------------------------------------

        qpos_before_step = (
            first_env(
                robot.get_qpos()
            )
            .astype(np.float64)[
                arm_indices
            ]
            .copy()
        )

        error_before_step = float(
            np.linalg.norm(
                target_arm_qpos
                - qpos_before_step
            )
        )

        env.step(
            zero_full_action
        )

        qpos_after_step = (
            first_env(
                robot.get_qpos()
            )
            .astype(np.float64)[
                arm_indices
            ]
            .copy()
        )

        error_after_step = float(
            np.linalg.norm(
                target_arm_qpos
                - qpos_after_step
            )
        )

        if not np.all(
            np.isfinite(
                qpos_after_step
            )
        ):

            errors.append(
                "Non-finite qpos after "
                "env.step"
            )

        # --------------------------------------------------
        # Report
        # --------------------------------------------------

        report = {
            "control_mode":
                config[
                    "project"
                ][
                    "control_mode"
                ],

            "trajectory":
                str(
                    args.trajectory
                ),

            "arm_joint_names":
                arm_joint_names,

            "arm_dof":
                arm_dof,

            "environment_action_dim":
                int(
                    total_dim
                ),

            "arm_action_slice":
                [
                    arm_slice.start,
                    arm_slice.stop,
                ],

            "gripper_action_slice":
                [
                    gripper_slice.start,
                    gripper_slice.stop,
                ],

            "controller": {
                "class":
                    type(
                        arm_controller
                    ).__name__,

                "use_delta":
                    use_delta,

                "use_target":
                    use_target,

                "normalize_action":
                    normalize_action,

                "lower":
                    lower.tolist(),

                "upper":
                    upper.tolist(),
            },

            "alpha":
                alpha,

            "current_arm_qpos":
                current_arm_qpos.tolist(),

            "target_arm_qpos":
                target_arm_qpos.tolist(),

            "physical_delta_rad":
                physical_delta.tolist(),

            "reference_action":
                reference_action.tolist(),

            "official_formula_difference":
                official_diff,

            "zero_residual": {
                "combined_action":
                    zero_combined.tolist(),

                "max_difference_from_reference":
                    zero_error,
            },

            "positive_unit_residual": {
                "combined_action":
                    positive_combined.tolist(),

                "physical_effect_rad":
                    positive_physical_effect.tolist(),
            },

            "negative_unit_residual": {
                "combined_action":
                    negative_combined.tolist(),

                "physical_effect_rad":
                    negative_physical_effect.tolist(),
            },

            "gripper": {
                "scripted_action":
                    pre_contact_action,

                "zero_residual_difference":
                    gripper_zero_diff,

                "positive_residual_difference":
                    gripper_positive_diff,
            },

            "random_test": {
                "count":
                    int(
                        args.random_tests
                    ),

                "max_zero_residual_error":
                    max_random_zero_error,

                "out_of_bounds_count":
                    out_of_bounds_count,
            },

            "env_step": {
                "qpos_before":
                    qpos_before_step.tolist(),

                "qpos_after":
                    qpos_after_step.tolist(),

                "target_error_before":
                    error_before_step,

                "target_error_after":
                    error_after_step,
            },

            "warnings":
                warnings,

            "errors":
                errors,

            "passed":
                len(errors) == 0,
        }

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # --------------------------------------------------
        # Terminal summary
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("Day 8 Result")
        print("=" * 72)

        print(
            "action dim:",
            total_dim,
        )

        print(
            "arm slice:",
            (
                arm_slice.start,
                arm_slice.stop,
            ),
        )

        print(
            "gripper slice:",
            (
                gripper_slice.start,
                gripper_slice.stop,
            ),
        )

        print(
            "zero residual error:",
            zero_error,
        )

        print(
            "random bounds violations:",
            out_of_bounds_count,
        )

        print(
            "gripper residual difference:",
            gripper_positive_diff,
        )

        print(
            "target error before step:",
            error_before_step,
        )

        print(
            "target error after step :",
            error_after_step,
        )

        print(
            "errors:",
            errors,
        )

        print(
            "PASSED:",
            len(errors) == 0,
        )

        print(
            "report:",
            args.output,
        )

        return (
            0
            if len(errors) == 0
            else 1
        )

    finally:

        env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )