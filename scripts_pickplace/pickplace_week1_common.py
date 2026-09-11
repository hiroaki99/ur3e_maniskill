#!/usr/bin/env python3
"""Shared Week1 utilities for scripted UR3e Pick-and-Place."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from day6_common import (
    build_controller_slices,
    find_controller,
    first_env,
    get_controller_joint_names,
    scalar_first,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"


def load_config(path: Path = CONFIG_PATH):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_trajectory(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_context(env, config):
    base_env = env.unwrapped
    robot = base_env.agent.robot
    controllers = base_env.agent.controller.controllers
    slices = build_controller_slices(controllers)
    arm_name = find_controller(controllers, "arm")
    gripper_name = find_controller(controllers, "gripper")

    active_names = [joint.name for joint in robot.active_joints]
    name_to_index = {name: i for i, name in enumerate(active_names)}
    arm_names = get_controller_joint_names(controllers[arm_name])
    arm_indices = [name_to_index[name] for name in arm_names]

    return {
        "base_env": base_env,
        "robot": robot,
        "controllers": controllers,
        "arm_slice": slices[arm_name],
        "gripper_slice": slices[gripper_name],
        "arm_names": arm_names,
        "arm_indices": arm_indices,
        "total_action_dim": int(env.action_space.shape[0]),
        "max_delta": float(config["control"]["max_joint_delta_rad"]),
    }


def step_robot(
    env,
    ctx,
    target_arm,
    gripper_command,
    *,
    render=False,
    sleep=0.0,
):
    current = first_env(ctx["robot"].get_qpos()).astype(float)
    error = np.asarray(target_arm, dtype=float) - current[ctx["arm_indices"]]

    action = np.zeros(ctx["total_action_dim"], dtype=np.float32)
    action[ctx["arm_slice"]] = np.clip(
        error / ctx["max_delta"],
        -1.0,
        1.0,
    )
    action[ctx["gripper_slice"]] = float(gripper_command)

    result = env.step(action)
    if render:
        env.render()
        if sleep > 0:
            time.sleep(sleep)
    return result


def run_path(
    env,
    ctx,
    q_path,
    gripper_command,
    *,
    control_steps,
    render=False,
    sleep=0.0,
    verbose=False,
    phase_name="path",
    safety_audit=None,
    metrics_state=None,
):
    last_info = None
    for i, q_target in enumerate(q_path):
        for _ in range(control_steps):
            obs, reward, terminated, truncated, info = step_robot(
                env,
                ctx,
                q_target,
                gripper_command,
                render=render,
                sleep=sleep,
            )
            last_info = info
            if metrics_state is not None:
                update_metrics(ctx["base_env"], info, metrics_state, phase_name)
            if safety_audit is not None:
                update_safety(safety_audit, metrics_state)

        if verbose:
            print(
                f"{phase_name}: waypoint {i+1:02d}/{len(q_path):02d} "
                f"L={bool(scalar_first(last_info['left_contact']))} "
                f"R={bool(scalar_first(last_info['right_contact']))}"
            )
    return last_info


def update_safety(safety_audit, metrics_state):
    if safety_audit is None or metrics_state is None:
        return
    obs = safety_audit.observe()
    for name, tensor in obs["per_link_force_n"].items():
        value = float(scalar_first(tensor))
        metrics_state["robot_table_max_force_n"][name] = max(
            metrics_state["robot_table_max_force_n"].get(name, 0.0),
            value,
        )


def update_metrics(base_env, info, state, phase_name):
    cube_p = first_env(base_env.cube.pose.p).astype(float)
    cube_lift = float(cube_p[2] - state["cube_initial_position_m"][2])
    state["max_cube_lift_m"] = max(state["max_cube_lift_m"], cube_lift)

    left = bool(scalar_first(info.get("left_contact", False)))
    right = bool(scalar_first(info.get("right_contact", False)))
    if left and right:
        state["ever_bilateral_contact"] = True

    if phase_name == "transport":
        state["min_cube_lift_during_transport_m"] = min(
            state["min_cube_lift_during_transport_m"],
            cube_lift,
        )

    state["max_left_contact_force_n"] = max(
        state["max_left_contact_force_n"],
        float(scalar_first(info.get("left_contact_force", 0.0))),
    )
    state["max_right_contact_force_n"] = max(
        state["max_right_contact_force_n"],
        float(scalar_first(info.get("right_contact_force", 0.0))),
    )


def init_metrics(base_env):
    cube_initial = first_env(base_env.cube.pose.p).astype(float).copy()
    return {
        "cube_initial_position_m": cube_initial,
        "max_cube_lift_m": 0.0,
        "min_cube_lift_during_transport_m": float("inf"),
        "ever_bilateral_contact": False,
        "max_left_contact_force_n": 0.0,
        "max_right_contact_force_n": 0.0,
        "robot_table_max_force_n": {},
    }


def execute_reference_episode(
    *,
    env,
    trajectory,
    config,
    seed,
    render=False,
    sleep=0.0,
    verbose=False,
    safety_audit=None,
):
    """Execute one full scripted Pick-and-Place episode."""

    pp = config["pick_place"]
    gripper = config["gripper"]

    obs, reset_info = env.reset(seed=int(seed))
    ctx = build_context(env, config)
    base_env = ctx["base_env"]
    robot = ctx["robot"]

    if trajectory["arm_joint_names"] != ctx["arm_names"]:
        raise RuntimeError("trajectory arm joint order differs from current robot")

    metrics = init_metrics(base_env)

    pre_contact = float(gripper["pre_contact_action"])
    close_action = float(gripper["close_action"])
    open_action = float(gripper["open_action"])
    contact_threshold = float(gripper["contact_force_threshold_n"])
    stable_grasp_force = float(gripper["stable_grasp_force_n"])
    control_steps = int(pp["control_steps_per_waypoint"])

    # 1) Pre-contact initialization while keeping the arm fixed.
    current_arm = first_env(robot.get_qpos()).astype(float)[ctx["arm_indices"]].copy()
    for _ in range(int(pp["reset_precontact_steps"])):
        _, _, _, _, info = step_robot(
            env, ctx, current_arm, pre_contact, render=render, sleep=sleep
        )
        update_metrics(base_env, info, metrics, "reset")
        update_safety(safety_audit, metrics)

    # 2) Approach and descend: reuse the already validated Pick-and-Lift segments.
    run_path(
        env,
        ctx,
        trajectory["segments"]["to_pregrasp"]["arm_qpos"],
        pre_contact,
        control_steps=control_steps,
        render=render,
        sleep=sleep,
        verbose=verbose,
        phase_name="to_pregrasp",
        safety_audit=safety_audit,
        metrics_state=metrics,
    )

    q_pregrasp = np.asarray(
        trajectory["segments"]["to_pregrasp"]["final_arm_qpos"], dtype=float
    )
    for _ in range(int(pp["pregrasp_hold_steps"])):
        _, _, _, _, info = step_robot(
            env, ctx, q_pregrasp, pre_contact, render=render, sleep=sleep
        )
        update_metrics(base_env, info, metrics, "pregrasp_hold")
        update_safety(safety_audit, metrics)

    run_path(
        env,
        ctx,
        trajectory["segments"]["descend"]["arm_qpos"],
        pre_contact,
        control_steps=control_steps,
        render=render,
        sleep=sleep,
        verbose=verbose,
        phase_name="descend",
        safety_audit=safety_audit,
        metrics_state=metrics,
    )
    q_grasp = np.asarray(
        trajectory["segments"]["descend"]["final_arm_qpos"], dtype=float
    )

    # 3) Grasp ramp.
    bilateral_streak = 0
    grasp_command = None
    for command in np.linspace(
        pre_contact,
        close_action,
        int(pp["grasp_ramp_steps"]),
    ):
        _, _, _, _, info = step_robot(
            env, ctx, q_grasp, float(command), render=render, sleep=sleep
        )
        update_metrics(base_env, info, metrics, "grasp")
        update_safety(safety_audit, metrics)

        left_force = float(scalar_first(info["left_contact_force"]))
        right_force = float(scalar_first(info["right_contact_force"]))
        if left_force >= contact_threshold and right_force >= contact_threshold:
            bilateral_streak += 1
        else:
            bilateral_streak = 0

        if bilateral_streak >= int(pp["bilateral_contact_streak"]):
            grasp_command = float(command)
            break

    if grasp_command is None:
        return finalize_result(
            base_env,
            ctx,
            metrics,
            final_info=info,
            failure_cause="grasp_failure",
            grasp_command=None,
        )

    direction = np.sign(close_action - pre_contact)
    grasp_command = float(
        np.clip(
            grasp_command + direction * float(pp["squeeze_margin_action"]),
            -1.0,
            1.0,
        )
    )

    # 4) Stable grasp check.
    stable_both = 0
    for _ in range(int(pp["pre_lift_hold_steps"])):
        _, _, _, _, info = step_robot(
            env, ctx, q_grasp, grasp_command, render=render, sleep=sleep
        )
        update_metrics(base_env, info, metrics, "stable_grasp")
        update_safety(safety_audit, metrics)

        left_force = float(scalar_first(info["left_contact_force"]))
        right_force = float(scalar_first(info["right_contact_force"]))
        if left_force >= stable_grasp_force and right_force >= stable_grasp_force:
            stable_both += 1

    required = int(
        np.ceil(
            int(pp["pre_lift_hold_steps"])
            * float(pp["pre_lift_required_contact_ratio"])
        )
    )
    if stable_both < required:
        return finalize_result(
            base_env,
            ctx,
            metrics,
            final_info=info,
            failure_cause="unstable_grasp",
            grasp_command=grasp_command,
        )

    # 5) Lift.
    run_path(
        env,
        ctx,
        trajectory["segments"]["lift"]["arm_qpos"],
        grasp_command,
        control_steps=control_steps,
        render=render,
        sleep=sleep,
        verbose=verbose,
        phase_name="lift",
        safety_audit=safety_audit,
        metrics_state=metrics,
    )

    if metrics["max_cube_lift_m"] < float(pp["success_lift_height_m"]):
        return finalize_result(
            base_env,
            ctx,
            metrics,
            final_info=info,
            failure_cause="lift_failure",
            grasp_command=grasp_command,
        )

    # 6) Transport.
    run_path(
        env,
        ctx,
        trajectory["segments"]["transport"]["arm_qpos"],
        grasp_command,
        control_steps=control_steps,
        render=render,
        sleep=sleep,
        verbose=verbose,
        phase_name="transport",
        safety_audit=safety_audit,
        metrics_state=metrics,
    )

    # If the cube fell almost back to the table before intended descent.
    if metrics["min_cube_lift_during_transport_m"] < 0.02:
        return finalize_result(
            base_env,
            ctx,
            metrics,
            final_info=info,
            failure_cause="drop_during_transport",
            grasp_command=grasp_command,
        )

    # 7) Descend to placement pose.
    run_path(
        env,
        ctx,
        trajectory["segments"]["descend_place"]["arm_qpos"],
        grasp_command,
        control_steps=control_steps,
        render=render,
        sleep=sleep,
        verbose=verbose,
        phase_name="descend_place",
        safety_audit=safety_audit,
        metrics_state=metrics,
    )
    q_place = np.asarray(
        trajectory["segments"]["descend_place"]["final_arm_qpos"], dtype=float
    )

    # 8) Open gripper gradually.
    for command in np.linspace(
        grasp_command,
        open_action,
        int(pp["release_ramp_steps"]),
    ):
        _, _, _, _, info = step_robot(
            env, ctx, q_place, float(command), render=render, sleep=sleep
        )
        update_metrics(base_env, info, metrics, "release")
        update_safety(safety_audit, metrics)

    for _ in range(int(pp["release_hold_steps"])):
        _, _, _, _, info = step_robot(
            env, ctx, q_place, open_action, render=render, sleep=sleep
        )
        update_metrics(base_env, info, metrics, "release_hold")
        update_safety(safety_audit, metrics)

    # 9) Retreat while the cube should remain on the table.
    run_path(
        env,
        ctx,
        trajectory["segments"]["retreat"]["arm_qpos"],
        open_action,
        control_steps=control_steps,
        render=render,
        sleep=sleep,
        verbose=verbose,
        phase_name="retreat",
        safety_audit=safety_audit,
        metrics_state=metrics,
    )
    q_retreat = np.asarray(
        trajectory["segments"]["retreat"]["final_arm_qpos"], dtype=float
    )

    for _ in range(int(pp["final_settle_steps"])):
        _, _, _, _, info = step_robot(
            env, ctx, q_retreat, open_action, render=render, sleep=sleep
        )
        update_metrics(base_env, info, metrics, "final_settle")
        update_safety(safety_audit, metrics)

    success = bool(scalar_first(info.get("success", False)))
    if success:
        cause = "success"
    elif not bool(scalar_first(info.get("released", False))):
        cause = "release_failure"
    elif not bool(scalar_first(info.get("within_goal_xy", False))):
        cause = "place_alignment_failure"
    else:
        cause = "unstable_placement"

    return finalize_result(
        base_env,
        ctx,
        metrics,
        final_info=info,
        failure_cause=cause,
        grasp_command=grasp_command,
    )


def finalize_result(base_env, ctx, metrics, *, final_info, failure_cause, grasp_command):
    cube_final = first_env(base_env.cube.pose.p).astype(float).copy()

    success = bool(scalar_first(final_info.get("success", False)))
    result = {
        "success": success,
        "failure_cause": "success" if success else failure_cause,
        "grasp_command": None if grasp_command is None else float(grasp_command),
        "cube_initial_position_m": metrics["cube_initial_position_m"].tolist(),
        "cube_final_position_m": cube_final.tolist(),
        "max_cube_lift_m": float(metrics["max_cube_lift_m"]),
        "min_cube_lift_during_transport_m": (
            None
            if np.isinf(metrics["min_cube_lift_during_transport_m"])
            else float(metrics["min_cube_lift_during_transport_m"])
        ),
        "ever_bilateral_contact": bool(metrics["ever_bilateral_contact"]),
        "goal_xy_error_m": float(scalar_first(final_info.get("goal_xy_error_m", np.nan))),
        "goal_z_error_m": float(scalar_first(final_info.get("goal_z_error_m", np.nan))),
        "released": bool(scalar_first(final_info.get("released", False))),
        "is_stable": bool(scalar_first(final_info.get("is_stable", False))),
        "stable_place_steps": int(scalar_first(final_info.get("stable_place_steps", 0))),
        "max_left_contact_force_n": float(metrics["max_left_contact_force_n"]),
        "max_right_contact_force_n": float(metrics["max_right_contact_force_n"]),
        "robot_table_max_force_n": {
            k: float(v) for k, v in metrics["robot_table_max_force_n"].items()
        },
    }
    return result
