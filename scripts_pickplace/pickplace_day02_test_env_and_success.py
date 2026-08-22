#!/usr/bin/env python3
"""Day2: smoke-test UR3ePickPlace-v0 and its success definition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mani_skill.utils.structs.pose import Pose
from day6_common import first_env, scalar_first
from pickplace_week1_common import build_context, step_robot

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pickplace_week1/day02_success_test.json"),
    )
    return p.parse_args()


def set_cube(base_env, p, velocity=None):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    base_env.cube.set_pose(
        Pose.create_from_pq(
            p=torch.tensor(p, device=base_env.device).reshape(1, 3),
            q=torch.tensor(q, device=base_env.device).reshape(1, 4),
        )
    )
    v = np.zeros(3, dtype=np.float32) if velocity is None else np.asarray(velocity, dtype=np.float32)
    base_env.cube.set_linear_velocity(torch.tensor(v, device=base_env.device).reshape(1, 3))
    base_env.cube.set_angular_velocity(torch.zeros((1, 3), device=base_env.device))


def set_gripper(env, ctx, command, steps=100):
    arm = first_env(ctx["robot"].get_qpos()).astype(float)[ctx["arm_indices"]].copy()
    info = None
    for _ in range(steps):
        _, _, _, _, info = step_robot(env, ctx, arm, command)
    return info


def run_case(env, config, name, position, *, gripper_open, settle_steps, velocity=None):
    env.reset(seed=int(config["project"]["seed"]))
    ctx = build_context(env, config)
    base = ctx["base_env"]

    command = (
        float(config["gripper"]["open_action"])
        if gripper_open
        else float(config["gripper"]["close_action"])
    )
    set_gripper(env, ctx, command, steps=120)
    set_cube(base, position, velocity=velocity)

    arm = first_env(ctx["robot"].get_qpos()).astype(float)[ctx["arm_indices"]].copy()
    info = base.evaluate()
    for _ in range(settle_steps):
        _, _, _, _, info = step_robot(env, ctx, arm, command)

    return {
        "name": name,
        "success": bool(scalar_first(info["success"])),
        "within_goal_xy": bool(scalar_first(info["within_goal_xy"])),
        "within_goal_z": bool(scalar_first(info["within_goal_z"])),
        "released": bool(scalar_first(info["released"])),
        "is_stable": bool(scalar_first(info["is_stable"])),
        "stable_place_steps": int(scalar_first(info["stable_place_steps"])),
        "goal_xy_error_m": float(scalar_first(info["goal_xy_error_m"])),
        "goal_z_error_m": float(scalar_first(info["goal_z_error_m"])),
    }


def main():
    args = parse_args()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    import envs.ur3e_pick_place  # noqa: F401

    env = gym.make(
        "UR3ePickPlace-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=args.sim_backend,
        max_episode_steps=int(config["pick_place"]["max_episode_steps"]),
    )

    goal = np.asarray(config["pick_place"]["goal_position"], dtype=float)
    n = int(config["pick_place"]["stable_steps"])

    cases = []
    try:
        cases.append(run_case(env, config, "center_open_stable", goal, gripper_open=True, settle_steps=n + 2))
        cases.append(run_case(env, config, "offset_10mm", goal + [0.010, 0, 0], gripper_open=True, settle_steps=n + 2))
        cases.append(run_case(env, config, "offset_20mm", goal + [0.020, 0, 0], gripper_open=True, settle_steps=n + 2))
        cases.append(run_case(env, config, "center_closed", goal, gripper_open=False, settle_steps=n + 2))
        cases.append(run_case(env, config, "center_before_stable_count", goal, gripper_open=True, settle_steps=max(0, n - 2)))
        cases.append(run_case(env, config, "center_moving", goal, gripper_open=True, settle_steps=0, velocity=[0.10, 0.0, 0.0]))

        expected = {
            "center_open_stable": True,
            "offset_10mm": True,
            "offset_20mm": False,
            "center_closed": False,
            "center_before_stable_count": False,
            "center_moving": False,
        }

        passed = True
        for row in cases:
            row["expected_success"] = expected[row["name"]]
            row["passed"] = row["success"] == row["expected_success"]
            passed = passed and row["passed"]
            print(
                f"{row['name']:28s} success={row['success']} "
                f"expected={row['expected_success']} passed={row['passed']}"
            )

        report = {
            "passed": bool(passed),
            "goal_position_m": goal.tolist(),
            "stable_steps_required": n,
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("PASSED:", passed)
        print("report:", args.output)
        return 0 if passed else 2

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
