#!/usr/bin/env python3
"""Day4 planner: extend the validated Pick-and-Lift reference with transport/place/retreat."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from day6_common import first_env, plan_cartesian_segment
from pickplace_week1_common import build_context, step_robot

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"
BASE_TRAJECTORY = REPO_ROOT / "trajectories" / "day6_pick_lift_reference_v2.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-trajectory", type=Path, default=BASE_TRAJECTORY)
    p.add_argument("--goal-x", type=float, default=None)
    p.add_argument("--goal-y", type=float, default=None)
    p.add_argument("--goal-z", type=float, default=None)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path("trajectories/pick_place_reference_v1.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with args.base_trajectory.open("r", encoding="utf-8") as f:
        base_traj = json.load(f)

    goal = np.asarray(config["pick_place"]["goal_position"], dtype=float)
    if args.goal_x is not None:
        goal[0] = args.goal_x
    if args.goal_y is not None:
        goal[1] = args.goal_y
    if args.goal_z is not None:
        goal[2] = args.goal_z

    import envs.ur3e_pick_place  # noqa: F401

    env = gym.make(
        "UR3ePickPlace-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=args.sim_backend,
    )

    try:
        env.reset(seed=int(config["project"]["seed"]))
        ctx = build_context(env, config)
        base = ctx["base_env"]
        robot = ctx["robot"]

        # Match prior Day6 planning setup: stabilize gripper at pre-contact.
        current_arm = first_env(robot.get_qpos()).astype(float)[ctx["arm_indices"]].copy()
        for _ in range(int(config["pick_place"]["reset_precontact_steps"])):
            step_robot(
                env,
                ctx,
                current_arm,
                float(config["gripper"]["pre_contact_action"]),
            )

        full_qpos = first_env(robot.get_qpos()).astype(float)
        qlimits = first_env(robot.get_qlimits()).astype(float)[ctx["arm_indices"]]

        q_lift = np.asarray(
            base_traj["segments"]["lift"]["final_arm_qpos"], dtype=float
        )

        grasp_offset = np.asarray(
            config["pick_place"]["grasp_center_offset_m"], dtype=float
        )
        place_target = goal + grasp_offset
        transport_target = place_target.copy()
        transport_target[2] += float(config["pick_place"]["transport_height_m"])
        retreat_target = place_target.copy()
        retreat_target[2] += float(config["pick_place"]["retreat_height_m"])

        common = {
            "robot": robot,
            "device": base.device,
            "left_links": base.left_contact_links,
            "right_links": base.right_contact_links,
            "template_qpos": full_qpos,
            "arm_indices": ctx["arm_indices"],
            "qlimits_arm": qlimits,
            "epsilon": float(config["ik"]["epsilon_rad"]),
            "damping": float(config["ik"]["damping"]),
            "tolerance": float(config["ik"]["tolerance_m"]),
            "max_iterations": int(config["ik"]["max_iterations"]),
            "max_step": float(config["ik"]["max_step_rad"]),
        }

        print("=" * 72)
        print("Pick-and-Place Day4 Reference Planner")
        print("=" * 72)
        print("goal center      :", goal.tolist())
        print("transport target :", transport_target.tolist())
        print("place target     :", place_target.tolist())
        print("retreat target   :", retreat_target.tolist())

        transport = plan_cartesian_segment(
            start_arm_qpos=q_lift,
            target_position=transport_target,
            waypoint_count=int(config["pick_place"]["transport_waypoints"]),
            **common,
        )
        q_transport = np.asarray(transport["final_arm_qpos"], dtype=float)

        descend_place = plan_cartesian_segment(
            start_arm_qpos=q_transport,
            target_position=place_target,
            waypoint_count=int(config["pick_place"]["descend_place_waypoints"]),
            **common,
        )
        q_place = np.asarray(descend_place["final_arm_qpos"], dtype=float)

        retreat = plan_cartesian_segment(
            start_arm_qpos=q_place,
            target_position=retreat_target,
            waypoint_count=int(config["pick_place"]["retreat_waypoints"]),
            **common,
        )

        report = deepcopy(base_traj)
        report["task"] = "fixed_pick_and_place_week1"
        report["base_pick_lift_trajectory"] = str(args.base_trajectory)
        report["goal_position_m"] = goal.tolist()
        report["transport_target_m"] = transport_target.tolist()
        report["place_target_m"] = place_target.tolist()
        report["retreat_target_m"] = retreat_target.tolist()
        report["segments"]["transport"] = transport
        report["segments"]["descend_place"] = descend_place
        report["segments"]["retreat"] = retreat
        report["reference_arm_qpos"] = (
            report["segments"]["to_pregrasp"]["arm_qpos"]
            + report["segments"]["descend"]["arm_qpos"]
            + report["segments"]["lift"]["arm_qpos"]
            + transport["arm_qpos"]
            + descend_place["arm_qpos"]
            + retreat["arm_qpos"]
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("transport final error:", transport["errors_m"][-1])
        print("place final error    :", descend_place["errors_m"][-1])
        print("retreat final error  :", retreat["errors_m"][-1])
        print("total waypoints      :", len(report["reference_arm_qpos"]))
        print("output:", args.output)
        return 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
