#!/usr/bin/env python3
"""Day1: scan candidate fixed Goal positions using the existing validated Pick-and-Lift IK tools."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from day6_common import (
    first_env,
    get_grasp_center,
    plan_cartesian_segment,
)
from pickplace_week1_common import build_context, step_robot

PP_CONFIG = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"
PICKLIFT_TRAJ = REPO_ROOT / "trajectories" / "day6_pick_lift_reference_v2.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--goal-y", type=float, nargs="*", default=None)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pickplace_week1/day01_reachability"),
    )
    return p.parse_args()


def qlimit_margin(q_path, limits):
    q = np.asarray(q_path, dtype=float)
    low = limits[:, 0]
    high = limits[:, 1]
    margin = np.minimum(q - low[None, :], high[None, :] - q)
    margin = margin[np.isfinite(margin)]
    return float(np.min(margin)) if margin.size else float("inf")


def path_length(q_path):
    q = np.asarray(q_path, dtype=float)
    if len(q) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def main():
    args = parse_args()
    with PP_CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with PICKLIFT_TRAJ.open("r", encoding="utf-8") as f:
        lift_traj = json.load(f)

    goal_y_values = (
        args.goal_y
        if args.goal_y
        else list(config["pick_place"]["goal_candidate_y_m"])
    )

    # Use the already validated Pick-and-Lift environment only for Day1 kinematics.
    import envs.ur3e_pick_lift  # noqa: F401

    env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=args.sim_backend,
        max_episode_steps=1000,
    )

    rows = []
    try:
        env.reset(seed=int(config["project"]["seed"]))
        ctx = build_context(env, config)
        base_env = ctx["base_env"]
        robot = ctx["robot"]

        # Match the Day6 planning condition: pre-contact gripper state.
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
        left_links = base_env.left_contact_links
        right_links = base_env.right_contact_links

        q_lift = np.asarray(
            lift_traj["segments"]["lift"]["final_arm_qpos"], dtype=float
        )

        grasp_offset = np.asarray(
            config["pick_place"]["grasp_center_offset_m"], dtype=float
        )
        transport_height = float(config["pick_place"]["transport_height_m"])
        retreat_height = float(config["pick_place"]["retreat_height_m"])
        goal_z = float(config["cube"]["position"][2])
        goal_x = float(config["cube"]["position"][0])

        common = {
            "robot": robot,
            "device": base_env.device,
            "left_links": left_links,
            "right_links": right_links,
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
        print("Pick-and-Place Day1 Goal Reachability Scan")
        print("=" * 72)
        print("validated lift grasp center:", get_grasp_center(left_links, right_links).tolist())
        print("candidate goal y:", goal_y_values)
        print()

        for goal_y in goal_y_values:
            goal = np.asarray([goal_x, float(goal_y), goal_z], dtype=float)
            place_target = goal + grasp_offset
            transport_target = place_target.copy()
            transport_target[2] += transport_height
            retreat_target = place_target.copy()
            retreat_target[2] += retreat_height

            try:
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

                all_q = (
                    transport["arm_qpos"]
                    + descend_place["arm_qpos"]
                    + retreat["arm_qpos"]
                )
                max_error = max(
                    float(transport["errors_m"][-1]),
                    float(descend_place["errors_m"][-1]),
                    float(retreat["errors_m"][-1]),
                )
                margin = qlimit_margin(all_q, qlimits)
                total_path = path_length(all_q)
                reachable = (
                    max_error <= 2.0 * float(config["ik"]["tolerance_m"])
                    and margin >= float(config["pick_place"]["min_joint_limit_margin_rad"])
                )

                row = {
                    "goal_x_m": goal[0],
                    "goal_y_m": goal[1],
                    "goal_z_m": goal[2],
                    "transport_final_error_m": float(transport["errors_m"][-1]),
                    "place_final_error_m": float(descend_place["errors_m"][-1]),
                    "retreat_final_error_m": float(retreat["errors_m"][-1]),
                    "max_final_error_m": max_error,
                    "min_joint_limit_margin_rad": margin,
                    "joint_path_length_rad": total_path,
                    "reachable": bool(reachable),
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "goal_x_m": goal[0],
                    "goal_y_m": goal[1],
                    "goal_z_m": goal[2],
                    "transport_final_error_m": "",
                    "place_final_error_m": "",
                    "retreat_final_error_m": "",
                    "max_final_error_m": "",
                    "min_joint_limit_margin_rad": "",
                    "joint_path_length_rad": "",
                    "reachable": False,
                    "error": repr(exc),
                }

            rows.append(row)
            print(
                f"goal_y={float(goal_y):.3f} m "
                f"reachable={row['reachable']} "
                f"max_err={row['max_final_error_m']} "
                f"margin={row['min_joint_limit_margin_rad']}"
            )

        valid = [r for r in rows if r["reachable"]]
        # Prefer joint-limit margin; use shorter path as tie-breaker.
        selected = None
        if valid:
            selected = max(
                valid,
                key=lambda r: (
                    float(r["min_joint_limit_margin_rad"]),
                    -float(r["joint_path_length_rad"]),
                ),
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "goal_reachability.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "candidate_count": len(rows),
            "reachable_count": len(valid),
            "selected_goal": selected,
            "note": (
                "Day1 is kinematic screening only. Day4/Day5 must verify the "
                "selected goal dynamically while carrying the cube."
            ),
        }
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print()
        print("selected goal:", selected)
        print("summary:", args.output_dir / "summary.json")
        return 0 if selected is not None else 2

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
