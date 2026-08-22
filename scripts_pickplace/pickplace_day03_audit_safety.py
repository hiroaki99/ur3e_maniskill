#!/usr/bin/env python3
"""Day3: audit raw robot-table contact signals without claiming collision validity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from day6_common import first_env, scalar_first
from pickplace_week1_common import build_context, step_robot
from rrl.pick_place_safety import RobotTableContactAudit

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pickplace_week1/day03_safety_audit.json"),
    )
    return p.parse_args()


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
    )

    try:
        env.reset(seed=int(config["project"]["seed"]))
        ctx = build_context(env, config)
        base = ctx["base_env"]

        audit = RobotTableContactAudit(
            scene=base.scene,
            table=base.table_scene.table,
            robot_links=base.agent.robot.links,
            force_threshold_n=float(config["safety"]["table_contact_force_threshold_n"]),
            ignored_link_names=config["safety"].get("ignored_table_contact_links", []),
            physically_validated=bool(config["safety"]["collision_instrumentation_validated"]),
        )

        arm = first_env(ctx["robot"].get_qpos()).astype(float)[ctx["arm_indices"]].copy()
        max_by_link = {}
        candidate_steps = 0

        for _ in range(args.steps):
            step_robot(
                env,
                ctx,
                arm,
                float(config["gripper"]["open_action"]),
            )
            obs = audit.observe()
            if bool(scalar_first(obs["raw_robot_table_contact_candidate"])):
                candidate_steps += 1
            for name, tensor in obs["per_link_force_n"].items():
                force = float(scalar_first(tensor))
                max_by_link[name] = max(max_by_link.get(name, 0.0), force)

        ranked = sorted(max_by_link.items(), key=lambda kv: kv[1], reverse=True)
        reportable = bool(config["safety"]["collision_instrumentation_validated"])

        print("=" * 72)
        print("Day3 Robot-Table Contact Audit")
        print("=" * 72)
        print("candidate contact steps:", candidate_steps, "/", args.steps)
        print("collision rate reportable:", reportable)
        print("top robot-table forces:")
        for name, force in ranked[:15]:
            print(f"  {name:45s} {force:9.4f} N")

        report = {
            "steps": args.steps,
            "raw_candidate_steps": candidate_steps,
            "force_threshold_n": float(config["safety"]["table_contact_force_threshold_n"]),
            "collision_instrumentation_validated": reportable,
            "collision_rate_reportable": reportable,
            "max_force_by_link_n": dict(ranked),
            "decision": (
                "Keep collision_rate=N/A until a separate positive-control robot-table "
                "collision test establishes the intended allowed/unsafe link set."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("report:", args.output)
        return 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
