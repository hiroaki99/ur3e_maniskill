#!/usr/bin/env python3
"""Day5: repeated fixed-condition Reference-only evaluation for Week1 Gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pickplace_week1_common import execute_reference_episode
from rrl.pick_place_safety import RobotTableContactAudit

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"
DEFAULT_TRAJECTORY = REPO_ROOT / "trajectories" / "pick_place_reference_v1.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pickplace_week1/day05_reference_eval"),
    )
    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with args.trajectory.open("r", encoding="utf-8") as f:
        trajectory = json.load(f)

    episodes = int(
        args.episodes
        if args.episodes is not None
        else config["pick_place"]["reference_eval_episodes"]
    )

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

    rows = []
    try:
        # First reset only to expose scene/link objects for the safety audit.
        env.reset(seed=args.seed)
        base = env.unwrapped
        safety = RobotTableContactAudit(
            scene=base.scene,
            table=base.table_scene.table,
            robot_links=base.agent.robot.links,
            force_threshold_n=float(config["safety"]["table_contact_force_threshold_n"]),
            ignored_link_names=config["safety"].get("ignored_table_contact_links", []),
            physically_validated=bool(config["safety"]["collision_instrumentation_validated"]),
        )

        for episode in range(episodes):
            seed = int(args.seed + episode)
            result = execute_reference_episode(
                env=env,
                trajectory=trajectory,
                config=config,
                seed=seed,
                render=False,
                sleep=0.0,
                verbose=False,
                safety_audit=safety,
            )
            row = {"episode": episode, "seed": seed, **result}
            # Dict-valued field is stored as JSON text for CSV.
            row["robot_table_max_force_n"] = json.dumps(
                row["robot_table_max_force_n"], ensure_ascii=False
            )
            rows.append(row)
            print(
                f"episode={episode:02d} seed={seed} "
                f"success={result['success']} cause={result['failure_cause']} "
                f"goal_xy={result['goal_xy_error_m']:.5f}"
            )

        success_count = sum(int(r["success"]) for r in rows)
        success_rate = success_count / episodes
        failures = Counter(
            r["failure_cause"] for r in rows if not r["success"]
        )

        # Week1 Gate: 31/32 or >= 95% for other requested episode counts.
        gate_passed = success_rate >= 0.95
        collision_validated = bool(config["safety"]["collision_instrumentation_validated"])

        summary = {
            "episodes": episodes,
            "success_count": success_count,
            "success_rate": success_rate,
            "failure_counts": dict(failures),
            "week1_gate_min_success_rate": 0.95,
            "week1_gate_passed": gate_passed,
            "collision_instrumentation_validated": collision_validated,
            "collision_rate": None,
            "collision_note": (
                "N/A until raw robot-table contact candidates are physically "
                "validated with a positive-control collision test."
            ),
            "interpretation_note": (
                "Fixed Cube/fixed Goal repeated runs are an implementation stability "
                "check, not independent generalization trials."
            ),
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "episodes.csv", rows)
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        summary_rows = [
            {
                "episodes": episodes,
                "success_count": success_count,
                "success_rate": success_rate,
                "week1_gate_passed": gate_passed,
                "collision_rate": "N/A",
            }
        ]
        write_csv(args.output_dir / "summary.csv", summary_rows)

        print()
        print("=" * 72)
        print("Week1 Fixed Pick-and-Place Gate")
        print("=" * 72)
        print(f"success: {success_count}/{episodes} = {success_rate:.3f}")
        print("failures:", dict(failures))
        print("collision rate: N/A")
        print("Gate >= 0.95:", gate_passed)
        print("summary:", args.output_dir / "summary.json")
        return 0 if gate_passed else 2

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
