#!/usr/bin/env python3
"""Day4: execute one full fixed Pick-and-Place reference episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pickplace_week1_common import execute_reference_episode

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"
DEFAULT_TRAJECTORY = REPO_ROOT / "trajectories" / "pick_place_reference_v1.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--render", action="store_true")
    p.add_argument("--sleep", type=float, default=0.01)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pickplace_week1/day04_reference_run.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with args.trajectory.open("r", encoding="utf-8") as f:
        trajectory = json.load(f)

    seed = int(config["project"]["seed"] if args.seed is None else args.seed)

    import envs.ur3e_pick_place  # noqa: F401

    env = gym.make(
        "UR3ePickPlace-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=args.sim_backend,
        render_mode="human" if args.render else None,
        max_episode_steps=int(config["pick_place"]["max_episode_steps"]),
    )

    try:
        result = execute_reference_episode(
            env=env,
            trajectory=trajectory,
            config=config,
            seed=seed,
            render=args.render,
            sleep=args.sleep,
            verbose=True,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print()
        print("=" * 72)
        print("Day4 Pick-and-Place Reference Result")
        print("=" * 72)
        print("success      :", result["success"])
        print("failure cause:", result["failure_cause"])
        print("max lift [m] :", result["max_cube_lift_m"])
        print("goal xy [m]  :", result["goal_xy_error_m"])
        print("goal z [m]   :", result["goal_z_error_m"])
        print("released      :", result["released"])
        print("stable steps  :", result["stable_place_steps"])
        print("report         :", args.output)
        return 0 if result["success"] else 2

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
