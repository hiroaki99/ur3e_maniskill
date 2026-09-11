#!/usr/bin/env python3
"""Optional Week3 diagnostic: inspect deterministic Actor actions by direction."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_pickplace"))

from pickplace_week3_common import (
    DEFAULT_TRAJECTORY,
    build_fixed_eval_env,
    load_config,
    make_td3,
    resolve_device,
    set_fixed_angles,
    write_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--mode",
        required=True,
        choices=["cube_only", "goal_only", "both"],
    )
    p.add_argument("--radius-mm", type=float, required=True)
    p.add_argument("--directions", type=int, default=8)
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--seed", type=int, default=3400)
    p.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=["physx_cpu", "physx_cuda"],
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    device = resolve_device(args.device)
    env, randomized = build_fixed_eval_env(
        config=cfg,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        mode=args.mode,
        radius_mm=args.radius_mm,
        seed=args.seed,
        apply_scaling=True,
    )
    try:
        policy = make_td3(cfg, env, device=device, seed=args.seed)
        policy.load(args.checkpoint)

        rows = []
        directions = np.linspace(
            0.0, 360.0, args.directions, endpoint=False
        )
        for i, angle in enumerate(directions):
            if args.mode == "cube_only":
                set_fixed_angles(randomized, args.mode, angle, None)
            elif args.mode == "goal_only":
                set_fixed_angles(randomized, args.mode, None, angle)
            else:
                set_fixed_angles(randomized, args.mode, angle, angle)

            obs, _ = env.reset(seed=args.seed + i)
            obs = np.asarray(obs, dtype=np.float32)

            # Record Actor at the first TD3-visible state of each phase.
            seen = set()
            for step in range(int(cfg["week3"]["td3"]["max_steps_per_episode"])):
                phase = int(np.argmax(obs[42:52]))
                if phase not in seen:
                    action = policy.select_action(obs)
                    row = {
                        "angle_deg": float(angle),
                        "phase": phase,
                    }
                    for j, value in enumerate(action):
                        row[f"joint_{j}"] = float(value)
                    rows.append(row)
                    seen.add(phase)
                action = policy.select_action(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                if terminated or truncated:
                    break

        # Summary of action variability by phase/joint
        summaries = []
        phases = sorted(set(r["phase"] for r in rows))
        for phase in phases:
            pr = [r for r in rows if r["phase"] == phase]
            for j in range(6):
                vals = np.asarray([r[f"joint_{j}"] for r in pr], dtype=float)
                summaries.append(
                    {
                        "phase": int(phase),
                        "joint": j,
                        "mean": float(np.mean(vals)),
                        "std_across_directions": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    }
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "actions_by_direction.csv", rows)
        write_csv(args.output_dir / "action_variability.csv", summaries)
        write_json(
            args.output_dir / "summary.json",
            {
                "checkpoint": str(args.checkpoint),
                "mode": args.mode,
                "radius_mm": args.radius_mm,
                "directions": args.directions,
                "note": (
                    "Low std across directions suggests a fixed residual bias rather "
                    "than perturbation-conditioned correction. Saturation should be "
                    "checked together with success, not alone."
                ),
            },
        )

        print("report:", args.output_dir / "action_variability.csv")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
