#!/usr/bin/env python3
"""Day7: validate Cube/Goal randomization geometry and reproducibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_pickplace"))

from pickplace_week2_common import DEFAULT_TRAJECTORY, build_env, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--radius-mm", type=float, default=None)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pickplace_week2/day07_randomization_test.json"),
    )
    return p.parse_args()


def close_vec(a, b, atol=2e-5):
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=atol))


def run_case(config, args, *, name, mode, cube_angle, goal_angle, expected_cube, expected_goal):
    env, _ = build_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        seed=7000,
        randomization_mode=mode,
        cube_radius_mm=args.radius_mm,
        goal_radius_mm=args.radius_mm,
        fixed_cube_angle_deg=cube_angle,
        fixed_goal_angle_deg=goal_angle,
        apply_scaling=False,
    )
    try:
        _, info = env.reset(seed=7000)
        cube_offset = np.asarray(info["cube_offset_m"], dtype=float)
        goal_offset = np.asarray(info["goal_offset_m"], dtype=float)
        passed = close_vec(cube_offset, expected_cube) and close_vec(goal_offset, expected_goal)
        return {
            "name": name,
            "mode": mode,
            "cube_offset_m": cube_offset.tolist(),
            "goal_offset_m": goal_offset.tolist(),
            "expected_cube_offset_m": list(expected_cube),
            "expected_goal_offset_m": list(expected_goal),
            "passed": passed,
        }
    finally:
        env.close()


def main():
    args = parse_args()
    config = load_config()
    if args.radius_mm is None:
        args.radius_mm = float(config["week2"]["randomization_test_radius_mm"])
    r = args.radius_mm / 1000.0

    cases = [
        run_case(
            config, args,
            name="none", mode="none",
            cube_angle=None, goal_angle=None,
            expected_cube=[0, 0, 0], expected_goal=[0, 0, 0],
        ),
        run_case(
            config, args,
            name="cube_plus_x", mode="cube_only",
            cube_angle=0.0, goal_angle=None,
            expected_cube=[r, 0, 0], expected_goal=[0, 0, 0],
        ),
        run_case(
            config, args,
            name="goal_plus_y", mode="goal_only",
            cube_angle=None, goal_angle=90.0,
            expected_cube=[0, 0, 0], expected_goal=[0, r, 0],
        ),
        run_case(
            config, args,
            name="both_opposite", mode="both",
            cube_angle=180.0, goal_angle=270.0,
            expected_cube=[-r, 0, 0], expected_goal=[0, -r, 0],
        ),
    ]

    # Seed reproducibility without fixed angles.
    env, _ = build_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        seed=7777,
        randomization_mode="both",
        cube_radius_mm=args.radius_mm,
        goal_radius_mm=args.radius_mm,
        apply_scaling=False,
    )
    try:
        _, a = env.reset(seed=7777)
        _, b = env.reset(seed=7777)
        reproducible = close_vec(a["cube_offset_m"], b["cube_offset_m"], 1e-7) and close_vec(
            a["goal_offset_m"], b["goal_offset_m"], 1e-7
        )
        cases.append(
            {
                "name": "same_seed_reproducibility",
                "cube_offset_first_m": a["cube_offset_m"],
                "cube_offset_second_m": b["cube_offset_m"],
                "goal_offset_first_m": a["goal_offset_m"],
                "goal_offset_second_m": b["goal_offset_m"],
                "passed": reproducible,
            }
        )
    finally:
        env.close()

    passed = all(bool(c["passed"]) for c in cases)
    report = {
        "radius_mm": args.radius_mm,
        "cases": cases,
        "gate_passed": passed,
        "note": "Day7 validates geometry/reproducibility only, not task success under randomization.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for c in cases:
        print(f"{c['name']:<28} passed={c['passed']}")
    print("Gate:", passed)
    print("report:", args.output)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
