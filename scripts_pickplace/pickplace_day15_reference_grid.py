#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(
    0,
    str(REPO_ROOT / "scripts_pickplace"),
)

from pickplace_week3_common import (
    DEFAULT_TRAJECTORY,
    build_fixed_eval_env,
    load_config,
    set_fixed_angles,
    write_csv,
    write_json,
)

from rrl.pick_place_evaluation import run_episode


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--radius-mm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--directions",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3350,
    )

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
        "--output-dir",
        type=Path,
        default=Path(
            "reports/"
            "pickplace_week3/"
            "day15_both10_reference_grid"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config()

    max_steps = int(
        config["week3"]["td3"][
            "max_steps_per_episode"
        ]
    )

    env, randomized = build_fixed_eval_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        mode="both",
        radius_mm=args.radius_mm,
        seed=args.seed,
        apply_scaling=True,
    )

    directions = np.linspace(
        0.0,
        360.0,
        args.directions,
        endpoint=False,
    )

    rows = []

    case_index = 0

    try:
        for cube_angle in directions:
            for goal_angle in directions:

                set_fixed_angles(
                    randomized,
                    "both",
                    float(cube_angle),
                    float(goal_angle),
                )

                episode_seed = (
                    args.seed
                    + case_index
                )

                result = run_episode(
                    env,
                    policy=None,
                    max_steps=max_steps,
                    seed=episode_seed,
                )

                row = {
                    "case": case_index,
                    "cube_angle_deg":
                        float(cube_angle),
                    "goal_angle_deg":
                        float(goal_angle),
                    "seed":
                        int(episode_seed),
                    **result,
                }

                rows.append(row)

                print(
                    f"case={case_index:02d} "
                    f"cube={cube_angle:6.1f} "
                    f"goal={goal_angle:6.1f} "
                    f"success={result['success']} "
                    f"reason="
                    f"{result['failure_cause']}"
                )

                case_index += 1

    finally:
        env.close()

    successes = sum(
        int(row["success"])
        for row in rows
    )

    failures = Counter(
        row["failure_cause"]
        for row in rows
        if not row["success"]
    )

    summary = {
        "radius_mm":
            float(args.radius_mm),

        "directions":
            int(args.directions),

        "cases":
            len(rows),

        "success_count":
            successes,

        "success_rate":
            successes / len(rows),

        "failure_counts":
            dict(failures),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        args.output_dir / "episodes.csv",
        rows,
    )

    write_json(
        args.output_dir / "summary.json",
        summary,
    )

    print()
    print("=" * 70)
    print("Day15 Both Reference Grid")
    print("=" * 70)

    print(
        f"Reference: "
        f"{successes}/{len(rows)} "
        f"= {summary['success_rate']:.3f}"
    )

    print(
        "Failures:",
        dict(failures),
    )

    print(
        "summary:",
        args.output_dir / "summary.json",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())