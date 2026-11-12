#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate object-yaw-aware grasp orientation + adaptive force-band squeeze + actual-state relative micro-lift over many seeds.

This script intentionally does NOT use any forced A yaw or forced A XY position.
It launches the actual-state-replan smoke test for each randomized reset and
summarizes success/failure without stopping at the first failed seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "expert_gp"
            / "alignment"
            / "multiseed_yaw_aware"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    test_script = (
        PROJECT_ROOT
        / "scripts_expert"
        / "test_alignment_grasp_A_yaw_aware.py"
    )

    if not test_script.exists():
        raise FileNotFoundError(
            f"Missing test script: {test_script}"
        )

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for seed in range(
        args.start_seed,
        args.start_seed
        + args.num_seeds,
    ):
        report_path = (
            output_dir
            / f"seed_{seed:03d}.json"
        )

        command = [
            sys.executable,
            str(test_script),
            "--seed",
            str(seed),
            "--output",
            str(report_path),
        ]

        print(
            "\n" + "=" * 72
        )
        print(
            f"Seed {seed}"
        )
        print(
            "=" * 72
        )

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )

        if report_path.exists():
            with report_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                report = json.load(f)

            row = {
                "seed":
                    seed,
                "process_returncode":
                    int(
                        completed.returncode
                    ),
                "success":
                    bool(
                        report.get(
                            "success",
                            False,
                        )
                    ),
                "failure_reasons":
                    report.get(
                        "failure_reasons",
                        [],
                    ),
                "A_lift_mm":
                    float(
                        report.get(
                            "A_lift_mm",
                            float("nan"),
                        )
                    ),
                "first_bilateral_contact_loss_waypoint":
                    report.get(
                        "first_bilateral_contact_loss_waypoint",
                        None,
                    ),
                "max_left_contact_force_N":
                    float(
                        report.get(
                            "max_left_contact_force_N",
                            float("nan"),
                        )
                    ),
                "max_right_contact_force_N":
                    float(
                        report.get(
                            "max_right_contact_force_N",
                            float("nan"),
                        )
                    ),
            }
        else:
            row = {
                "seed":
                    seed,
                "process_returncode":
                    int(
                        completed.returncode
                    ),
                "success":
                    False,
                "failure_reasons":
                    [
                        "no_report"
                    ],
                "A_lift_mm":
                    float("nan"),
                "first_bilateral_contact_loss_waypoint":
                    None,
                "max_left_contact_force_N":
                    float("nan"),
                "max_right_contact_force_N":
                    float("nan"),
            }

        rows.append(
            row
        )

        print(
            "summary:",
            row,
        )

    success_count = sum(
        int(
            row["success"]
        )
        for row in rows
    )

    summary = {
        "start_seed":
            int(
                args.start_seed
            ),
        "num_seeds":
            int(
                args.num_seeds
            ),
        "success_count":
            int(
                success_count
            ),
        "success_rate":
            float(
                success_count
                / max(
                    1,
                    len(rows),
                )
            ),
        "results":
            rows,
    }

    summary_path = (
        output_dir
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n" + "=" * 72
    )
    print(
        "MULTI-SEED SUMMARY"
    )
    print(
        "=" * 72
    )
    print(
        f"success: {success_count}/{len(rows)}"
    )
    print(
        f"success_rate: {summary['success_rate']:.3f}"
    )
    print(
        "summary:",
        summary_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
