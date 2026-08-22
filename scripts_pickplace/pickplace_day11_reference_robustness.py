#!/usr/bin/env python3
"""Day11: Reference-only robustness sweep for Cube / Goal / Both perturbations."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_pickplace"))

from pickplace_week3_common import (
    DEFAULT_TRAJECTORY,
    evaluate_reference_fixed_directions,
    load_config,
    write_csv,
    write_json,
)


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument(
        "--modes",
        default="cube_only,goal_only,both",
        help="comma separated subset of cube_only,goal_only,both",
    )
    p.add_argument(
        "--radii-mm",
        default=None,
        help="e.g. 0,5,10,15,20,25,30. Defaults to config.",
    )
    p.add_argument("--directions", type=int, default=None)
    p.add_argument(
        "--both-relation",
        choices=["same", "opposite"],
        default=None,
    )
    p.add_argument("--seed", type=int, default=11000)
    p.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=["physx_cpu", "physx_cuda"],
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pickplace_week3/day11_reference_sweep"),
    )
    return p.parse_args()


def select_radius(summary_rows, mode, lo, hi, target):
    candidates = [
        r for r in summary_rows
        if r["mode"] == mode and float(r["radius_mm"]) > 0.0
    ]
    in_band = [
        r for r in candidates
        if lo <= float(r["success_rate"]) <= hi
    ]
    pool = in_band if in_band else candidates
    if not pool:
        return None
    selected = min(
        pool,
        key=lambda r: (
            abs(float(r["success_rate"]) - target),
            float(r["radius_mm"]),
        ),
    )
    return {
        "radius_mm": float(selected["radius_mm"]),
        "success_rate": float(selected["success_rate"]),
        "target_band_found": bool(in_band),
        "selection_rule": (
            "closest to target success within target band"
            if in_band
            else "no target-band radius; closest tested radius to target success"
        ),
    }


def main():
    args = parse_args()
    cfg = load_config()
    sweep = cfg["week3"]["reference_sweep"]
    radii = (
        parse_float_list(args.radii_mm)
        if args.radii_mm is not None
        else [float(x) for x in sweep["radii_mm"]]
    )
    directions = int(
        args.directions
        if args.directions is not None
        else sweep["num_directions"]
    )
    both_relation = (
        args.both_relation
        if args.both_relation is not None
        else str(sweep.get("both_relation", "same"))
    )
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    valid = {"cube_only", "goal_only", "both"}
    if any(m not in valid for m in modes):
        raise ValueError(f"modes must be subset of {sorted(valid)}")

    max_steps = int(cfg["week3"]["td3"]["max_steps_per_episode"])
    all_rows = []
    summary_rows = []

    print("=" * 76)
    print("Pick-and-Place Day11 Reference Robustness Sweep")
    print("=" * 76)
    print("modes:", modes)
    print("radii [mm]:", radii)
    print("directions:", directions)
    print("both relation:", both_relation)
    print()

    case_seed = int(args.seed)
    for mode_index, mode in enumerate(modes):
        for radius_index, radius in enumerate(radii):
            rows = evaluate_reference_fixed_directions(
                config=cfg,
                trajectory=args.trajectory,
                sim_backend=args.sim_backend,
                mode=mode,
                radius_mm=float(radius),
                num_directions=directions,
                seed=case_seed + mode_index * 100000 + radius_index * 1000,
                max_steps=max_steps,
                both_relation=both_relation,
            )
            all_rows.extend(rows)
            n = len(rows)
            successes = sum(int(r["success"]) for r in rows)
            failures = Counter(
                r["failure_cause"] for r in rows if not r["success"]
            )
            summary = {
                "mode": mode,
                "radius_mm": float(radius),
                "episodes": n,
                "success_count": successes,
                "success_rate": successes / n if n else 0.0,
                "failure_counts": dict(failures),
                "mean_goal_xy_error_m": float(
                    np.nanmean([r["goal_xy_error_m"] for r in rows])
                ),
            }
            summary_rows.append(summary)
            print(
                f"{mode:10s} radius={radius:5.1f} mm "
                f"success={successes:2d}/{n:2d}={summary['success_rate']:.3f} "
                f"failures={dict(failures)}"
            )

    lo = float(sweep["target_success_rate_min"])
    hi = float(sweep["target_success_rate_max"])
    target = float(sweep["selection_target_success_rate"])

    selected = {
        mode: select_radius(summary_rows, mode, lo, hi, target)
        for mode in modes
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", all_rows)
    # failure_counts is a dict, store JSON string for CSV readability.
    csv_rows = []
    for row in summary_rows:
        csv_rows.append(
            {
                **{k: v for k, v in row.items() if k != "failure_counts"},
                "failure_counts": str(row["failure_counts"]),
            }
        )
    write_csv(args.output_dir / "summary.csv", csv_rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "modes": modes,
            "radii_mm": radii,
            "directions": directions,
            "both_relation": both_relation,
            "target_success_band": [lo, hi],
            "selection_target_success_rate": target,
            "results": summary_rows,
            "selected_training_radii": selected,
            "note": (
                "If target_band_found is false, expand the sweep before formal "
                "training. A radius is not validated merely because it was selected."
            ),
        },
    )
    write_json(
        args.output_dir / "selected_training_radii.json",
        selected,
    )

    print()
    print("=" * 76)
    print("Day11 radius suggestions")
    print("=" * 76)
    for mode, item in selected.items():
        print(mode, ":", item)
    print("summary:", args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
