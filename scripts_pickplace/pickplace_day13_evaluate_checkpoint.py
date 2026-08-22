#!/usr/bin/env python3
"""Week3 matched evaluation of one Pick-and-Place TD3 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_pickplace"))

from pickplace_week3_common import (
    DEFAULT_TRAJECTORY,
    build_fixed_eval_env,
    evaluate_paired_policy,
    load_config,
    make_td3,
    resolve_device,
    summarize_paired,
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
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--directions", type=int, default=8)
    p.add_argument("--episodes-per-direction", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=3300)
    p.add_argument(
        "--both-relation",
        choices=["same", "opposite"],
        default="same",
    )
    p.add_argument(
        "--both-grid",
        action="store_true",
        help="For mode=both, evaluate all directions x directions.",
    )
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
    max_steps = int(cfg["week3"]["td3"]["max_steps_per_episode"])

    probe_env, _ = build_fixed_eval_env(
        config=cfg,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        mode=args.mode,
        radius_mm=args.radius_mm,
        seed=args.eval_seed,
        apply_scaling=True,
    )
    try:
        policy = make_td3(
            cfg, probe_env, device=device, seed=args.eval_seed
        )
        policy.load(args.checkpoint)
    finally:
        probe_env.close()

    rows = evaluate_paired_policy(
        config=cfg,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        mode=args.mode,
        radius_mm=args.radius_mm,
        num_directions=args.directions,
        episodes_per_direction=args.episodes_per_direction,
        seed=args.eval_seed,
        max_steps=max_steps,
        policy=policy,
        both_relation=args.both_relation,
        both_grid=args.both_grid,
    )
    summary = summarize_paired(rows)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "mode": args.mode,
            "radius_mm": args.radius_mm,
            "directions": args.directions,
            "episodes_per_direction": args.episodes_per_direction,
            "both_relation": args.both_relation,
            "both_grid": bool(args.both_grid),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_results.csv", rows)
    write_json(args.output_dir / "summary.json", summary)

    print("=" * 76)
    print("Week3 Matched Checkpoint Evaluation")
    print("=" * 76)
    print(
        f"Reference: {summary['reference_success_count']}/"
        f"{summary['episodes']} = {summary['reference_success_rate']:.3f}"
    )
    print(
        f"Residual : {summary['residual_success_count']}/"
        f"{summary['episodes']} = {summary['residual_success_rate']:.3f}"
    )
    print(
        "rescued/broken/net:",
        summary["rescued"],
        summary["broken"],
        summary["net_gain"],
    )
    print("Reference failures:", summary["reference_failure_counts"])
    print("Residual failures :", summary["residual_failure_counts"])
    print("summary:", args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
