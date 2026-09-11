#!/usr/bin/env python3
"""Day8A: audit raw 56-D observation ranges before choosing TD3 scaling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_pickplace"))

from pickplace_week2_common import DEFAULT_TRAJECTORY, build_env, load_config, write_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--radius-mm", type=float, default=None)
    p.add_argument("--seed", type=int, default=8000)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pickplace_week2/day08_observation_audit"),
    )
    return p.parse_args()


def round_candidate(scale):
    candidates = np.asarray([1, 2, 5, 10, 20, 50, 100], dtype=float)
    return float(candidates[np.argmin(np.abs(candidates - scale))])


def main():
    args = parse_args()
    config = load_config()
    episodes = int(args.episodes or config["week2"]["observation_audit_episodes"])
    radius_mm = float(
        args.radius_mm
        if args.radius_mm is not None
        else config["week2"]["observation_audit_radius_mm"]
    )

    env, _ = build_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        seed=args.seed,
        randomization_mode="both",
        cube_radius_mm=radius_mm,
        goal_radius_mm=radius_mm,
        apply_scaling=False,
    )
    layout = dict(env.observation_layout)
    samples = []
    phase_counts = {}
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            samples.append(np.asarray(obs, dtype=np.float64).copy())
            for _ in range(int(config["pick_place"]["max_episode_steps"])):
                obs, reward, terminated, truncated, info = env.step(
                    np.zeros(6, dtype=np.float32)
                )
                if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                    raise RuntimeError("non-finite observation/reward")
                samples.append(np.asarray(obs, dtype=np.float64).copy())
                phase = str(info.get("phase_name", "unknown"))
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                if terminated or truncated:
                    break
    finally:
        env.close()

    x = np.stack(samples, axis=0)
    rows = []
    recommendations = {}
    for name, (start, stop) in layout.items():
        block = x[:, start:stop]
        abs_values = np.abs(block).reshape(-1)
        p95 = float(np.percentile(abs_values, 95))
        target = 1.0 if p95 <= 1e-12 else float(np.clip(1.0 / p95, 1.0, 100.0))
        candidate = round_candidate(target)
        rows.append(
            {
                "field": name,
                "start": start,
                "stop": stop,
                "mean": float(np.mean(block)),
                "std": float(np.std(block)),
                "min": float(np.min(block)),
                "max": float(np.max(block)),
                "p95_abs": p95,
                "scale_to_p95_about_1": candidate,
            }
        )
        if name in {"cube_to_grasp", "cube_to_goal", "grasp_to_goal"}:
            recommendations[name] = candidate

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "observation_field_stats.csv", rows)
    report = {
        "episodes": episodes,
        "radius_mm": radius_mm,
        "sample_count": int(x.shape[0]),
        "observation_dim": int(x.shape[1]),
        "phase_counts": phase_counts,
        "relative_feature_scale_recommendation": recommendations,
        "configured_candidate_scales": config["observation_scaling"],
        "note": (
            "Recommended scales are magnitude diagnostics, not tuned hyperparameters. "
            "Week3 should keep a selected scaling fixed during controlled comparisons."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("observation dim:", x.shape[1])
    for row in rows:
        if row["field"] in {"cube_to_grasp", "cube_to_goal", "grasp_to_goal", "reference_error"}:
            print(
                f"{row['field']:<18} p95_abs={row['p95_abs']:.5f} "
                f"candidate_scale={row['scale_to_p95_about_1']:.1f}"
            )
    print("configured scales:", config["observation_scaling"])
    print("summary:", args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
