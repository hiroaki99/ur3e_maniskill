#!/usr/bin/env python3
"""Day10: final Week2 gate before TD3 training.

This is not a performance benchmark. It verifies that the complete wrapper stack
can be reset/stepped under nominal, Cube-only, Goal-only, and Both randomization,
with 56-D scaled observations, 6-D residual actions, finite rewards, and
structured failure diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_pickplace"))

from pickplace_week2_common import DEFAULT_TRAJECTORY, build_env, load_config, write_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--radius-mm", type=float, default=10.0)
    p.add_argument("--directions", type=int, default=None)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pickplace_week2/day10_training_interface"),
    )
    return p.parse_args()


def scalar(value, default=np.nan):
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def run_one(env, seed, max_steps):
    obs, reset_info = env.reset(seed=seed)
    if np.asarray(obs).shape != (56,) or not np.all(np.isfinite(obs)):
        raise RuntimeError("invalid reset observation")
    total_return = 0.0
    reward_term_keys = set()
    final_info = {}
    for step in range(max_steps):
        action = np.zeros(6, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if np.asarray(obs).shape != (56,) or not np.all(np.isfinite(obs)):
            raise RuntimeError("invalid step observation")
        if not np.isfinite(reward):
            raise RuntimeError("non-finite reward")
        total_return += float(reward)
        reward_term_keys.update(info.get("reward_terms", {}).keys())
        final_info = dict(info)
        if terminated or truncated:
            break
    return {
        "success": bool(final_info.get("success", False)),
        "terminal_reason": final_info.get("terminal_reason"),
        "return": total_return,
        "steps": step + 1,
        "goal_xy_error_m": scalar(final_info.get("goal_xy_error_m")),
        "goal_z_error_m": scalar(final_info.get("goal_z_error_m")),
        "max_cube_lift_m": scalar(final_info.get("max_cube_lift_m")),
        "cube_offset_m": reset_info.get("cube_offset_m"),
        "goal_offset_m": reset_info.get("goal_offset_m"),
        "reward_term_keys": sorted(reward_term_keys),
    }


def main():
    args = parse_args()
    config = load_config()
    directions = int(
        args.directions
        if args.directions is not None
        else config["week2"]["training_interface_smoke_directions"]
    )
    angles = np.linspace(0.0, 360.0, directions, endpoint=False)
    modes = ["none", "cube_only", "goal_only", "both"]
    rows = []
    technical_ok = True

    for mode_i, mode in enumerate(modes):
        mode_angles = [0.0] if mode == "none" else list(angles)
        for i, angle in enumerate(mode_angles):
            cube_angle = float(angle) if mode in {"cube_only", "both"} else None
            # Offset Goal direction by 90 deg in the Both smoke test so the two
            # random variables are not accidentally identical.
            goal_angle = (
                float((angle + 90.0) % 360.0)
                if mode == "both"
                else (float(angle) if mode == "goal_only" else None)
            )
            env, _ = build_env(
                config=config,
                trajectory=args.trajectory,
                sim_backend=args.sim_backend,
                seed=args.seed,
                randomization_mode=mode,
                cube_radius_mm=(0.0 if mode == "none" else args.radius_mm),
                goal_radius_mm=(0.0 if mode == "none" else args.radius_mm),
                fixed_cube_angle_deg=cube_angle,
                fixed_goal_angle_deg=goal_angle,
                apply_scaling=True,
            )
            try:
                if env.observation_space.shape != (56,) or env.action_space.shape != (6,):
                    technical_ok = False
                seed = int(args.seed + mode_i * 100 + i)
                result = run_one(env, seed, int(config["pick_place"]["max_episode_steps"]))
            finally:
                env.close()

            row = {
                "mode": mode,
                "angle_deg": angle,
                "seed": seed,
                "success": result["success"],
                "terminal_reason": result["terminal_reason"],
                "return": result["return"],
                "steps": result["steps"],
                "goal_xy_error_m": result["goal_xy_error_m"],
                "goal_z_error_m": result["goal_z_error_m"],
                "max_cube_lift_m": result["max_cube_lift_m"],
                "cube_offset_m": json.dumps(result["cube_offset_m"]),
                "goal_offset_m": json.dumps(result["goal_offset_m"]),
            }
            rows.append(row)
            print(
                f"mode={mode:<9} angle={angle:6.1f} "
                f"success={row['success']} reason={row['terminal_reason']} "
                f"goal_xy={row['goal_xy_error_m']:.5f}"
            )

    nominal_rows = [r for r in rows if r["mode"] == "none"]
    nominal_ok = all(r["success"] for r in nominal_rows)
    failure_counts = {}
    for mode in modes:
        c = Counter(
            r["terminal_reason"] for r in rows if r["mode"] == mode and not r["success"]
        )
        failure_counts[mode] = dict(c)

    gate = technical_ok and nominal_ok
    summary = {
        "radius_mm": args.radius_mm,
        "directions": directions,
        "observation_dim": 56,
        "action_dim": 6,
        "observation_scaling": config["observation_scaling"],
        "technical_interface_ok": technical_ok,
        "nominal_reference_success": nominal_ok,
        "failure_counts": failure_counts,
        "week2_gate_passed": gate,
        "interpretation_note": (
            "Randomized success rates in this smoke test are diagnostic only. "
            "Week3 performs the formal Reference robustness sweep before training."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("Week2 Training-Interface Gate")
    print("=" * 72)
    print("technical interface:", technical_ok)
    print("nominal success    :", nominal_ok)
    print("randomized failures:", failure_counts)
    print("Gate:", gate)
    print("summary:", args.output_dir / "summary.json")
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
