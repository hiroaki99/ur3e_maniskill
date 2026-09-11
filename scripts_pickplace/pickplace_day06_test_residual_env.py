#!/usr/bin/env python3
"""Day6: zero-residual equivalence test.

Gate:
- ResidualPickPlaceEnv observation=(56,), action=(6,)
- zero residual keeps the fixed Week1 reference success rate >= 0.95
- when Week1 episodes.csv is available, success/failure outcome must match each seed
"""

from __future__ import annotations

import argparse
import csv
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
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--week1-episodes-csv",
        type=Path,
        default=Path("reports/pickplace_week1/day05_reference_eval/episodes.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/pickplace_week2/day06_zero_equivalence"),
    )
    return p.parse_args()


def read_week1(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        seed = int(row["seed"])
        result[seed] = {
            "success": str(row["success"]).lower() in {"true", "1"},
            "goal_xy_error_m": float(row.get("goal_xy_error_m", "nan")),
        }
    return result


def main():
    args = parse_args()
    config = load_config()
    episodes = int(
        args.episodes
        if args.episodes is not None
        else config["week2"]["zero_equivalence_episodes"]
    )
    min_rate = float(config["week2"]["zero_equivalence_min_success_rate"])
    week1 = read_week1(args.week1_episodes_csv)

    env, _ = build_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        seed=args.seed,
        randomization_mode="none",
        apply_scaling=False,
    )

    print("observation space:", env.observation_space)
    print("action space     :", env.action_space)
    assert env.observation_space.shape == (56,)
    assert env.action_space.shape == (6,)

    rows = []
    mismatch = 0
    try:
        for episode in range(episodes):
            seed = int(args.seed + episode)
            obs, reset_info = env.reset(seed=seed)
            assert np.asarray(obs).shape == (56,)
            assert np.all(np.isfinite(obs))

            total_return = 0.0
            final_info = {}
            for step in range(int(config["pick_place"]["max_episode_steps"])):
                obs, reward, terminated, truncated, info = env.step(
                    np.zeros(6, dtype=np.float32)
                )
                assert np.asarray(obs).shape == (56,)
                assert np.all(np.isfinite(obs))
                assert np.isfinite(reward)
                total_return += float(reward)
                final_info = dict(info)
                if terminated or truncated:
                    break
            success = bool(final_info.get("success", False))
            expected = week1.get(seed, {}).get("success")
            matched = True if expected is None else (success == expected)
            mismatch += int(not matched)
            row = {
                "episode": episode,
                "seed": seed,
                "week1_success": "N/A" if expected is None else expected,
                "zero_residual_success": success,
                "outcome_match": matched,
                "terminal_reason": final_info.get("terminal_reason"),
                "goal_xy_error_m": float(final_info.get("goal_xy_error_m", np.nan)),
                "goal_z_error_m": float(final_info.get("goal_z_error_m", np.nan)),
                "max_cube_lift_m": float(final_info.get("max_cube_lift_m", np.nan)),
                "return": total_return,
                "steps": step + 1,
            }
            rows.append(row)
            print(
                f"episode={episode:02d} seed={seed} success={success} "
                f"match={matched} reason={row['terminal_reason']} "
                f"goal_xy={row['goal_xy_error_m']:.5f}"
            )
    finally:
        env.close()

    success_count = sum(int(r["zero_residual_success"]) for r in rows)
    rate = success_count / episodes
    week1_comparable = bool(week1)
    gate = rate >= min_rate and (not week1_comparable or mismatch == 0)
    summary = {
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": rate,
        "minimum_success_rate": min_rate,
        "week1_csv_found": week1_comparable,
        "paired_outcome_mismatches": mismatch,
        "observation_dim": 56,
        "action_dim": 6,
        "gate_passed": gate,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("Day6 Zero-Residual Equivalence Gate")
    print("=" * 72)
    print(f"success: {success_count}/{episodes} = {rate:.3f}")
    print("paired mismatches:", mismatch if week1_comparable else "N/A")
    print("Gate:", gate)
    print("summary:", args.output_dir / "summary.json")
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
