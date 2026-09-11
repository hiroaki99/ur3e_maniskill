#!/usr/bin/env python3
"""Day8B: prove that only the three relative fields are scaled for TD3."""

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

TARGET_FIELDS = {"cube_to_grasp", "cube_to_goal", "grasp_to_goal"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--radius-mm", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=8100)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pickplace_week2/day08_scaling_test.json"),
    )
    return p.parse_args()


def check(raw, scaled, layout, scales):
    raw = np.asarray(raw, dtype=np.float32)
    scaled = np.asarray(scaled, dtype=np.float32)
    if raw.shape != (56,) or scaled.shape != (56,):
        return False, "shape"
    expected = raw.copy()
    for name in TARGET_FIELDS:
        start, stop = layout[name]
        expected[start:stop] *= np.float32(scales[name])
    if not np.allclose(scaled, expected, atol=1e-6, rtol=1e-6):
        return False, "value"
    return True, "ok"


def main():
    args = parse_args()
    config = load_config()
    env, _ = build_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        seed=args.seed,
        randomization_mode="both",
        cube_radius_mm=args.radius_mm,
        goal_radius_mm=args.radius_mm,
        apply_scaling=True,
    )
    inner = env.env
    layout = dict(inner.observation_layout)
    scales = dict(config["observation_scaling"])
    checks = []
    try:
        scaled_obs, _ = env.reset(seed=args.seed)
        raw_obs = inner._get_observation(inner.last_metrics)
        passed, reason = check(raw_obs, scaled_obs, layout, scales)
        checks.append({"label": "reset", "passed": passed, "reason": reason})

        for i in range(args.steps):
            scaled_obs, reward, terminated, truncated, info = env.step(
                np.zeros(6, dtype=np.float32)
            )
            raw_obs = inner._get_observation(inner.last_metrics)
            passed, reason = check(raw_obs, scaled_obs, layout, scales)
            reward_sum = float(sum(info.get("reward_terms", {}).values()))
            reward_consistent = bool(np.isclose(float(reward), reward_sum, atol=1e-8))
            checks.append(
                {
                    "label": f"step_{i+1}",
                    "passed": bool(passed and reward_consistent),
                    "reason": reason,
                    "reward_consistent": reward_consistent,
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    gate = all(c["passed"] for c in checks)
    report = {
        "configured_scales": scales,
        "checks": checks,
        "gate_passed": gate,
        "contract": (
            "Only cube_to_grasp, cube_to_goal, grasp_to_goal are changed. "
            "Reward remains the raw-environment reward."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("checks:", len(checks))
    print("Gate:", gate)
    print("report:", args.output)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
