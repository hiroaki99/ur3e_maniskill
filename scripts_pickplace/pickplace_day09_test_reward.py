#!/usr/bin/env python3
"""Day9: unit + integration tests for the Pick-and-Place reward."""

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
from rrl.pick_place_reward import ResidualPickPlaceReward

LAYOUT = {
    "arm_qpos": [0, 6],
    "arm_qvel": [6, 12],
    "grasp_center": [12, 15],
    "cube_position": [15, 18],
    "goal_position": [18, 21],
    "reference_qpos": [21, 27],
    "reference_error": [27, 33],
    "cube_to_grasp": [33, 36],
    "cube_to_goal": [36, 39],
    "grasp_to_goal": [39, 42],
    "phase_onehot": [42, 52],
    "contact_flags": [52, 54],
    "cube_lift": [54, 55],
    "phase_progress": [55, 56],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pickplace_week2/day09_reward_test.json"),
    )
    return p.parse_args()


def obs(**fields):
    x = np.zeros(56, dtype=np.float32)
    for name, value in fields.items():
        start, stop = LAYOUT[name]
        x[start:stop] = np.asarray(value, dtype=np.float32).reshape(stop - start)
    return x


def compute(model, prev, cur, phase, **info_extra):
    info = {
        "phase_name_before": phase,
        "left_contact": False,
        "right_contact": False,
        "within_goal_xy": False,
        "within_goal_z": False,
        "released": False,
        "stable_place_steps": 0,
        "success": False,
        **info_extra,
    }
    return model.compute(prev, cur, np.zeros(6, dtype=np.float32), info, False)


def main():
    args = parse_args()
    config = load_config()
    model = ResidualPickPlaceReward(
        config["reward_pick_place"], LAYOUT, config["residual_pick_place"]["alpha"]
    )
    tests = []

    # Approach: 50 mm -> 40 mm should be positive; reverse should be negative.
    prev = obs(cube_to_grasp=[0.05, 0, 0])
    cur = obs(cube_to_grasp=[0.04, 0, 0])
    r, terms = compute(model, prev, cur, "to_pregrasp")
    tests.append({"name": "approach_closer_positive", "passed": terms["approach"] > 0, "value": terms["approach"]})
    r, terms = compute(model, cur, prev, "to_pregrasp")
    tests.append({"name": "approach_farther_negative", "passed": terms["approach"] < 0, "value": terms["approach"]})

    # Grasp and hold bonuses require bilateral contact in the correct phases.
    r, terms = compute(
        model, obs(), obs(), "grasp", left_contact=True, right_contact=True
    )
    tests.append({"name": "bilateral_grasp_bonus", "passed": terms["grasp"] > 0, "value": terms["grasp"]})
    r, terms = compute(
        model, obs(), obs(), "transport", left_contact=True, right_contact=True
    )
    tests.append({"name": "transport_hold_bonus", "passed": terms["hold"] > 0, "value": terms["hold"]})

    # Lift progress is rewarded only in lift-related phases.
    prev = obs(cube_lift=[0.020])
    cur = obs(cube_lift=[0.025])
    r, terms = compute(model, prev, cur, "lift")
    tests.append({"name": "lift_up_positive", "passed": terms["lift"] > 0, "value": terms["lift"]})
    r, terms = compute(model, prev, cur, "descend_place")
    tests.append({"name": "no_lift_bonus_during_place", "passed": abs(terms["lift"]) < 1e-12, "value": terms["lift"]})

    # Transport: reduce XY cube-to-goal distance.
    prev = obs(cube_to_goal=[0.10, 0, 0.05])
    cur = obs(cube_to_goal=[0.09, 0, 0.05])
    r, terms = compute(model, prev, cur, "transport")
    tests.append({"name": "transport_closer_positive", "passed": terms["transport"] > 0, "value": terms["transport"]})

    # Placement: reduce 3-D cube-to-goal distance.
    prev = obs(cube_to_goal=[0.005, 0, 0.030])
    cur = obs(cube_to_goal=[0.005, 0, 0.020])
    r, terms = compute(model, prev, cur, "descend_place")
    tests.append({"name": "place_closer_positive", "passed": terms["place"] > 0, "value": terms["place"]})

    # Release bonus only near goal and released.
    r, terms = compute(
        model, obs(), obs(), "release",
        within_goal_xy=True, within_goal_z=True, released=True,
    )
    tests.append({"name": "release_near_goal_bonus", "passed": terms["release"] > 0, "value": terms["release"]})
    r, terms = compute(
        model, obs(), obs(), "release",
        within_goal_xy=False, within_goal_z=True, released=True,
    )
    tests.append({"name": "no_release_bonus_away_from_goal", "passed": abs(terms["release"]) < 1e-12, "value": terms["release"]})

    # Residual penalty is zero for zero action and negative otherwise.
    info = {"phase_name_before": "transport", "success": False}
    _, zero_terms = model.compute(obs(), obs(), np.zeros(6), info, False)
    _, nonzero_terms = model.compute(obs(), obs(), np.ones(6), info, False)
    tests.append({"name": "zero_residual_no_penalty", "passed": abs(zero_terms["residual_penalty"]) < 1e-12, "value": zero_terms["residual_penalty"]})
    tests.append({"name": "nonzero_residual_penalty", "passed": nonzero_terms["residual_penalty"] < 0, "value": nonzero_terms["residual_penalty"]})

    # Safety penalties are intentionally disabled until instrumentation is validated.
    unsafe_info = {
        "phase_name_before": "transport",
        "unsafe_collision": True,
        "left_contact_force": 1000.0,
        "right_contact_force": 1000.0,
        "success": False,
    }
    _, safety_terms = model.compute(obs(), obs(), np.zeros(6), unsafe_info, False)
    tests.append({"name": "collision_penalty_disabled_week2", "passed": abs(safety_terms["unsafe_collision"]) < 1e-12, "value": safety_terms["unsafe_collision"]})
    tests.append({"name": "force_penalty_disabled_week2", "passed": abs(safety_terms["excessive_force"]) < 1e-12, "value": safety_terms["excessive_force"]})

    # Terminal success/failure signs.
    success_info = {"phase_name_before": "final_settle", "success": True}
    _, success_terms = model.compute(obs(), obs(), np.zeros(6), success_info, True)
    fail_info = {"phase_name_before": "final_settle", "success": False}
    _, fail_terms = model.compute(obs(), obs(), np.zeros(6), fail_info, True)
    tests.append({"name": "terminal_success_positive", "passed": success_terms["terminal"] > 0, "value": success_terms["terminal"]})
    tests.append({"name": "terminal_failure_negative", "passed": fail_terms["terminal"] < 0, "value": fail_terms["terminal"]})

    # One real fixed-condition integration episode: finite reward + success.
    env, _ = build_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        seed=9000,
        randomization_mode="none",
        apply_scaling=False,
    )
    total_return = 0.0
    final_info = {}
    finite = True
    try:
        ob, _ = env.reset(seed=9000)
        for step in range(int(config["pick_place"]["max_episode_steps"])):
            ob, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
            finite = finite and bool(np.isfinite(reward)) and bool(np.all(np.isfinite(ob)))
            total_return += float(reward)
            final_info = dict(info)
            if terminated or truncated:
                break
    finally:
        env.close()
    integration_pass = finite and bool(final_info.get("success", False))
    tests.append({"name": "fixed_reference_integration", "passed": integration_pass, "return": total_return, "terminal_reason": final_info.get("terminal_reason")})

    gate = all(bool(t["passed"]) for t in tests)
    report = {
        "tests": tests,
        "gate_passed": gate,
        "safety_note": (
            "Force/collision penalties are disabled in Week2 because collision instrumentation "
            "has not passed a physical positive-control validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for t in tests:
        print(f"{t['name']:<40} passed={t['passed']}")
    print("Gate:", gate)
    print("report:", args.output)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
