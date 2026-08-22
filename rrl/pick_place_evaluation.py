#!/usr/bin/env python3
"""Evaluation utilities and operational failure labels for Pick-and-Place."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


FAILURE_LABELS = [
    "grasp_timeout",
    "unstable_grasp",
    "insufficient_lift",
    "drop_during_transport",
    "place_alignment_failure",
    "place_height_failure",
    "release_failure",
    "unstable_placement",
    "insufficient_stable_steps",
    "base_env_truncated",
    "timeout",
    "other_failure",
]


def classify_pick_place_failure(info: dict[str, Any]) -> str:
    if bool(info.get("success", False)):
        return "success"
    reason = info.get("terminal_reason")
    if reason in FAILURE_LABELS:
        return str(reason)
    return "other_failure"


def run_episode(env, policy=None, *, max_steps=1200, seed=None):
    """Run one episode. policy=None means zero residual Reference-only."""
    observation, reset_info = env.reset(seed=seed)
    observation = np.asarray(observation, dtype=np.float32)
    action_dim = int(env.action_space.shape[0])

    total_return = 0.0
    max_lift = 0.0
    max_residual_l2 = 0.0
    max_abs_residual = 0.0
    saturation_steps = 0
    final_info = {}

    for step in range(int(max_steps)):
        if policy is None:
            action = np.zeros(action_dim, dtype=np.float32)
        else:
            action = np.asarray(policy(observation), dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        observation, reward, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation, dtype=np.float32)
        total_return += float(reward)
        max_lift = max(max_lift, float(info.get("max_cube_lift_m", 0.0)))
        max_residual_l2 = max(max_residual_l2, float(np.linalg.norm(action)))
        max_abs_residual = max(max_abs_residual, float(np.max(np.abs(action))))
        if np.any(np.abs(action) >= 0.999):
            saturation_steps += 1
        final_info = dict(info)
        if terminated or truncated:
            break
    else:
        final_info = dict(final_info)
        final_info["terminal_reason"] = "timeout"
        final_info["success"] = False

    steps = step + 1
    return {
        "success": bool(final_info.get("success", False)),
        "failure_cause": classify_pick_place_failure(final_info),
        "return": float(total_return),
        "steps": int(steps),
        "max_cube_lift_m": float(max_lift),
        "goal_xy_error_m": float(final_info.get("goal_xy_error_m", np.nan)),
        "goal_z_error_m": float(final_info.get("goal_z_error_m", np.nan)),
        "released": bool(final_info.get("released", False)),
        "stable_place_steps": int(final_info.get("stable_place_steps", 0)),
        "max_residual_l2": float(max_residual_l2),
        "max_abs_residual": float(max_abs_residual),
        "residual_saturation_rate": float(saturation_steps / max(steps, 1)),
        "terminal_reason": final_info.get("terminal_reason"),
        "cube_offset_m": reset_info.get("cube_offset_m"),
        "goal_offset_m": reset_info.get("goal_offset_m"),
    }


def summarize_rows(rows):
    n = len(rows)
    success_count = sum(int(r["success"]) for r in rows)
    failures = Counter(r["failure_cause"] for r in rows if not r["success"])
    return {
        "episodes": n,
        "success_count": success_count,
        "success_rate": success_count / n if n else 0.0,
        "failure_counts": dict(failures),
        "mean_return": float(np.mean([r["return"] for r in rows])) if rows else 0.0,
        "mean_goal_xy_error_m": float(np.nanmean([r["goal_xy_error_m"] for r in rows])) if rows else float("nan"),
    }
