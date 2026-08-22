#!/usr/bin/env python3
"""Shared utilities for Randomized Pick-and-Place Week3.

Week3 reuses the validated Week2 wrapper stack:
    UR3ePickPlace-v0
      -> PickPlaceRandomizationWrapper
      -> ResidualPickPlaceEnv
      -> PickPlaceObservationScaleWrapper

This module adds:
- small ReplayBuffer
- TD3 builder
- fixed-direction paired Reference/Residual evaluation
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_pickplace"))

from pickplace_week2_common import build_env
from rrl.pick_place_evaluation import run_episode
from rrl.td3 import TD3

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"
DEFAULT_TRAJECTORY = REPO_ROOT / "trajectories" / "pick_place_reference_v1.json"


def load_config(path: Path = CONFIG_PATH):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(requested: str):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=200_000, seed=0):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(int(seed))
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = float(reward)
        self.not_done[self.ptr] = 1.0 - float(bool(done))
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise RuntimeError(
                f"ReplayBuffer too small: size={self.size}, batch={batch_size}"
            )
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return (
            self.state[idx],
            self.action[idx],
            self.next_state[idx],
            self.reward[idx],
            self.not_done[idx],
        )

    def __len__(self):
        return self.size


def make_td3(config, env, *, device: str, seed: int):
    td3 = config["week3"]["td3"]
    return TD3(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        max_action=float(env.action_space.high[0]),
        actor_hidden_dim=int(td3["actor_hidden_dim"]),
        critic_hidden_dim=int(td3["critic_hidden_dim"]),
        discount=float(td3["discount"]),
        tau=float(td3["tau"]),
        policy_noise=float(td3["policy_noise"]),
        noise_clip=float(td3["noise_clip"]),
        policy_freq=int(td3["policy_freq"]),
        actor_lr=float(td3["actor_lr"]),
        critic_lr=float(td3["critic_lr"]),
        weight_decay=float(td3["weight_decay"]),
        device=device,
        seed=int(seed),
    )


def radii_for_mode(mode: str, radius_mm: float):
    if mode == "cube_only":
        return float(radius_mm), 0.0
    if mode == "goal_only":
        return 0.0, float(radius_mm)
    if mode == "both":
        return float(radius_mm), float(radius_mm)
    if mode == "none":
        return 0.0, 0.0
    raise ValueError(mode)


def set_fixed_angles(randomized, mode: str, cube_deg=None, goal_deg=None):
    randomized.fixed_cube_angle_rad = (
        None if cube_deg is None else float(np.deg2rad(cube_deg))
    )
    randomized.fixed_goal_angle_rad = (
        None if goal_deg is None else float(np.deg2rad(goal_deg))
    )


def build_fixed_eval_env(
    *,
    config,
    trajectory,
    sim_backend,
    mode,
    radius_mm,
    seed,
    apply_scaling=True,
):
    cube_radius_mm, goal_radius_mm = radii_for_mode(mode, radius_mm)
    # Passing a fixed angle at construction forces fixed_ring sampling.
    cube_angle = 0.0 if mode in {"cube_only", "both"} else None
    goal_angle = 0.0 if mode in {"goal_only", "both"} else None
    return build_env(
        config=config,
        trajectory=trajectory,
        sim_backend=sim_backend,
        seed=int(seed),
        randomization_mode=mode,
        cube_radius_mm=cube_radius_mm,
        goal_radius_mm=goal_radius_mm,
        fixed_cube_angle_deg=cube_angle,
        fixed_goal_angle_deg=goal_angle,
        apply_scaling=apply_scaling,
    )


def evaluate_reference_fixed_directions(
    *,
    config,
    trajectory,
    sim_backend,
    mode,
    radius_mm,
    num_directions,
    seed,
    max_steps,
    both_relation="same",
):
    env, randomized = build_fixed_eval_env(
        config=config,
        trajectory=trajectory,
        sim_backend=sim_backend,
        mode=mode,
        radius_mm=radius_mm,
        seed=seed,
        apply_scaling=True,
    )
    rows = []
    try:
        directions = [0.0] if mode == "none" else np.linspace(
            0.0, 360.0, int(num_directions), endpoint=False
        )
        for i, angle in enumerate(directions):
            cube_deg = None
            goal_deg = None
            if mode == "cube_only":
                cube_deg = float(angle)
            elif mode == "goal_only":
                goal_deg = float(angle)
            elif mode == "both":
                cube_deg = float(angle)
                goal_deg = (
                    float(angle)
                    if both_relation == "same"
                    else float((angle + 180.0) % 360.0)
                )
            set_fixed_angles(randomized, mode, cube_deg, goal_deg)
            ep_seed = int(seed) + i
            result = run_episode(
                env,
                policy=None,
                max_steps=int(max_steps),
                seed=ep_seed,
            )
            rows.append(
                {
                    "mode": mode,
                    "radius_mm": float(radius_mm),
                    "angle_deg": float(angle),
                    "cube_angle_deg": cube_deg,
                    "goal_angle_deg": goal_deg,
                    "seed": ep_seed,
                    **result,
                }
            )
    finally:
        env.close()
    return rows


def evaluate_paired_policy(
    *,
    config,
    trajectory,
    sim_backend,
    mode,
    radius_mm,
    num_directions,
    episodes_per_direction,
    seed,
    max_steps,
    policy,
    both_relation="same",
    both_grid=False,
):
    env, randomized = build_fixed_eval_env(
        config=config,
        trajectory=trajectory,
        sim_backend=sim_backend,
        mode=mode,
        radius_mm=radius_mm,
        seed=seed,
        apply_scaling=True,
    )

    rows = []
    try:
        directions = np.linspace(
            0.0, 360.0, int(num_directions), endpoint=False
        )

        angle_pairs = []
        if mode == "none":
            angle_pairs = [(0.0, None, None)]
        elif mode == "cube_only":
            angle_pairs = [(float(a), float(a), None) for a in directions]
        elif mode == "goal_only":
            angle_pairs = [(float(a), None, float(a)) for a in directions]
        elif mode == "both" and both_grid:
            for c in directions:
                for g in directions:
                    angle_pairs.append((float(c), float(c), float(g)))
        elif mode == "both":
            for a in directions:
                g = (
                    float(a)
                    if both_relation == "same"
                    else float((a + 180.0) % 360.0)
                )
                angle_pairs.append((float(a), float(a), g))
        else:
            raise ValueError(mode)

        case_index = 0
        for angle_label, cube_deg, goal_deg in angle_pairs:
            set_fixed_angles(randomized, mode, cube_deg, goal_deg)
            for repeat in range(int(episodes_per_direction)):
                ep_seed = int(seed) + case_index * 1000 + repeat

                reference = run_episode(
                    env,
                    policy=None,
                    max_steps=int(max_steps),
                    seed=ep_seed,
                )
                residual = run_episode(
                    env,
                    policy=policy.select_action,
                    max_steps=int(max_steps),
                    seed=ep_seed,
                )

                category = (
                    "both_success"
                    if reference["success"] and residual["success"]
                    else "rescued"
                    if (not reference["success"]) and residual["success"]
                    else "broken"
                    if reference["success"] and (not residual["success"])
                    else "both_failure"
                )

                rows.append(
                    {
                        "mode": mode,
                        "radius_mm": float(radius_mm),
                        "angle_deg": float(angle_label),
                        "cube_angle_deg": cube_deg,
                        "goal_angle_deg": goal_deg,
                        "repeat": int(repeat),
                        "seed": ep_seed,
                        "category": category,
                        "reference_success": bool(reference["success"]),
                        "reference_failure_cause": reference["failure_cause"],
                        "reference_return": reference["return"],
                        "reference_max_lift_m": reference["max_cube_lift_m"],
                        "reference_goal_xy_error_m": reference["goal_xy_error_m"],
                        "residual_success": bool(residual["success"]),
                        "residual_failure_cause": residual["failure_cause"],
                        "residual_return": residual["return"],
                        "residual_max_lift_m": residual["max_cube_lift_m"],
                        "residual_goal_xy_error_m": residual["goal_xy_error_m"],
                        "residual_max_l2": residual["max_residual_l2"],
                        "residual_max_abs": residual["max_abs_residual"],
                        "residual_saturation_rate": residual[
                            "residual_saturation_rate"
                        ],
                    }
                )
            case_index += 1
    finally:
        env.close()

    return rows


def summarize_paired(rows):
    n = len(rows)
    ref_success = sum(int(r["reference_success"]) for r in rows)
    res_success = sum(int(r["residual_success"]) for r in rows)
    rescued = sum(r["category"] == "rescued" for r in rows)
    broken = sum(r["category"] == "broken" for r in rows)
    ref_failures = Counter(
        r["reference_failure_cause"]
        for r in rows
        if not r["reference_success"]
    )
    res_failures = Counter(
        r["residual_failure_cause"]
        for r in rows
        if not r["residual_success"]
    )
    return {
        "episodes": n,
        "reference_success_count": ref_success,
        "reference_success_rate": ref_success / n if n else 0.0,
        "residual_success_count": res_success,
        "residual_success_rate": res_success / n if n else 0.0,
        "improvement_percentage_points": (
            100.0 * (res_success - ref_success) / n if n else 0.0
        ),
        "rescued": rescued,
        "broken": broken,
        "net_gain": rescued - broken,
        "reference_failure_counts": dict(ref_failures),
        "residual_failure_counts": dict(res_failures),
        "mean_residual_saturation_rate": (
            float(np.mean([r["residual_saturation_rate"] for r in rows]))
            if rows
            else 0.0
        ),
        "mean_residual_max_l2": (
            float(np.mean([r["residual_max_l2"] for r in rows]))
            if rows
            else 0.0
        ),
    }
