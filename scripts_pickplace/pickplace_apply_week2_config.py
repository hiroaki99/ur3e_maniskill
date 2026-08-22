#!/usr/bin/env python3
"""Merge Week2 Residual Pick-and-Place settings into the Week1 config.

The script preserves all existing Week1 values, creates a .week1_backup copy once,
and adds/replaces only Week2-specific sections.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ur3e_pick_place.yaml"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    path = args.config
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")

    backup = path.with_suffix(path.suffix + ".week1_backup")
    if not backup.exists():
        shutil.copy2(path, backup)

    config["residual_pick_place"] = {
        "alpha": 0.20,
        "transport_drop_threshold_m": 0.020,
        "minimum_lift_after_lift_m": 0.040,
    }

    config["randomization"] = {
        "sampling": "area_uniform_disk",
        "cube_radius_min_m": 0.0,
        "cube_radius_max_m": 0.010,
        "goal_radius_min_m": 0.0,
        "goal_radius_max_m": 0.010,
    }

    config["observation_scaling"] = {
        "cube_to_grasp": 10.0,
        "cube_to_goal": 10.0,
        "grasp_to_goal": 10.0,
    }

    config["reward_pick_place"] = {
        "approach_progress_scale_m": 0.010,
        "approach_progress_weight": 0.20,
        "bilateral_contact_bonus": 0.05,
        "hold_contact_bonus": 0.02,
        "lift_progress_weight": 50.0,
        "max_lift_delta_per_step_m": 0.010,
        "transport_progress_scale_m": 0.010,
        "transport_progress_weight": 0.30,
        "place_progress_scale_m": 0.010,
        "place_progress_weight": 0.30,
        "release_bonus": 0.50,
        "stable_step_bonus": 0.02,
        "success_bonus": 10.0,
        "failure_penalty": 5.0,
        "residual_penalty_weight": 0.05,
        "enable_excessive_force_penalty": False,
        "max_safe_gripper_force_n": 100.0,
        "excessive_force_penalty_weight": 0.20,
        "enable_unsafe_collision_penalty": False,
        "unsafe_collision_penalty": 5.0,
    }

    config["week2"] = {
        "zero_equivalence_episodes": 32,
        "zero_equivalence_min_success_rate": 0.95,
        "randomization_test_radius_mm": 10.0,
        "observation_audit_episodes": 16,
        "observation_audit_radius_mm": 10.0,
        "training_interface_smoke_directions": 4,
    }

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print("updated:", path)
    print("backup :", backup)
    print("residual alpha:", config["residual_pick_place"]["alpha"])
    print("observation scaling candidates:", config["observation_scaling"])
    print("collision penalty enabled:", config["reward_pick_place"]["enable_unsafe_collision_penalty"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
