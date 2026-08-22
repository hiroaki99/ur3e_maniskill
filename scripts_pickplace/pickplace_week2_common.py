#!/usr/bin/env python3
"""Shared builders for Pick-and-Place Week2 scripts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import gymnasium as gym
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"
DEFAULT_TRAJECTORY = REPO_ROOT / "trajectories" / "pick_place_reference_v1.json"


def load_config(path: Path = CONFIG_PATH):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_env(
    *,
    config,
    trajectory: Path,
    sim_backend: str,
    seed: int,
    randomization_mode: str = "none",
    cube_radius_mm: float = 0.0,
    goal_radius_mm: float = 0.0,
    fixed_cube_angle_deg: float | None = None,
    fixed_goal_angle_deg: float | None = None,
    apply_scaling: bool = False,
):
    import numpy as np
    import envs.ur3e_pick_place  # noqa: F401
    from rrl.pick_place_randomization import PickPlaceRandomizationWrapper
    from rrl.residual_pick_place_env import ResidualPickPlaceEnv
    from rrl.pick_place_observation_scaling import PickPlaceObservationScaleWrapper

    base = gym.make(
        "UR3ePickPlace-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=sim_backend,
        max_episode_steps=int(config["pick_place"]["max_episode_steps"]),
    )

    randomized = PickPlaceRandomizationWrapper(
        base,
        mode=randomization_mode,
        cube_radius_min_m=float(cube_radius_mm) / 1000.0,
        cube_radius_max_m=float(cube_radius_mm) / 1000.0,
        goal_radius_min_m=float(goal_radius_mm) / 1000.0,
        goal_radius_max_m=float(goal_radius_mm) / 1000.0,
        sampling="fixed_ring" if (
            fixed_cube_angle_deg is not None or fixed_goal_angle_deg is not None
        ) else config.get("randomization", {}).get("sampling", "area_uniform_disk"),
        fixed_cube_angle_rad=(
            None if fixed_cube_angle_deg is None else float(np.deg2rad(fixed_cube_angle_deg))
        ),
        fixed_goal_angle_rad=(
            None if fixed_goal_angle_deg is None else float(np.deg2rad(fixed_goal_angle_deg))
        ),
        seed=int(seed),
    )

    residual = ResidualPickPlaceEnv(
        env=randomized,
        trajectory_path=trajectory,
        config_path=CONFIG_PATH,
    )

    if not apply_scaling:
        return residual, randomized

    scale_cfg = config["observation_scaling"]
    scaled = PickPlaceObservationScaleWrapper(
        residual,
        cube_to_grasp_scale=float(scale_cfg["cube_to_grasp"]),
        cube_to_goal_scale=float(scale_cfg["cube_to_goal"]),
        grasp_to_goal_scale=float(scale_cfg["grasp_to_goal"]),
    )
    return scaled, randomized
