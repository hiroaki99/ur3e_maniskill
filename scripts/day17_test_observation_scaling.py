#!/usr/bin/env python3
"""Day17 smoke test for cube_to_grasp observation scaling.

Checks:
1. Observation remains 44-D.
2. Only [30:33] changes.
3. [30:33] is exactly raw * scale on reset and step.
4. 20 mm fixed-angle perturbations are actually applied before the final
   ResidualPickLiftEnv observation is generated.
5. Reward remains finite; the scaling wrapper itself never recomputes reward.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rrl import CubePositionPerturbationWrapper, ResidualPickLiftEnv
from rrl.observation_scaling import CubeToGraspObservationScaleWrapper


CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"
DEFAULT_TRAJECTORY = REPO_ROOT / "trajectories" / "day6_pick_lift_reference_v2.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=["physx_cpu", "physx_cuda"],
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--radius-mm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1700)
    return parser.parse_args()


def build_env(config, args):
    import envs.ur3e_pick_lift  # noqa: F401

    day9 = config.get("day9", {})
    day12 = config["day12"]
    day17 = config.get("day17", {})

    scale = float(
        args.scale
        if args.scale is not None
        else day17.get("cube_to_grasp_scale", 10.0)
    )
    radius_mm = float(
        args.radius_mm
        if args.radius_mm is not None
        else day17.get("train_radius_mm", day12["selected_radius_mm"])
    )

    base_env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=args.sim_backend,
        max_episode_steps=int(day9.get("base_env_max_episode_steps", 1000)),
    )

    perturb_env = CubePositionPerturbationWrapper(
        base_env,
        radius_min_m=radius_mm / 1000.0,
        radius_max_m=radius_mm / 1000.0,
        direction_mode="random_angle",
        fixed_angle_rad=None,
        seed=args.seed,
    )

    residual_env = ResidualPickLiftEnv(
        env=perturb_env,
        trajectory_path=args.trajectory,
        config_path=CONFIG_PATH,
    )

    scaled_env = CubeToGraspObservationScaleWrapper(
        residual_env,
        scale=scale,
    )

    return scaled_env, residual_env, perturb_env, scale, radius_mm


def check_scaled(raw, scaled, scale, label):
    raw = np.asarray(raw, dtype=np.float32)
    scaled = np.asarray(scaled, dtype=np.float32)

    assert raw.shape == (44,), f"{label}: raw shape={raw.shape}"
    assert scaled.shape == (44,), f"{label}: scaled shape={scaled.shape}"

    mask = np.ones(44, dtype=bool)
    mask[30:33] = False

    max_other_diff = float(np.max(np.abs(raw[mask] - scaled[mask])))
    scale_diff = float(np.max(np.abs(scaled[30:33] - raw[30:33] * scale)))

    assert max_other_diff <= 1e-7, (
        f"{label}: values outside [30:33] changed: {max_other_diff}"
    )
    assert scale_diff <= 1e-6, (
        f"{label}: [30:33] is not raw*scale: diff={scale_diff}"
    )

    expected_raw = raw[15:18] - raw[12:15]
    relation_diff = float(np.max(np.abs(raw[30:33] - expected_raw)))
    assert relation_diff <= 1e-6, (
        f"{label}: raw cube_to_grasp contract broken: diff={relation_diff}"
    )

    return max_other_diff, scale_diff, relation_diff


def main():
    args = parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env, residual_env, perturb_env, scale, radius_mm = build_env(config, args)

    try:
        print("=" * 72)
        print("Day17 Observation Scaling Smoke Test")
        print("=" * 72)
        print("scale:", scale)
        print("radius_mm:", radius_mm)

        rows = []
        for angle_deg in [0.0, 90.0, 180.0, 270.0]:
            perturb_env.fixed_angle_rad = float(np.deg2rad(angle_deg))

            scaled_obs, info = env.reset(seed=args.seed)
            raw_obs = residual_env._get_observation(
                residual_env.last_metrics
            ).copy()

            check_scaled(raw_obs, scaled_obs, scale, f"reset {angle_deg}")

            offset = np.asarray(info["cube_offset_m"], dtype=np.float64)
            expected = np.asarray(
                [
                    radius_mm / 1000.0 * np.cos(np.deg2rad(angle_deg)),
                    radius_mm / 1000.0 * np.sin(np.deg2rad(angle_deg)),
                    0.0,
                ],
                dtype=np.float64,
            )
            offset_diff = float(np.linalg.norm(offset - expected))
            assert offset_diff <= 5e-5, (
                f"angle={angle_deg}: perturbation mismatch: "
                f"actual={offset}, expected={expected}"
            )

            rows.append((angle_deg, offset.copy(), raw_obs[30:33].copy(), scaled_obs[30:33].copy()))

        # One real step: wrapper must also scale the post-step observation.
        perturb_env.fixed_angle_rad = 0.0
        scaled_obs, _ = env.reset(seed=args.seed)
        zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        scaled_next, reward, terminated, truncated, info = env.step(zero_action)
        raw_next = residual_env._get_observation(residual_env.last_metrics).copy()
        check_scaled(raw_next, scaled_next, scale, "step")

        assert np.isfinite(float(reward)), f"reward is not finite: {reward}"
        assert "reward_terms" in info, "reward_terms missing"

        for angle_deg, offset, raw_ctg, scaled_ctg in rows:
            print(
                f"angle={angle_deg:6.1f} deg  "
                f"offset_mm={offset[:2] * 1000.0}  "
                f"raw_ctg={raw_ctg}  scaled_ctg={scaled_ctg}"
            )

        print("reward after zero-action step:", float(reward))
        print("terminated/truncated:", bool(terminated), bool(truncated))
        print("PASSED: True")
        return 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
