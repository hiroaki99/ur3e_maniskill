#!/usr/bin/env python3
"""Day17 angle x joint residual diagnostic for a scaled-observation checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rrl import CubePositionPerturbationWrapper, ResidualPickLiftEnv, TD3
from rrl.observation_scaling import CubeToGraspObservationScaleWrapper

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"
DEFAULT_TRAJECTORY = REPO_ROOT / "trajectories" / "day6_pick_lift_reference_v2.json"
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-directions", type=int, default=None)
    p.add_argument("--episodes-per-direction", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("reports/day17_angle_joint"))
    return p.parse_args()


def resolve_device(x):
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if x == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return x


def make_env(config, args, scale, radius_mm, seed):
    import envs.ur3e_pick_lift  # noqa: F401
    day9 = config.get("day9", {})
    base = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=args.sim_backend,
        max_episode_steps=int(day9.get("base_env_max_episode_steps", 1000)),
    )
    perturb = CubePositionPerturbationWrapper(
        base,
        radius_min_m=radius_mm / 1000.0,
        radius_max_m=radius_mm / 1000.0,
        direction_mode="random_angle",
        fixed_angle_rad=None,
        seed=seed,
    )
    residual = ResidualPickLiftEnv(
        env=perturb,
        trajectory_path=args.trajectory,
        config_path=CONFIG_PATH,
    )
    return CubeToGraspObservationScaleWrapper(residual, scale=scale), perturb


def make_policy(config, env, device, seed):
    d = config["day11"]
    return TD3(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        max_action=float(env.action_space.high[0]),
        actor_hidden_dim=int(d["actor_hidden_dim"]),
        critic_hidden_dim=int(d["critic_hidden_dim"]),
        discount=float(d["discount"]),
        tau=float(d["tau"]),
        policy_noise=float(d["policy_noise"]),
        noise_clip=float(d["noise_clip"]),
        policy_freq=int(d["policy_freq"]),
        actor_lr=float(d["actor_lr"]),
        critic_lr=float(d["critic_lr"]),
        weight_decay=float(d["weight_decay"]),
        device=device,
        seed=seed,
    )


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    args = parse_args()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    d17 = config["day17"]
    scale = float(d17["cube_to_grasp_scale"])
    radius_mm = float(d17["train_radius_mm"])
    seed = int(d17["eval_seed"])
    max_steps = int(d17["max_steps_per_episode"])
    sat_thr = float(d17["residual_saturation_threshold"])
    n_dir = int(args.num_directions or d17["final_eval_num_directions"])
    device = resolve_device(args.device)

    env, perturb = make_env(config, args, scale, radius_mm, seed)
    try:
        policy = make_policy(config, env, device, int(d17["train_seed"]))
        policy.load(args.checkpoint)
        angles = np.linspace(0.0, 360.0, n_dir, endpoint=False)
        angle_rows = []

        for i, angle in enumerate(angles):
            sums = np.zeros(6, dtype=np.float64)
            abs_sums = np.zeros(6, dtype=np.float64)
            sat_counts = np.zeros(6, dtype=np.int64)
            steps = 0
            successes = 0

            perturb.fixed_angle_rad = float(np.deg2rad(angle))
            for repeat in range(args.episodes_per_direction):
                episode_seed = seed + i * 1000 + repeat
                obs, _ = env.reset(seed=episode_seed)
                obs = np.asarray(obs, dtype=np.float32)
                final_info = {}
                for _ in range(max_steps):
                    action = np.clip(policy.select_action(obs), -1.0, 1.0)
                    sums += action
                    abs_sums += np.abs(action)
                    sat_counts += (np.abs(action) >= sat_thr).astype(np.int64)
                    steps += 1
                    obs, _, terminated, truncated, info = env.step(action.astype(np.float32))
                    obs = np.asarray(obs, dtype=np.float32)
                    final_info = info
                    if terminated or truncated:
                        break
                successes += int(bool(final_info.get("success", False)))

            row = {
                "angle_deg": float(angle),
                "success_count": successes,
                "episodes": args.episodes_per_direction,
                "steps": steps,
            }
            for j, name in enumerate(JOINT_NAMES):
                row[f"{name}_mean"] = float(sums[j] / max(steps, 1))
                row[f"{name}_mean_abs"] = float(abs_sums[j] / max(steps, 1))
                row[f"{name}_saturation"] = float(sat_counts[j] / max(steps, 1))
            angle_rows.append(row)

        joint_rows = []
        for name in JOINT_NAMES:
            vals = np.asarray([r[f"{name}_mean"] for r in angle_rows], dtype=np.float64)
            joint_rows.append(
                {
                    "joint": name,
                    "mean_over_angles": float(vals.mean()),
                    "std_angle": float(vals.std()),
                    "min_angle_mean": float(vals.min()),
                    "max_angle_mean": float(vals.max()),
                    "range_angle_mean": float(vals.max() - vals.min()),
                }
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "angle_joint.csv", angle_rows)
        write_csv(args.output_dir / "joint_summary.csv", joint_rows)
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(args.checkpoint),
                    "cube_to_grasp_scale": scale,
                    "radius_mm": radius_mm,
                    "num_directions": n_dir,
                    "episodes_per_direction": args.episodes_per_direction,
                    "joints": joint_rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("=" * 72)
        print("Day17 Angle x Joint Diagnostic")
        print("=" * 72)
        for row in joint_rows:
            print(
                f"{row['joint']:14s} mean={row['mean_over_angles']:+.3f} "
                f"std_angle={row['std_angle']:.3f} "
                f"min={row['min_angle_mean']:+.3f} max={row['max_angle_mean']:+.3f}"
            )
        print("summary:", args.output_dir / "summary.json")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
