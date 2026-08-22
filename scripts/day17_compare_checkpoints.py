#!/usr/bin/env python3
"""Compare Day13 baseline and Day17 candidates on identical 32-direction physics.

Crucial preprocessing rule:
- Day13 checkpoint is evaluated with cube_to_grasp scale = 1.0.
- Day17 checkpoint(s) are evaluated with the Day17 training scale (default 10.0).

Applying scale=10 to the old Day13 checkpoint would be an invalid comparison
because its actor was trained on the unscaled observation distribution.
"""

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

from rrl import (
    CubePositionPerturbationWrapper,
    ResidualPickLiftEnv,
    TD3,
    evaluate_residual_policy,
)
from rrl.observation_scaling import CubeToGraspObservationScaleWrapper


CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"
DEFAULT_TRAJECTORY = REPO_ROOT / "trajectories" / "day6_pick_lift_reference_v2.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--day13-checkpoint",
        type=Path,
        default=Path("runs/day13_td3_seed1300/checkpoints/best.pt"),
    )
    parser.add_argument("--day17-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=["physx_cpu", "physx_cuda"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-directions", type=int, default=None)
    parser.add_argument("--episodes-per-direction", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/day17_checkpoint_comparison"),
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_env(*, config, trajectory, sim_backend, radius_mm, seed, scale):
    import envs.ur3e_pick_lift  # noqa: F401

    day9 = config.get("day9", {})
    base_env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=sim_backend,
        max_episode_steps=int(day9.get("base_env_max_episode_steps", 1000)),
    )

    radius_m = float(radius_mm) / 1000.0
    perturb_env = CubePositionPerturbationWrapper(
        base_env,
        radius_min_m=radius_m,
        radius_max_m=radius_m,
        direction_mode="random_angle",
        fixed_angle_rad=None,
        seed=seed,
    )
    residual_env = ResidualPickLiftEnv(
        env=perturb_env,
        trajectory_path=trajectory,
        config_path=CONFIG_PATH,
    )
    if float(scale) == 1.0:
        return residual_env, perturb_env
    return (
        CubeToGraspObservationScaleWrapper(residual_env, scale=float(scale)),
        perturb_env,
    )


def make_policy(config, env, device, seed):
    day11 = config["day11"]
    return TD3(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        max_action=float(env.action_space.high[0]),
        actor_hidden_dim=int(day11["actor_hidden_dim"]),
        critic_hidden_dim=int(day11["critic_hidden_dim"]),
        discount=float(day11["discount"]),
        tau=float(day11["tau"]),
        policy_noise=float(day11["policy_noise"]),
        noise_clip=float(day11["noise_clip"]),
        policy_freq=int(day11["policy_freq"]),
        actor_lr=float(day11["actor_lr"]),
        critic_lr=float(day11["critic_lr"]),
        weight_decay=float(day11["weight_decay"]),
        device=device,
        seed=seed,
    )


def paired_counts(reference_rows, residual_rows):
    ref_map = {
        (float(r["angle_deg"]), int(r["repeat"]), int(r["seed"])): bool(r["success"])
        for r in reference_rows
    }
    rescued = broken = kept_success = kept_failure = 0
    for r in residual_rows:
        key = (float(r["angle_deg"]), int(r["repeat"]), int(r["seed"]))
        ref_success = ref_map[key]
        res_success = bool(r["success"])
        if not ref_success and res_success:
            rescued += 1
        elif ref_success and not res_success:
            broken += 1
        elif ref_success and res_success:
            kept_success += 1
        else:
            kept_failure += 1
    return {
        "rescued": rescued,
        "broken": broken,
        "kept_success": kept_success,
        "kept_failure": kept_failure,
        "net_gain": rescued - broken,
    }


def rank_key(row):
    return (
        int(row["success_count"]),
        -int(row["broken"]),
        -float(row["residual_saturation_rate"]),
        -float(row["mean_residual_l2"]),
    )


def evaluate_condition(
    *, env, perturb, policy, angles_deg, episodes_per_direction, seed,
    max_steps, lift_threshold_m, saturation_threshold
):
    return evaluate_residual_policy(
        env=env,
        perturbation_wrapper=perturb,
        policy=policy,
        angles_deg=angles_deg,
        episodes_per_direction=episodes_per_direction,
        seed=seed,
        max_steps=max_steps,
        lift_threshold_m=lift_threshold_m,
        saturation_threshold=saturation_threshold,
    )


def main():
    args = parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    day12 = config["day12"]
    day13 = config["day13"]
    day17 = config["day17"]

    radius_mm = float(day17.get("train_radius_mm", day12["selected_radius_mm"]))
    eval_seed = int(day17.get("eval_seed", 1200))
    train_seed = int(day17.get("train_seed", 1300))
    day17_scale = float(day17["cube_to_grasp_scale"])
    num_directions = int(
        args.num_directions
        if args.num_directions is not None
        else day17["final_eval_num_directions"]
    )
    episodes_per_direction = int(
        args.episodes_per_direction
        if args.episodes_per_direction is not None
        else day17["final_eval_episodes_per_direction"]
    )
    max_steps = int(day17["max_steps_per_episode"])
    lift_threshold_m = float(day17["lift_threshold_m"])
    saturation_threshold = float(day17["residual_saturation_threshold"])
    gate_min_success_rate = float(day17.get("gate_min_success_rate", 0.60))
    device = resolve_device(args.device)

    angles_deg = np.linspace(0.0, 360.0, num_directions, endpoint=False).tolist()

    # Day13 environment: raw scale=1.0
    env13, perturb13 = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
        scale=1.0,
    )

    # Day17 environment: scale=10 (or configured value)
    env17, perturb17 = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
        scale=day17_scale,
    )

    try:
        # Reference is evaluated in both envs as a consistency check. The outer
        # observation scaling must not change reference physics/reward.
        ref13_summary, ref13_rows = evaluate_condition(
            env=env13,
            perturb=perturb13,
            policy=None,
            angles_deg=angles_deg,
            episodes_per_direction=episodes_per_direction,
            seed=eval_seed,
            max_steps=max_steps,
            lift_threshold_m=lift_threshold_m,
            saturation_threshold=saturation_threshold,
        )
        ref17_summary, ref17_rows = evaluate_condition(
            env=env17,
            perturb=perturb17,
            policy=None,
            angles_deg=angles_deg,
            episodes_per_direction=episodes_per_direction,
            seed=eval_seed,
            max_steps=max_steps,
            lift_threshold_m=lift_threshold_m,
            saturation_threshold=saturation_threshold,
        )

        reference_consistent = (
            int(ref13_summary["success_count"]) == int(ref17_summary["success_count"])
        )

        summary_rows = []
        episode_rows = []

        for row in ref13_rows:
            x = dict(row)
            x["condition"] = "reference"
            x["checkpoint"] = ""
            x["cube_to_grasp_scale"] = 1.0
            episode_rows.append(x)

        # Day13 checkpoint: MUST use scale 1.0.
        policy13 = make_policy(config, env13, device, train_seed)
        policy13.load(args.day13_checkpoint)
        d13_summary, d13_rows = evaluate_condition(
            env=env13,
            perturb=perturb13,
            policy=policy13,
            angles_deg=angles_deg,
            episodes_per_direction=episodes_per_direction,
            seed=eval_seed,
            max_steps=max_steps,
            lift_threshold_m=lift_threshold_m,
            saturation_threshold=saturation_threshold,
        )
        d13_pairs = paired_counts(ref13_rows, d13_rows)
        d13_record = {
            "condition": "day13_baseline",
            "checkpoint": str(args.day13_checkpoint),
            "cube_to_grasp_scale": 1.0,
            **d13_summary,
            **d13_pairs,
        }
        d13_record["gate3_eligible"] = bool(d13_record["success_rate"] >= gate_min_success_rate)
        summary_rows.append(d13_record)
        for row in d13_rows:
            x = dict(row)
            x["condition"] = "day13_baseline"
            x["checkpoint"] = str(args.day13_checkpoint)
            x["cube_to_grasp_scale"] = 1.0
            episode_rows.append(x)

        # Day17 candidates: MUST use Day17 scale.
        for checkpoint in args.day17_checkpoints:
            policy17 = make_policy(config, env17, device, train_seed)
            policy17.load(checkpoint)
            d17_summary, d17_rows = evaluate_condition(
                env=env17,
                perturb=perturb17,
                policy=policy17,
                angles_deg=angles_deg,
                episodes_per_direction=episodes_per_direction,
                seed=eval_seed,
                max_steps=max_steps,
                lift_threshold_m=lift_threshold_m,
                saturation_threshold=saturation_threshold,
            )
            d17_pairs = paired_counts(ref17_rows, d17_rows)
            record = {
                "condition": "day17_scaled",
                "checkpoint": str(checkpoint),
                "cube_to_grasp_scale": day17_scale,
                **d17_summary,
                **d17_pairs,
            }
            record["gate3_eligible"] = bool(record["success_rate"] >= gate_min_success_rate)
            summary_rows.append(record)

            for row in d17_rows:
                x = dict(row)
                x["condition"] = "day17_scaled"
                x["checkpoint"] = str(checkpoint)
                x["cube_to_grasp_scale"] = day17_scale
                episode_rows.append(x)

        day17_ranked = [r for r in summary_rows if r["condition"] == "day17_scaled"]
        day17_ranked.sort(key=rank_key, reverse=True)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "summary.csv", summary_rows)
        write_csv(args.output_dir / "episodes.csv", episode_rows)

        result = {
            "radius_mm": radius_mm,
            "eval_seed": eval_seed,
            "num_directions": num_directions,
            "episodes_per_direction": episodes_per_direction,
            "total_episodes_per_condition": num_directions * episodes_per_direction,
            "reference_scale1": ref13_summary,
            "reference_scale_day17": ref17_summary,
            "reference_consistent": reference_consistent,
            "day13": d13_record,
            "day17_ranked": day17_ranked,
            "gate_min_success_rate": gate_min_success_rate,
        }
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("=" * 72)
        print("Day17 Checkpoint Comparison")
        print("=" * 72)
        print("Reference consistency (scale1 vs scaled wrapper):", reference_consistent)
        print(
            "Reference:",
            f"{ref13_summary['success_count']}/{ref13_summary['episodes']} = "
            f"{ref13_summary['success_rate']:.3f}",
        )
        print(
            "Day13:",
            f"{d13_record['success_count']}/{d13_record['episodes']} = "
            f"{d13_record['success_rate']:.3f}",
            "broken=", d13_record["broken"],
        )
        for rank, row in enumerate(day17_ranked, start=1):
            print(
                f"Day17 rank {rank}: {row['checkpoint']}  "
                f"success={row['success_count']}/{row['episodes']} "
                f"({row['success_rate']:.3f})  rescued={row['rescued']} "
                f"broken={row['broken']} sat={row['residual_saturation_rate']:.3f} "
                f"L2={row['mean_residual_l2']:.3f} "
                f"gate3_eligible={row['gate3_eligible']}"
            )
        print("summary:", args.output_dir / "summary.json")
        return 0

    finally:
        env13.close()
        env17.close()


if __name__ == "__main__":
    raise SystemExit(main())
