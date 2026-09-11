#!/usr/bin/env python3
"""Day18: held-out final checkpoint selection for UR3e Residual Pick-and-Lift.

Purpose
-------
1. Compare Reference, Day13 25k, Day13 best, and Day17 top candidates
   under exactly the same controlled 20 mm perturbation benchmark.
2. Evaluate Day13 checkpoints with cube_to_grasp scale=1.0 and Day17
   checkpoints with the scale used during Day17 training (normally 10.0).
3. Use a held-out evaluation seed and a finer directional grid than Day17
   candidate selection to reduce checkpoint-selection bias.
4. Rank checkpoints by:
      success_count (higher)
      -> broken (lower)
      -> residual saturation (lower)
      -> residual L2 (lower)
5. If Gate3 passes, copy the selected checkpoint to a fixed final path and
   write metadata including the required observation scale and SHA256.
6. Perform a lightweight safety-info audit so Day19 can decide whether
   collision instrumentation is already observable or still needs code changes.

This script does NOT train or modify the policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rrl import (  # noqa: E402
    CubePositionPerturbationWrapper,
    ResidualPickLiftEnv,
    TD3,
    evaluate_residual_policy,
)
from rrl.observation_scaling import (  # noqa: E402
    CubeToGraspObservationScaleWrapper,
)


CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"
DEFAULT_TRAJECTORY = (
    REPO_ROOT / "trajectories" / "day6_pick_lift_reference_v2.json"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--day13-25k",
        type=Path,
        default=Path("runs/day13_td3_seed1300/checkpoints/step_00025000.pt"),
    )
    parser.add_argument(
        "--day13-best",
        type=Path,
        default=Path("runs/day13_td3_seed1300/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--day17-rank1",
        type=Path,
        default=Path(
            "runs/day17_td3_obs_scale10_seed1300/checkpoints/candidate_rank1.pt"
        ),
    )
    parser.add_argument(
        "--day17-rank2",
        type=Path,
        default=Path(
            "runs/day17_td3_obs_scale10_seed1300/checkpoints/candidate_rank2.pt"
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
    )
    parser.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=["physx_cpu", "physx_cuda"],
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--num-directions", type=int, default=None)
    parser.add_argument("--episodes-per-direction", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/day18_final_selection"),
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("runs/day18_final"),
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Evaluate only; do not copy a final checkpoint.",
    )

    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (
        z
        * np.sqrt(
            (p * (1.0 - p) / total) + (z2 / (4.0 * total * total))
        )
        / denom
    )
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def make_env(
    *,
    config,
    trajectory: Path,
    sim_backend: str,
    radius_mm: float,
    seed: int,
    scale: float,
):
    import envs.ur3e_pick_lift  # noqa: F401

    day9 = config.get("day9", {})

    base_env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config["robot"]["uid"],
        num_envs=1,
        obs_mode="state",
        control_mode=config["project"]["control_mode"],
        sim_backend=sim_backend,
        max_episode_steps=int(
            day9.get("base_env_max_episode_steps", 1000)
        ),
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

    if np.isclose(float(scale), 1.0):
        return residual_env, perturb_env

    scaled_env = CubeToGraspObservationScaleWrapper(
        residual_env,
        scale=float(scale),
    )

    return scaled_env, perturb_env


def make_policy(config, env, device: str, seed: int):
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
        (float(r["angle_deg"]), int(r["repeat"]), int(r["seed"])): bool(
            r["success"]
        )
        for r in reference_rows
    }

    rescued = 0
    broken = 0
    kept_success = 0
    kept_failure = 0

    for row in residual_rows:
        key = (
            float(row["angle_deg"]),
            int(row["repeat"]),
            int(row["seed"]),
        )
        if key not in ref_map:
            raise RuntimeError(f"Missing paired reference episode: {key}")

        ref_success = ref_map[key]
        res_success = bool(row["success"])

        if (not ref_success) and res_success:
            rescued += 1
        elif ref_success and (not res_success):
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


def paired_policy_delta(rows_a, rows_b):
    """Compare two policies on exactly paired episodes.

    Returns how often B improves/degrades relative to A.
    """
    a_map = {
        (float(r["angle_deg"]), int(r["repeat"]), int(r["seed"])): bool(
            r["success"]
        )
        for r in rows_a
    }

    improved = 0
    degraded = 0
    both_success = 0
    both_failure = 0

    for row in rows_b:
        key = (
            float(row["angle_deg"]),
            int(row["repeat"]),
            int(row["seed"]),
        )
        if key not in a_map:
            raise RuntimeError(f"Missing paired policy episode: {key}")

        a = a_map[key]
        b = bool(row["success"])

        if (not a) and b:
            improved += 1
        elif a and (not b):
            degraded += 1
        elif a and b:
            both_success += 1
        else:
            both_failure += 1

    return {
        "improved": improved,
        "degraded": degraded,
        "both_success": both_success,
        "both_failure": both_failure,
        "net_gain": improved - degraded,
    }


def rank_key(row):
    return (
        int(row["success_count"]),
        -int(row["broken"]),
        -float(row["residual_saturation_rate"]),
        -float(row["mean_residual_l2"]),
    )


def evaluate_condition(
    *,
    env,
    perturb,
    policy,
    angles_deg,
    episodes_per_direction: int,
    seed: int,
    max_steps: int,
    lift_threshold_m: float,
    saturation_threshold: float,
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


def add_common_metrics(record: dict):
    low, high = wilson_interval(
        int(record["success_count"]),
        int(record["episodes"]),
    )
    record["success_ci95_low"] = low
    record["success_ci95_high"] = high
    return record


def flatten_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.append(name)
            keys.extend(flatten_keys(child, name))
    return keys


def safety_audit(
    *,
    env,
    perturb,
    policy,
    seed: int,
    max_steps: int,
):
    """Inspect runtime info keys without claiming collision detection is correct."""

    angles = [0.0, 90.0, 180.0, 270.0]
    all_keys: set[str] = set()
    safety_like_keys: set[str] = set()
    max_left_force = 0.0
    max_right_force = 0.0

    keywords = (
        "collision",
        "unsafe",
        "contact",
        "force",
        "joint_limit",
        "limit_violation",
    )

    for index, angle in enumerate(angles):
        perturb.fixed_angle_rad = float(np.deg2rad(angle))
        obs, _ = env.reset(seed=seed + 100000 + index)
        obs = np.asarray(obs, dtype=np.float32)

        for _ in range(max_steps):
            if policy is None:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            else:
                action = np.asarray(policy.select_action(obs), dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)

            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)

            keys = flatten_keys(info)
            all_keys.update(keys)
            for key in keys:
                lower = key.lower()
                if any(token in lower for token in keywords):
                    safety_like_keys.add(key)

            max_left_force = max(
                max_left_force,
                float(info.get("left_contact_force", 0.0)),
            )
            max_right_force = max(
                max_right_force,
                float(info.get("right_contact_force", 0.0)),
            )

            if terminated or truncated:
                break

    collision_candidates = sorted(
        key
        for key in safety_like_keys
        if ("collision" in key.lower() or "unsafe" in key.lower())
    )

    return {
        "angles_deg": angles,
        "max_left_contact_force_n": float(max_left_force),
        "max_right_contact_force_n": float(max_right_force),
        "safety_like_keys": sorted(safety_like_keys),
        "collision_related_key_candidates": collision_candidates,
        "collision_instrumentation_key_found": bool(collision_candidates),
        "note": (
            "A collision-related key only means a candidate signal exists; "
            "it does not prove that collision detection is physically correct."
        ),
    }


def main():
    args = parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "day18" not in config:
        raise RuntimeError(
            "configs/ur3e_pick_lift.yaml に day18 セクションを追加してください。"
        )

    day12 = config["day12"]
    day17 = config["day17"]
    day18 = config["day18"]

    radius_mm = float(
        day18.get(
            "radius_mm",
            day17.get("train_radius_mm", day12["selected_radius_mm"]),
        )
    )
    eval_seed = int(
        args.eval_seed
        if args.eval_seed is not None
        else day18.get("eval_seed", 1800)
    )
    train_seed = int(day17.get("train_seed", 1300))
    day17_scale = float(day17["cube_to_grasp_scale"])

    num_directions = int(
        args.num_directions
        if args.num_directions is not None
        else day18.get("num_directions", 64)
    )
    episodes_per_direction = int(
        args.episodes_per_direction
        if args.episodes_per_direction is not None
        else day18.get("episodes_per_direction", 1)
    )

    max_steps = int(day18.get("max_steps_per_episode", 600))
    lift_threshold_m = float(day18.get("lift_threshold_m", 0.05))
    saturation_threshold = float(
        day18.get("residual_saturation_threshold", 0.95)
    )
    gate_min_success_rate = float(
        day18.get("gate_min_success_rate", 0.60)
    )

    device = resolve_device(args.device)

    checkpoints = [
        {
            "label": "day13_25k",
            "checkpoint": args.day13_25k,
            "scale": 1.0,
            "training_step": 25000,
        },
        {
            "label": "day13_best",
            "checkpoint": args.day13_best,
            "scale": 1.0,
            "training_step": None,
        },
        {
            "label": "day17_rank1",
            "checkpoint": args.day17_rank1,
            "scale": day17_scale,
            "training_step": 25000,
        },
        {
            "label": "day17_rank2",
            "checkpoint": args.day17_rank2,
            "scale": day17_scale,
            "training_step": 35000,
        },
    ]

    for item in checkpoints:
        if not item["checkpoint"].exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {item['label']} -> {item['checkpoint']}"
            )

    if num_directions < 30 and episodes_per_direction == 1:
        print(
            "WARNING: total unique directions < 30. "
            "Day18 formal evaluation should use >=30 episodes."
        )

    angles_deg = np.linspace(
        0.0,
        360.0,
        num_directions,
        endpoint=False,
    ).tolist()

    # Two observation contracts are needed because old and new actors were
    # trained under different input scaling.
    env_raw, perturb_raw = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
        scale=1.0,
    )

    env_scaled, perturb_scaled = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
        scale=day17_scale,
    )

    try:
        print("=" * 72)
        print("Day18 Final Checkpoint Selection")
        print("=" * 72)
        print("device:", device)
        print("radius_mm:", radius_mm)
        print("held-out eval_seed:", eval_seed)
        print(
            "evaluation:",
            f"{num_directions} directions x {episodes_per_direction} = "
            f"{num_directions * episodes_per_direction} episodes/condition",
        )
        print("Day17 cube_to_grasp scale:", day17_scale)
        print()

        # Reference is physical-control baseline and is evaluated once with the
        # raw observation contract. Observation scaling cannot affect a zero
        # residual policy.
        reference_summary, reference_rows = evaluate_condition(
            env=env_raw,
            perturb=perturb_raw,
            policy=None,
            angles_deg=angles_deg,
            episodes_per_direction=episodes_per_direction,
            seed=eval_seed,
            max_steps=max_steps,
            lift_threshold_m=lift_threshold_m,
            saturation_threshold=saturation_threshold,
        )

        all_episode_rows: list[dict] = []
        summary_rows: list[dict] = []
        policy_rows_by_label: dict[str, list[dict]] = {}
        policies_by_label = {}

        for row in reference_rows:
            out = dict(row)
            out["condition"] = "reference"
            out["checkpoint"] = ""
            out["cube_to_grasp_scale"] = 1.0
            all_episode_rows.append(out)

        print(
            "Reference:",
            f"{reference_summary['success_count']}/{reference_summary['episodes']} "
            f"= {reference_summary['success_rate']:.3f}",
        )

        for item in checkpoints:
            label = str(item["label"])
            scale = float(item["scale"])
            env = env_raw if np.isclose(scale, 1.0) else env_scaled
            perturb = perturb_raw if np.isclose(scale, 1.0) else perturb_scaled

            policy = make_policy(
                config,
                env,
                device,
                train_seed,
            )
            policy.load(item["checkpoint"])
            policies_by_label[label] = (policy, env, perturb)

            summary, rows = evaluate_condition(
                env=env,
                perturb=perturb,
                policy=policy,
                angles_deg=angles_deg,
                episodes_per_direction=episodes_per_direction,
                seed=eval_seed,
                max_steps=max_steps,
                lift_threshold_m=lift_threshold_m,
                saturation_threshold=saturation_threshold,
            )

            pairs = paired_counts(reference_rows, rows)

            record = {
                "condition": label,
                "checkpoint": str(item["checkpoint"]),
                "training_step": (
                    "" if item["training_step"] is None else int(item["training_step"])
                ),
                "cube_to_grasp_scale": scale,
                **summary,
                **pairs,
            }
            add_common_metrics(record)
            record["gate3_eligible"] = bool(
                float(record["success_rate"]) >= gate_min_success_rate
            )

            summary_rows.append(record)
            policy_rows_by_label[label] = rows

            for row in rows:
                out = dict(row)
                out["condition"] = label
                out["checkpoint"] = str(item["checkpoint"])
                out["cube_to_grasp_scale"] = scale
                all_episode_rows.append(out)

            print(
                f"{label:12s}: "
                f"success={record['success_count']}/{record['episodes']} "
                f"({record['success_rate']:.3f}) "
                f"rescued={record['rescued']} broken={record['broken']} "
                f"sat={record['residual_saturation_rate']:.3f} "
                f"L2={record['mean_residual_l2']:.3f} "
                f"gate3={record['gate3_eligible']}"
            )

        # Controlled matched-training-budget comparison: Day13 25k vs Day17 25k.
        matched_25k = paired_policy_delta(
            policy_rows_by_label["day13_25k"],
            policy_rows_by_label["day17_rank1"],
        )
        matched_25k.update(
            {
                "day13_25k_success_rate": next(
                    float(r["success_rate"])
                    for r in summary_rows
                    if r["condition"] == "day13_25k"
                ),
                "day17_25k_success_rate": next(
                    float(r["success_rate"])
                    for r in summary_rows
                    if r["condition"] == "day17_rank1"
                ),
            }
        )
        matched_25k["success_rate_improvement"] = (
            matched_25k["day17_25k_success_rate"]
            - matched_25k["day13_25k_success_rate"]
        )

        ranked = sorted(summary_rows, key=rank_key, reverse=True)
        selected = dict(ranked[0])
        gate3_passed = bool(selected["success_rate"] >= gate_min_success_rate)

        frozen_checkpoint = None
        frozen_metadata = None

        if gate3_passed and not args.no_freeze:
            final_checkpoint_dir = args.freeze_dir / "checkpoints"
            final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

            frozen_checkpoint = final_checkpoint_dir / "final_selected.pt"
            shutil.copy2(Path(selected["checkpoint"]), frozen_checkpoint)

            frozen_metadata = {
                "selected_condition": selected["condition"],
                "source_checkpoint": selected["checkpoint"],
                "frozen_checkpoint": str(frozen_checkpoint),
                "sha256": sha256_file(frozen_checkpoint),
                "cube_to_grasp_scale": float(selected["cube_to_grasp_scale"]),
                "radius_mm": radius_mm,
                "eval_seed": eval_seed,
                "num_directions": num_directions,
                "episodes_per_direction": episodes_per_direction,
                "success_count": int(selected["success_count"]),
                "episodes": int(selected["episodes"]),
                "success_rate": float(selected["success_rate"]),
                "rescued": int(selected["rescued"]),
                "broken": int(selected["broken"]),
                "residual_saturation_rate": float(
                    selected["residual_saturation_rate"]
                ),
                "mean_residual_l2": float(selected["mean_residual_l2"]),
                "gate3_min_success_rate": gate_min_success_rate,
                "gate3_passed": True,
            }

            args.freeze_dir.mkdir(parents=True, exist_ok=True)
            with (args.freeze_dir / "final_policy_metadata.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(frozen_metadata, f, ensure_ascii=False, indent=2)

        # Safety instrumentation audit on the selected model. This only checks
        # what signals exist; it does not assert that collision detection is valid.
        selected_label = str(selected["condition"])
        selected_policy, selected_env, selected_perturb = policies_by_label[
            selected_label
        ]
        safety = safety_audit(
            env=selected_env,
            perturb=selected_perturb,
            policy=selected_policy,
            seed=eval_seed,
            max_steps=max_steps,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "summary.csv", summary_rows)
        write_csv(args.output_dir / "episodes.csv", all_episode_rows)

        result = {
            "evaluation_protocol": {
                "radius_mm": radius_mm,
                "eval_seed": eval_seed,
                "num_directions": num_directions,
                "episodes_per_direction": episodes_per_direction,
                "episodes_per_condition": num_directions * episodes_per_direction,
                "angles_deg": angles_deg,
                "gate3_min_success_rate": gate_min_success_rate,
                "day17_cube_to_grasp_scale": day17_scale,
            },
            "reference": reference_summary,
            "matched_25k_comparison": matched_25k,
            "ranked_candidates": ranked,
            "selected": selected,
            "gate3_passed": gate3_passed,
            "frozen_checkpoint": (
                None if frozen_checkpoint is None else str(frozen_checkpoint)
            ),
            "frozen_metadata": frozen_metadata,
            "safety_info_audit": safety,
        }

        with (args.output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print()
        print("-" * 72)
        print("Matched 25k comparison: Day13 scale1 -> Day17 scale10")
        print("-" * 72)
        print(
            "success:",
            f"{matched_25k['day13_25k_success_rate']:.3f} -> "
            f"{matched_25k['day17_25k_success_rate']:.3f}",
        )
        print(
            "improved/degraded/net:",
            matched_25k["improved"],
            matched_25k["degraded"],
            matched_25k["net_gain"],
        )

        print()
        print("-" * 72)
        print("Final ranking")
        print("-" * 72)
        for rank, row in enumerate(ranked, start=1):
            print(
                f"{rank}. {row['condition']:12s} "
                f"success={row['success_count']}/{row['episodes']} "
                f"({row['success_rate']:.3f}) "
                f"broken={row['broken']} "
                f"sat={row['residual_saturation_rate']:.3f} "
                f"L2={row['mean_residual_l2']:.3f}"
            )

        print()
        print("selected:", selected_label)
        print("Gate3 PASSED:", gate3_passed)
        if frozen_checkpoint is not None:
            print("frozen checkpoint:", frozen_checkpoint)
            print("SHA256:", frozen_metadata["sha256"])
        else:
            print("checkpoint was not frozen")

        print()
        print("collision-related key candidates:")
        if safety["collision_related_key_candidates"]:
            for key in safety["collision_related_key_candidates"]:
                print("  -", key)
        else:
            print("  (none found)")
        print(
            "NOTE: this audit does not prove that collision instrumentation is valid."
        )

        print("summary:", args.output_dir / "summary.json")

        return 0 if (gate3_passed or args.no_freeze) else 2

    finally:
        env_raw.close()
        env_scaled.close()


if __name__ == "__main__":
    raise SystemExit(main())
