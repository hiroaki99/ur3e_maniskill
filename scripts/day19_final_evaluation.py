#!/usr/bin/env python3
"""
Day19 final evaluation for the frozen UR3e Residual RL policy.

No training. No checkpoint selection.

Benchmarks
----------
A. Nominal:
   cube perturbation radius = 0 mm
   one geometric condition, repeated with held-out episode seeds

B. Controlled perturbation:
   cube perturbation radius = 20 mm
   fixed angular grid around the nominal cube position

Policies
--------
Reference-only:
    residual action = 0

Residual RL:
    frozen Day18 checkpoint
    cube_to_grasp observation scaling = 10

Outputs
-------
summary.json
summary.csv
episode_diagnostics.csv
paired_results.csv
failure_counts.csv

Important
---------
Day15 failure labels are reused as operational diagnostic labels.
They are not proof of physical causality.

Collision rate is reported only when config day19 explicitly marks
collision instrumentation as physically validated. Otherwise it is N/A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rrl import CubePositionPerturbationWrapper, ResidualPickLiftEnv, TD3
from rrl.observation_scaling import CubeToGraspObservationScaleWrapper
from scripts.day15_classify_failures import classify_failure, PHASE_NAMES

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"
DEFAULT_TRAJECTORY = (
    REPO_ROOT / "trajectories" / "day6_pick_lift_reference_v2.json"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/day18_final/checkpoints/final_selected.pt"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("runs/day18_final/final_policy_metadata.json"),
    )
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)

    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--nominal-episodes", type=int, default=None)
    parser.add_argument("--perturb-num-directions", type=int, default=None)
    parser.add_argument("--perturb-episodes-per-direction", type=int, default=None)

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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/day19_final_evaluation"),
    )

    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def resolve_repo_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scalar(value, default=0.0):
    try:
        array = np.asarray(value)
        if array.size == 1:
            return float(array.reshape(-1)[0])
    except Exception:
        pass
    return float(default)


def observation_vector(observation):
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def cube_position(observation):
    observation = observation_vector(observation)
    return np.asarray(observation[15:18], dtype=np.float64)


def cube_to_grasp_distance(observation, observation_scale: float):
    observation = observation_vector(observation)
    vector = np.asarray(observation[30:33], dtype=np.float64)
    return float(np.linalg.norm(vector / float(observation_scale)))


def phase_index(observation):
    observation = observation_vector(observation)
    return int(np.argmax(observation[33:40]))


def make_env(
    *,
    config,
    trajectory,
    sim_backend,
    radius_mm,
    seed,
    scale,
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

    scaled_env = CubeToGraspObservationScaleWrapper(
        residual_env,
        scale=float(scale),
    )

    return scaled_env, perturb_env


def make_policy(*, config, env, device, seed):
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


def run_episode(
    *,
    env,
    perturb_env,
    policy,
    condition,
    benchmark,
    angle_deg,
    repeat,
    episode_seed,
    max_steps,
    day15,
    observation_scale,
):
    perturb_env.fixed_angle_rad = float(np.deg2rad(angle_deg))

    observation, reset_info = env.reset(seed=episode_seed)
    observation = observation_vector(observation)

    action_dim = int(env.action_space.shape[0])
    alpha = float(env.alpha)

    initial_cube_position = cube_position(observation)

    min_grasp_distance = cube_to_grasp_distance(
        observation,
        observation_scale,
    )

    ever_left = False
    ever_right = False
    ever_bilateral = False

    current_bilateral_run = 0
    max_bilateral_run = 0

    first_left_step = None
    first_right_step = None
    first_bilateral_step = None

    lost_contact_after_bilateral = False
    lost_contact_after_lift = False

    max_cube_lift = 0.0
    final_cube_lift = 0.0
    max_xy_displacement = 0.0

    first_lift_start_step = None
    first_lift_threshold_step = None
    reached_lift_threshold = False

    residual_l2_sum = 0.0
    residual_abs_sum = 0.0
    residual_elements = 0
    residual_saturated_elements = 0

    saturation_threshold = float(day15["residual_saturation_threshold"])

    # Direct collision signal only. reward_terms is not treated as a detector.
    unsafe_collision = False
    unsafe_collision_observable = False
    direct_collision_key_steps = 0

    # Correct force keys for ResidualPickLiftEnv.
    force_observable = False
    max_left_force = 0.0
    max_right_force = 0.0

    episode_return = 0.0
    max_phase_index = phase_index(observation)
    final_info = {}

    terminated = False
    truncated = False
    executed_steps = 0

    lift_threshold = float(day15["lift_threshold_m"])
    lift_start_threshold = float(day15["lift_start_threshold_m"])

    for step in range(max_steps):
        if policy is None:
            residual_action = np.zeros(action_dim, dtype=np.float32)
        else:
            residual_action = policy.select_action(observation)
            residual_action = np.asarray(residual_action, dtype=np.float32)
            residual_action = np.clip(residual_action, -1.0, 1.0)

        residual_l2_sum += float(np.linalg.norm(residual_action))
        residual_abs_sum += float(np.sum(np.abs(residual_action)))
        residual_elements += int(residual_action.size)
        residual_saturated_elements += int(
            np.count_nonzero(np.abs(residual_action) >= saturation_threshold)
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(residual_action)

        next_observation = observation_vector(next_observation)
        executed_steps += 1
        episode_return += float(reward)
        final_info = info

        current_phase_index = phase_index(next_observation)
        max_phase_index = max(max_phase_index, current_phase_index)

        left_contact = bool(info.get("left_contact", False))
        right_contact = bool(info.get("right_contact", False))
        bilateral = left_contact and right_contact

        if left_contact:
            if not ever_left:
                first_left_step = step
            ever_left = True

        if right_contact:
            if not ever_right:
                first_right_step = step
            ever_right = True

        if bilateral:
            if not ever_bilateral:
                first_bilateral_step = step

            ever_bilateral = True
            current_bilateral_run += 1
            max_bilateral_run = max(max_bilateral_run, current_bilateral_run)
        else:
            if ever_bilateral:
                lost_contact_after_bilateral = True
            if reached_lift_threshold:
                lost_contact_after_lift = True
            current_bilateral_run = 0

        current_cube_position = cube_position(next_observation)

        xy_displacement = float(
            np.linalg.norm(
                current_cube_position[:2] - initial_cube_position[:2]
            )
        )
        max_xy_displacement = max(max_xy_displacement, xy_displacement)

        cube_lift = scalar(
            info.get("cube_lift_m"),
            default=(
                current_cube_position[2]
                - initial_cube_position[2]
            ),
        )

        final_cube_lift = cube_lift
        max_cube_lift = max(max_cube_lift, cube_lift)

        if first_lift_start_step is None and cube_lift >= lift_start_threshold:
            first_lift_start_step = step

        if not reached_lift_threshold and cube_lift >= lift_threshold:
            reached_lift_threshold = True
            first_lift_threshold_step = step

        min_grasp_distance = min(
            min_grasp_distance,
            cube_to_grasp_distance(
                next_observation,
                observation_scale,
            ),
        )

        if "unsafe_collision" in info:
            unsafe_collision_observable = True
            direct_collision_key_steps += 1
            unsafe_collision = (
                unsafe_collision
                or bool(info["unsafe_collision"])
            )

        if "left_contact_force" in info:
            force_observable = True
            max_left_force = max(
                max_left_force,
                scalar(info["left_contact_force"]),
            )

        if "right_contact_force" in info:
            force_observable = True
            max_right_force = max(
                max_right_force,
                scalar(info["right_contact_force"]),
            )

        observation = next_observation

        if terminated or truncated:
            break

    success = bool(final_info.get("success", False))

    timeout = bool(
        not terminated
        and not truncated
        and executed_steps >= max_steps
    )

    mean_residual_l2 = residual_l2_sum / max(executed_steps, 1)
    mean_abs_residual = residual_abs_sum / max(residual_elements, 1)
    saturation_rate = (
        residual_saturated_elements / max(residual_elements, 1)
    )

    metrics = {
        "benchmark": benchmark,
        "condition": condition,
        "angle_deg": float(angle_deg),
        "repeat": int(repeat),
        "seed": int(episode_seed),
        "offset_x_mm": float(
            reset_info.get("cube_offset_m", [0.0, 0.0, 0.0])[0] * 1000.0
        ),
        "offset_y_mm": float(
            reset_info.get("cube_offset_m", [0.0, 0.0, 0.0])[1] * 1000.0
        ),

        "success": success,
        "episode_return": float(episode_return),
        "steps": int(executed_steps),

        "max_phase_index": int(max_phase_index),
        "max_phase_name": (
            PHASE_NAMES[max_phase_index]
            if 0 <= max_phase_index < len(PHASE_NAMES)
            else "unknown"
        ),

        "ever_left_contact": bool(ever_left),
        "ever_right_contact": bool(ever_right),
        "ever_bilateral_contact": bool(ever_bilateral),
        "first_left_step": first_left_step,
        "first_right_step": first_right_step,
        "first_bilateral_step": first_bilateral_step,
        "max_consecutive_bilateral_steps": int(max_bilateral_run),
        "final_consecutive_bilateral_steps": int(current_bilateral_run),
        "lost_contact_after_bilateral": bool(lost_contact_after_bilateral),
        "lost_contact_after_lift": bool(lost_contact_after_lift),

        "min_cube_to_grasp_m": float(min_grasp_distance),
        "max_cube_lift_m": float(max_cube_lift),
        "final_cube_lift_m": float(final_cube_lift),
        "first_lift_start_step": first_lift_start_step,
        "first_lift_threshold_step": first_lift_threshold_step,
        "reached_lift_threshold": bool(reached_lift_threshold),
        "max_cube_xy_displacement_m": float(max_xy_displacement),

        "unsafe_collision_observable": bool(unsafe_collision_observable),
        "unsafe_collision": bool(unsafe_collision),
        "direct_collision_key_steps": int(direct_collision_key_steps),

        "force_observable": bool(force_observable),
        "max_left_contact_force_n": float(max_left_force),
        "max_right_contact_force_n": float(max_right_force),

        "mean_residual_l2": float(mean_residual_l2),
        "mean_abs_residual": float(mean_abs_residual),
        "effective_mean_abs_residual": float(alpha * mean_abs_residual),
        "residual_saturation_rate": float(saturation_rate),

        "terminal_reason": final_info.get("terminal_reason"),
        "timeout": timeout,
    }

    metrics["failure_cause"] = classify_failure(metrics, day15)

    return metrics


def failure_counter(rows):
    return Counter(
        row["failure_cause"]
        for row in rows
        if not row["success"]
    )


def paired_comparison(reference_rows, residual_rows):
    ref_map = {
        (
            float(r["angle_deg"]),
            int(r["repeat"]),
            int(r["seed"]),
        ): r
        for r in reference_rows
    }

    paired_rows = []
    counts = {
        "rescued": 0,
        "broken": 0,
        "kept_success": 0,
        "kept_failure": 0,
    }

    for residual in residual_rows:
        key = (
            float(residual["angle_deg"]),
            int(residual["repeat"]),
            int(residual["seed"]),
        )

        if key not in ref_map:
            raise RuntimeError(f"Missing paired reference episode: {key}")

        reference = ref_map[key]

        ref_success = bool(reference["success"])
        res_success = bool(residual["success"])

        if (not ref_success) and res_success:
            category = "rescued"
        elif ref_success and (not res_success):
            category = "broken"
        elif ref_success and res_success:
            category = "kept_success"
        else:
            category = "kept_failure"

        counts[category] += 1

        paired_rows.append(
            {
                "benchmark": residual["benchmark"],
                "angle_deg": residual["angle_deg"],
                "repeat": residual["repeat"],
                "seed": residual["seed"],
                "category": category,
                "reference_success": ref_success,
                "residual_success": res_success,
                "reference_failure_cause": reference["failure_cause"],
                "residual_failure_cause": residual["failure_cause"],
                "reference_max_lift_m": reference["max_cube_lift_m"],
                "residual_max_lift_m": residual["max_cube_lift_m"],
                "reference_return": reference["episode_return"],
                "residual_return": residual["episode_return"],
                "residual_l2": residual["mean_residual_l2"],
                "residual_saturation_rate": residual["residual_saturation_rate"],
            }
        )

    counts["net_gain"] = counts["rescued"] - counts["broken"]
    return counts, paired_rows


def summarize_condition(
    rows,
    *,
    lift_threshold,
    collision_instrumentation_validated,
):
    episodes = len(rows)
    success_count = sum(int(r["success"]) for r in rows)
    grasp_count = sum(int(r["ever_bilateral_contact"]) for r in rows)
    lift_count = sum(
        int(r["max_cube_lift_m"] >= lift_threshold)
        for r in rows
    )
    drop_count = sum(
        int(r["failure_cause"] == "drop_after_lift")
        for r in rows
    )

    direct_observable_count = sum(
        int(r["unsafe_collision_observable"])
        for r in rows
    )
    raw_collision_count = sum(
        int(r["unsafe_collision"])
        for r in rows
    )

    collision_rate = None
    if (
        collision_instrumentation_validated
        and direct_observable_count == episodes
        and episodes > 0
    ):
        collision_rate = raw_collision_count / episodes

    counter = failure_counter(rows)

    def mean(key):
        if not rows:
            return 0.0
        return float(np.mean([float(r[key]) for r in rows]))

    return {
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": success_count / episodes if episodes else 0.0,
        "grasp_count": grasp_count,
        "grasp_rate": grasp_count / episodes if episodes else 0.0,
        "lift_count": lift_count,
        "lift_rate": lift_count / episodes if episodes else 0.0,
        "drop_count": drop_count,
        "drop_rate": drop_count / episodes if episodes else 0.0,
        "mean_return": mean("episode_return"),
        "mean_max_cube_lift_m": mean("max_cube_lift_m"),
        "mean_residual_l2": mean("mean_residual_l2"),
        "mean_abs_residual": mean("mean_abs_residual"),
        "effective_mean_abs_residual": mean("effective_mean_abs_residual"),
        "residual_saturation_rate": mean("residual_saturation_rate"),
        "max_left_contact_force_n": (
            max(float(r["max_left_contact_force_n"]) for r in rows)
            if rows else 0.0
        ),
        "max_right_contact_force_n": (
            max(float(r["max_right_contact_force_n"]) for r in rows)
            if rows else 0.0
        ),
        "collision_direct_observable_episodes": direct_observable_count,
        "collision_raw_count": raw_collision_count,
        "collision_rate": collision_rate,
        "failure_counts": dict(counter),
    }


def make_failure_count_rows(benchmark, condition, rows):
    counter = failure_counter(rows)
    failures = [r for r in rows if not r["success"]]

    output = []
    for cause, count in counter.most_common():
        output.append(
            {
                "benchmark": benchmark,
                "condition": condition,
                "failure_cause": cause,
                "count": int(count),
                "rate_among_failures": (
                    count / len(failures) if failures else 0.0
                ),
                "rate_all_episodes": (
                    count / len(rows) if rows else 0.0
                ),
            }
        )
    return output


def evaluate_benchmark(
    *,
    benchmark,
    radius_mm,
    angles_deg,
    episodes_per_direction,
    eval_seed,
    env,
    perturb_env,
    policy,
    max_steps,
    day15,
    observation_scale,
):
    reference_rows = []
    residual_rows = []

    for angle_index, angle_deg in enumerate(angles_deg):
        for repeat in range(episodes_per_direction):
            episode_seed = eval_seed + angle_index * 1000 + repeat

            reference_rows.append(
                run_episode(
                    env=env,
                    perturb_env=perturb_env,
                    policy=None,
                    condition="reference",
                    benchmark=benchmark,
                    angle_deg=angle_deg,
                    repeat=repeat,
                    episode_seed=episode_seed,
                    max_steps=max_steps,
                    day15=day15,
                    observation_scale=observation_scale,
                )
            )

            residual_rows.append(
                run_episode(
                    env=env,
                    perturb_env=perturb_env,
                    policy=policy,
                    condition="residual_td3",
                    benchmark=benchmark,
                    angle_deg=angle_deg,
                    repeat=repeat,
                    episode_seed=episode_seed,
                    max_steps=max_steps,
                    day15=day15,
                    observation_scale=observation_scale,
                )
            )

    paired_counts, paired_rows = paired_comparison(
        reference_rows,
        residual_rows,
    )

    return reference_rows, residual_rows, paired_counts, paired_rows


def main():
    args = parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    day15 = config["day15"]
    day19 = config.get("day19", {})

    checkpoint = resolve_repo_path(
        Path(day19.get("final_checkpoint", args.checkpoint))
        if args.checkpoint == Path("runs/day18_final/checkpoints/final_selected.pt")
        else args.checkpoint
    )
    metadata_path = resolve_repo_path(
        Path(day19.get("final_metadata", args.metadata))
        if args.metadata == Path("runs/day18_final/final_policy_metadata.json")
        else args.metadata
    )
    trajectory = resolve_repo_path(args.trajectory)

    eval_seed = int(
        args.eval_seed if args.eval_seed is not None
        else day19.get("eval_seed", 1900)
    )
    nominal_episodes = int(
        args.nominal_episodes if args.nominal_episodes is not None
        else day19.get("nominal_episodes", 64)
    )
    perturb_num_directions = int(
        args.perturb_num_directions
        if args.perturb_num_directions is not None
        else day19.get("perturb_num_directions", 64)
    )
    perturb_episodes_per_direction = int(
        args.perturb_episodes_per_direction
        if args.perturb_episodes_per_direction is not None
        else day19.get("perturb_episodes_per_direction", 1)
    )

    nominal_radius_mm = float(day19.get("nominal_radius_mm", 0.0))
    perturb_radius_mm = float(day19.get("perturb_radius_mm", 20.0))
    observation_scale = float(day19.get("cube_to_grasp_scale", 10.0))
    max_steps = int(day19.get("max_steps_per_episode", 600))
    target_success_rate = float(day19.get("target_success_rate", 0.60))
    collision_validated = bool(
        day19.get("collision_instrumentation_validated", False)
    )

    device = resolve_device(args.device)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    checkpoint_sha256 = sha256_file(checkpoint)

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        expected_sha256 = metadata.get("sha256")
        if expected_sha256 and checkpoint_sha256 != expected_sha256:
            raise RuntimeError(
                "Frozen checkpoint SHA256 mismatch:\n"
                f" expected={expected_sha256}\n"
                f" actual  ={checkpoint_sha256}"
            )

        metadata_scale = metadata.get("cube_to_grasp_scale")
        if (
            metadata_scale is not None
            and not np.isclose(
                float(metadata_scale),
                observation_scale,
            )
        ):
            raise RuntimeError(
                "cube_to_grasp scale mismatch between Day18 metadata "
                f"({metadata_scale}) and Day19 config ({observation_scale})"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Day19 Final Evaluation")
    print("=" * 72)
    print("checkpoint:", checkpoint)
    print("SHA256:", checkpoint_sha256)
    print("device:", device)
    print("eval_seed:", eval_seed)
    print("cube_to_grasp_scale:", observation_scale)
    print("collision instrumentation validated:", collision_validated)

    all_episode_rows = []
    all_paired_rows = []
    all_failure_count_rows = []

    benchmark_results = {}

    benchmark_specs = [
        {
            "name": "nominal_0mm",
            "radius_mm": nominal_radius_mm,
            "angles_deg": [0.0],
            "episodes_per_direction": nominal_episodes,
            "seed": eval_seed,
        },
        {
            "name": "perturb_20mm",
            "radius_mm": perturb_radius_mm,
            "angles_deg": np.linspace(
                0.0,
                360.0,
                perturb_num_directions,
                endpoint=False,
            ).tolist(),
            "episodes_per_direction": perturb_episodes_per_direction,
            "seed": eval_seed + 1_000_000,
        },
    ]

    for spec in benchmark_specs:
        env, perturb_env = make_env(
            config=config,
            trajectory=trajectory,
            sim_backend=args.sim_backend,
            radius_mm=spec["radius_mm"],
            seed=spec["seed"],
            scale=observation_scale,
        )

        try:
            env.reset(seed=spec["seed"])

            policy = make_policy(
                config=config,
                env=env,
                device=device,
                seed=int(config["day13"]["train_seed"]),
            )
            policy.load(checkpoint)

            print()
            print("-" * 72)
            print(
                f"{spec['name']}: radius={spec['radius_mm']} mm, "
                f"{len(spec['angles_deg'])} directions x "
                f"{spec['episodes_per_direction']}"
            )
            print("-" * 72)

            (
                reference_rows,
                residual_rows,
                paired_counts,
                paired_rows,
            ) = evaluate_benchmark(
                benchmark=spec["name"],
                radius_mm=spec["radius_mm"],
                angles_deg=spec["angles_deg"],
                episodes_per_direction=spec["episodes_per_direction"],
                eval_seed=spec["seed"],
                env=env,
                perturb_env=perturb_env,
                policy=policy,
                max_steps=max_steps,
                day15=day15,
                observation_scale=observation_scale,
            )

            reference_summary = summarize_condition(
                reference_rows,
                lift_threshold=float(day15["lift_threshold_m"]),
                collision_instrumentation_validated=collision_validated,
            )
            residual_summary = summarize_condition(
                residual_rows,
                lift_threshold=float(day15["lift_threshold_m"]),
                collision_instrumentation_validated=collision_validated,
            )

            benchmark_results[spec["name"]] = {
                "radius_mm": float(spec["radius_mm"]),
                "reference": reference_summary,
                "residual_td3": residual_summary,
                "paired": paired_counts,
                "success_rate_improvement": (
                    residual_summary["success_rate"]
                    - reference_summary["success_rate"]
                ),
            }

            all_episode_rows.extend(reference_rows)
            all_episode_rows.extend(residual_rows)
            all_paired_rows.extend(paired_rows)

            all_failure_count_rows.extend(
                make_failure_count_rows(
                    spec["name"],
                    "reference",
                    reference_rows,
                )
            )
            all_failure_count_rows.extend(
                make_failure_count_rows(
                    spec["name"],
                    "residual_td3",
                    residual_rows,
                )
            )

            print(
                "Reference:",
                f"{reference_summary['success_count']}/"
                f"{reference_summary['episodes']}",
                f"= {reference_summary['success_rate']:.3f}",
            )
            print(
                "Residual :",
                f"{residual_summary['success_count']}/"
                f"{residual_summary['episodes']}",
                f"= {residual_summary['success_rate']:.3f}",
            )
            print(
                "Rescued/Broken/Net:",
                paired_counts["rescued"],
                paired_counts["broken"],
                paired_counts["net_gain"],
            )
            print(
                "Residual L2/Sat:",
                f"{residual_summary['mean_residual_l2']:.3f}",
                f"{residual_summary['residual_saturation_rate']:.3f}",
            )
            print("Reference failures:", reference_summary["failure_counts"])
            print("Residual failures :", residual_summary["failure_counts"])

        finally:
            env.close()

    perturb_success_rate = benchmark_results["perturb_20mm"][
        "residual_td3"
    ]["success_rate"]

    final_target_met = bool(perturb_success_rate >= target_success_rate)

    summary = {
        "final_policy": {
            "checkpoint": str(checkpoint),
            "sha256": checkpoint_sha256,
            "cube_to_grasp_scale": observation_scale,
            "day18_metadata": metadata,
        },
        "evaluation_protocol": {
            "eval_seed": eval_seed,
            "nominal_radius_mm": nominal_radius_mm,
            "nominal_episodes": nominal_episodes,
            "perturb_radius_mm": perturb_radius_mm,
            "perturb_num_directions": perturb_num_directions,
            "perturb_episodes_per_direction": perturb_episodes_per_direction,
            "max_steps_per_episode": max_steps,
            "target_success_rate": target_success_rate,
            "checkpoint_reselection_allowed": False,
        },
        "collision_instrumentation": {
            "physically_validated": collision_validated,
            "policy": (
                "collision_rate is N/A unless physically validated "
                "direct instrumentation is explicitly enabled"
            ),
        },
        "benchmarks": benchmark_results,
        "final_target_met_on_perturbation": final_target_met,
        "classification_note": (
            "Failure labels reuse Day15 operational diagnostic definitions; "
            "they are not proof of physical causality."
        ),
    }

    summary_rows = []
    for benchmark_name, benchmark in benchmark_results.items():
        paired = benchmark["paired"]

        for condition in ["reference", "residual_td3"]:
            s = benchmark[condition]
            row = {
                "benchmark": benchmark_name,
                "radius_mm": benchmark["radius_mm"],
                "condition": condition,
                "episodes": s["episodes"],
                "success_count": s["success_count"],
                "success_rate": s["success_rate"],
                "grasp_count": s["grasp_count"],
                "grasp_rate": s["grasp_rate"],
                "lift_count": s["lift_count"],
                "lift_rate": s["lift_rate"],
                "drop_count": s["drop_count"],
                "drop_rate": s["drop_rate"],
                "mean_return": s["mean_return"],
                "mean_max_cube_lift_m": s["mean_max_cube_lift_m"],
                "mean_residual_l2": s["mean_residual_l2"],
                "residual_saturation_rate": s["residual_saturation_rate"],
                "max_left_contact_force_n": s["max_left_contact_force_n"],
                "max_right_contact_force_n": s["max_right_contact_force_n"],
                "collision_rate": (
                    ""
                    if s["collision_rate"] is None
                    else s["collision_rate"]
                ),
                "rescued": paired["rescued"] if condition == "residual_td3" else "",
                "broken": paired["broken"] if condition == "residual_td3" else "",
                "net_gain": paired["net_gain"] if condition == "residual_td3" else "",
            }
            summary_rows.append(row)

    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "episode_diagnostics.csv", all_episode_rows)
    write_csv(args.output_dir / "paired_results.csv", all_paired_rows)
    write_csv(args.output_dir / "failure_counts.csv", all_failure_count_rows)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("Day19 Final Result")
    print("=" * 72)

    nominal = benchmark_results["nominal_0mm"]
    perturb = benchmark_results["perturb_20mm"]

    print(
        "Nominal Reference -> Residual:",
        f"{nominal['reference']['success_rate']:.3f}",
        "->",
        f"{nominal['residual_td3']['success_rate']:.3f}",
    )
    print(
        "20mm Reference -> Residual:",
        f"{perturb['reference']['success_rate']:.3f}",
        "->",
        f"{perturb['residual_td3']['success_rate']:.3f}",
    )
    print(
        "20mm Rescued/Broken/Net:",
        perturb["paired"]["rescued"],
        perturb["paired"]["broken"],
        perturb["paired"]["net_gain"],
    )
    print("Final target met:", final_target_met)
    print(
        "Collision rate:",
        "N/A" if not collision_validated
        else perturb["residual_td3"]["collision_rate"],
    )
    print("summary:", summary_path)
    print()
    print(
        "IMPORTANT: This script never reselects the checkpoint. "
        "The Day18 frozen policy remains final regardless of this result."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
