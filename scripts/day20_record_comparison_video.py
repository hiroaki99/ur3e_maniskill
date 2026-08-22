#!/usr/bin/env python3
"""
Day20 matched video recorder.

Records the same Day19 rescued case twice:
  1) Reference-only: residual action = 0
  2) Residual RL: frozen Day18 final checkpoint

The case angle and episode seed are identical between the two runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from mani_skill.utils.wrappers.record import RecordEpisode

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rrl import CubePositionPerturbationWrapper, ResidualPickLiftEnv, TD3
from rrl.observation_scaling import CubeToGraspObservationScaleWrapper

CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"
DEFAULT_TRAJECTORY = (
    REPO_ROOT / "trajectories" / "day6_pick_lift_reference_v2.json"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--angle-deg", type=float, required=True)
    p.add_argument("--episode-seed", type=int, required=True)
    p.add_argument("--case-name", type=str, default="case01")

    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/day18_final/checkpoints/final_selected.pt"),
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("runs/day18_final/final_policy_metadata.json"),
    )
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)

    p.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=["physx_cpu", "physx_cuda"],
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("videos/day20"),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless Reference fails and Residual succeeds.",
    )
    return p.parse_args()


def resolve_repo_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_policy(*, config, env, device):
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
        seed=int(config["day13"]["train_seed"]),
    )


def make_record_env(
    *,
    config,
    trajectory,
    sim_backend,
    radius_mm,
    angle_deg,
    episode_seed,
    observation_scale,
    output_dir,
    video_fps,
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
        render_mode="rgb_array",
        max_episode_steps=int(day9.get("base_env_max_episode_steps", 1000)),
    )

    radius_m = float(radius_mm) / 1000.0

    perturb_env = CubePositionPerturbationWrapper(
        base_env,
        radius_min_m=radius_m,
        radius_max_m=radius_m,
        direction_mode="random_angle",
        fixed_angle_rad=float(np.deg2rad(angle_deg)),
        seed=episode_seed,
    )

    residual_env = ResidualPickLiftEnv(
        env=perturb_env,
        trajectory_path=trajectory,
        config_path=CONFIG_PATH,
    )

    scaled_env = CubeToGraspObservationScaleWrapper(
        residual_env,
        scale=float(observation_scale),
    )

    record_env = RecordEpisode(
        scaled_env,
        output_dir=str(output_dir),
        save_trajectory=False,
        save_video=True,
        info_on_video=False,
        save_on_reset=True,
        video_fps=int(video_fps),
        render_substeps=False,
        avoid_overwriting_video=False,
    )

    return record_env


def run_one(
    *,
    condition,
    config,
    trajectory,
    checkpoint,
    device,
    sim_backend,
    radius_mm,
    angle_deg,
    episode_seed,
    observation_scale,
    output_root,
    case_name,
    video_fps,
    max_steps,
):
    condition_dir = output_root / case_name / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    env = make_record_env(
        config=config,
        trajectory=trajectory,
        sim_backend=sim_backend,
        radius_mm=radius_mm,
        angle_deg=angle_deg,
        episode_seed=episode_seed,
        observation_scale=observation_scale,
        output_dir=condition_dir,
        video_fps=video_fps,
    )

    try:
        observation, reset_info = env.reset(seed=episode_seed)
        observation = np.asarray(observation, dtype=np.float32)

        action_dim = int(env.action_space.shape[0])

        policy = None
        if condition == "residual":
            policy = make_policy(
                config=config,
                env=env,
                device=device,
            )
            policy.load(checkpoint)

        episode_return = 0.0
        max_cube_lift = 0.0
        grasp_detected = False
        final_info = {}
        steps = 0

        for _ in range(max_steps):
            if policy is None:
                action = np.zeros(action_dim, dtype=np.float32)
            else:
                action = policy.select_action(observation)
                action = np.clip(
                    np.asarray(action, dtype=np.float32),
                    -1.0,
                    1.0,
                )

            observation, reward, terminated, truncated, info = env.step(action)
            observation = np.asarray(observation, dtype=np.float32)

            steps += 1
            episode_return += float(reward)
            final_info = dict(info)

            grasp_detected = (
                grasp_detected
                or (
                    bool(info.get("left_contact", False))
                    and bool(info.get("right_contact", False))
                )
            )
            max_cube_lift = max(
                max_cube_lift,
                float(info.get("cube_lift_m", 0.0)),
            )

            if terminated or truncated:
                break

        success = bool(final_info.get("success", False))

        video_name = f"{case_name}_{condition}"
        env.flush_video(
            name=video_name,
            verbose=True,
        )

        result = {
            "condition": condition,
            "angle_deg": float(angle_deg),
            "episode_seed": int(episode_seed),
            "cube_offset_m": reset_info.get("cube_offset_m"),
            "success": success,
            "grasp_detected": bool(grasp_detected),
            "max_cube_lift_m": float(max_cube_lift),
            "episode_return": float(episode_return),
            "steps": int(steps),
            "terminal_reason": final_info.get("terminal_reason"),
            "video_path": str(condition_dir / f"{video_name}.mp4"),
        }

        with (condition_dir / f"{video_name}_result.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result
    finally:
        env.close()


def main():
    args = parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint = resolve_repo_path(args.checkpoint)
    metadata_path = resolve_repo_path(args.metadata)
    trajectory = resolve_repo_path(args.trajectory)
    output_root = resolve_repo_path(args.output_dir)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    actual_sha = sha256_file(checkpoint)
    expected_sha = metadata.get("sha256")

    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            "Frozen checkpoint SHA256 mismatch:\n"
            f" expected={expected_sha}\n"
            f" actual  ={actual_sha}"
        )

    observation_scale = float(metadata["cube_to_grasp_scale"])
    radius_mm = float(metadata["radius_mm"])
    device = resolve_device(args.device)

    max_steps = int(
        config.get("day19", {}).get(
            "max_steps_per_episode",
            config["day13"]["max_steps_per_episode"],
        )
    )

    print("=" * 72)
    print("Day20 Matched Video Recording")
    print("=" * 72)
    print("case:", args.case_name)
    print("angle_deg:", args.angle_deg)
    print("episode_seed:", args.episode_seed)
    print("radius_mm:", radius_mm)
    print("cube_to_grasp_scale:", observation_scale)
    print("checkpoint SHA256:", actual_sha)
    print()

    reference = run_one(
        condition="reference",
        config=config,
        trajectory=trajectory,
        checkpoint=checkpoint,
        device=device,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        angle_deg=args.angle_deg,
        episode_seed=args.episode_seed,
        observation_scale=observation_scale,
        output_root=output_root,
        case_name=args.case_name,
        video_fps=args.video_fps,
        max_steps=max_steps,
    )

    residual = run_one(
        condition="residual",
        config=config,
        trajectory=trajectory,
        checkpoint=checkpoint,
        device=device,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        angle_deg=args.angle_deg,
        episode_seed=args.episode_seed,
        observation_scale=observation_scale,
        output_root=output_root,
        case_name=args.case_name,
        video_fps=args.video_fps,
        max_steps=max_steps,
    )

    rescued_reproduced = (
        (not reference["success"])
        and residual["success"]
    )

    comparison = {
        "case_name": args.case_name,
        "checkpoint": str(checkpoint),
        "sha256": actual_sha,
        "cube_to_grasp_scale": observation_scale,
        "radius_mm": radius_mm,
        "angle_deg": float(args.angle_deg),
        "episode_seed": int(args.episode_seed),
        "reference": reference,
        "residual": residual,
        "rescued_reproduced": bool(rescued_reproduced),
    }

    case_dir = output_root / args.case_name
    with (case_dir / "comparison_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("-" * 72)
    print(
        "Reference:",
        "SUCCESS" if reference["success"] else "FAIL",
        f"max_lift={reference['max_cube_lift_m']:.5f} m",
    )
    print(
        "Residual :",
        "SUCCESS" if residual["success"] else "FAIL",
        f"max_lift={residual['max_cube_lift_m']:.5f} m",
    )
    print("rescued reproduced:", rescued_reproduced)
    print("metadata:", case_dir / "comparison_metadata.json")

    if args.strict and not rescued_reproduced:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
