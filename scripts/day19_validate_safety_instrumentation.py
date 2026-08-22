#!/usr/bin/env python3
"""
Day19 pre-check: audit safety / collision instrumentation.

This script does NOT claim that collision detection is physically valid.
It checks what signals are actually exposed by the current environment.

Key distinction
---------------
reward_terms["unsafe_collision"] is a reward term, not an independent
collision sensor. The reward model computes it from info["unsafe_collision"]
and defaults to False when that direct key is absent.

Therefore Day19 reports collision rate only after an independently validated
direct collision signal is available. Until then collision_rate should be N/A.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--radius-mm", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
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
        default=Path("reports/day19_safety_audit"),
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

    env = CubeToGraspObservationScaleWrapper(
        residual_env,
        scale=float(scale),
    )

    return env, perturb_env


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


def scalar(value, default=0.0):
    try:
        array = np.asarray(value)
        if array.size == 1:
            return float(array.reshape(-1)[0])
    except Exception:
        pass
    return float(default)


def main():
    args = parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    day19 = config.get("day19", {})

    eval_seed = int(
        args.eval_seed if args.eval_seed is not None
        else day19.get("eval_seed", 1900)
    )
    radius_mm = float(
        args.radius_mm if args.radius_mm is not None
        else day19.get("perturb_radius_mm", 20.0)
    )
    scale = float(
        args.scale if args.scale is not None
        else day19.get("cube_to_grasp_scale", 10.0)
    )
    max_steps = int(day19.get("max_steps_per_episode", 600))

    device = resolve_device(args.device)
    checkpoint = resolve_repo_path(args.checkpoint)
    trajectory = resolve_repo_path(args.trajectory)

    env, perturb_env = make_env(
        config=config,
        trajectory=trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
        scale=scale,
    )

    try:
        env.reset(seed=eval_seed)

        policy = make_policy(
            config=config,
            env=env,
            device=device,
            seed=int(config["day13"]["train_seed"]),
        )
        policy.load(checkpoint)

        action_dim = int(env.action_space.shape[0])

        conditions = [
            ("reference", None),
            ("residual_td3", policy),
        ]
        angles_deg = [0.0, 90.0, 180.0, 270.0]

        total_steps = 0
        direct_collision_key_steps = 0
        direct_collision_true_steps = 0
        reward_collision_key_steps = 0
        reward_collision_nonzero_steps = 0

        left_force_key_steps = 0
        right_force_key_steps = 0
        max_left_force = 0.0
        max_right_force = 0.0

        all_info_keys = set()
        all_reward_term_keys = set()

        for condition_index, (condition, current_policy) in enumerate(conditions):
            for angle_index, angle_deg in enumerate(angles_deg):
                perturb_env.fixed_angle_rad = float(np.deg2rad(angle_deg))

                episode_seed = (
                    eval_seed
                    + condition_index * 100_000
                    + angle_index * 1000
                )

                observation, _ = env.reset(seed=episode_seed)
                observation = np.asarray(observation, dtype=np.float32)

                for _ in range(max_steps):
                    if current_policy is None:
                        action = np.zeros(action_dim, dtype=np.float32)
                    else:
                        action = current_policy.select_action(observation)
                        action = np.clip(action, -1.0, 1.0).astype(np.float32)

                    observation, _, terminated, truncated, info = env.step(action)
                    observation = np.asarray(observation, dtype=np.float32)

                    total_steps += 1
                    all_info_keys.update(str(k) for k in info.keys())

                    if "unsafe_collision" in info:
                        direct_collision_key_steps += 1
                        if bool(info["unsafe_collision"]):
                            direct_collision_true_steps += 1

                    reward_terms = info.get("reward_terms", {})
                    if isinstance(reward_terms, dict):
                        all_reward_term_keys.update(str(k) for k in reward_terms.keys())

                        if "unsafe_collision" in reward_terms:
                            reward_collision_key_steps += 1
                            if abs(scalar(reward_terms["unsafe_collision"])) > 0.0:
                                reward_collision_nonzero_steps += 1

                    if "left_contact_force" in info:
                        left_force_key_steps += 1
                        max_left_force = max(
                            max_left_force,
                            scalar(info["left_contact_force"]),
                        )

                    if "right_contact_force" in info:
                        right_force_key_steps += 1
                        max_right_force = max(
                            max_right_force,
                            scalar(info["right_contact_force"]),
                        )

                    if terminated or truncated:
                        break

        if direct_collision_key_steps == 0:
            collision_status = "direct_signal_unavailable"
        elif direct_collision_key_steps < total_steps:
            collision_status = "direct_signal_partial_not_validated"
        else:
            collision_status = "direct_signal_present_not_physically_validated"

        # This script has no positive-control collision experiment, so it cannot
        # validate physical correctness even if a direct key exists.
        collision_rate_reportable = False

        summary = {
            "checkpoint": str(checkpoint),
            "radius_mm": radius_mm,
            "cube_to_grasp_scale": scale,
            "eval_seed": eval_seed,
            "episodes_checked": len(conditions) * len(angles_deg),
            "total_steps_checked": total_steps,
            "direct_unsafe_collision": {
                "key_steps": direct_collision_key_steps,
                "true_steps": direct_collision_true_steps,
            },
            "reward_terms_unsafe_collision": {
                "key_steps": reward_collision_key_steps,
                "nonzero_steps": reward_collision_nonzero_steps,
                "note": (
                    "This is a reward term computed from info['unsafe_collision']; "
                    "it is not an independent collision detector."
                ),
            },
            "contact_force": {
                "left_key_steps": left_force_key_steps,
                "right_key_steps": right_force_key_steps,
                "max_left_force_n": max_left_force,
                "max_right_force_n": max_right_force,
            },
            "collision_instrumentation_status": collision_status,
            "collision_rate_reportable": collision_rate_reportable,
            "all_info_keys": sorted(all_info_keys),
            "all_reward_term_keys": sorted(all_reward_term_keys),
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "summary.json"

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("=" * 72)
        print("Day19 Safety Instrumentation Audit")
        print("=" * 72)
        print("direct unsafe_collision key steps:",
              f"{direct_collision_key_steps}/{total_steps}")
        print("direct unsafe_collision true steps:", direct_collision_true_steps)
        print("reward_terms.unsafe_collision key steps:",
              f"{reward_collision_key_steps}/{total_steps}")
        print("max left/right contact force [N]:",
              f"{max_left_force:.3f}", f"{max_right_force:.3f}")
        print("collision status:", collision_status)
        print("collision rate reportable:", collision_rate_reportable)
        print("summary:", summary_path)

        if direct_collision_key_steps == 0:
            print()
            print(
                "Decision: current collision signal is not directly observable. "
                "Day19 final evaluation should record collision_rate=N/A."
            )
        else:
            print()
            print(
                "Decision: a direct signal exists, but this audit does not provide "
                "a physical positive-control validation. Keep collision_rate=N/A "
                "unless a separate collision validation is completed."
            )

        return 0

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
