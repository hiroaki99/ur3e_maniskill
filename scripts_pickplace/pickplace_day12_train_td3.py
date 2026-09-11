#!/usr/bin/env python3
"""Week3 generic TD3 trainer for Randomized Pick-and-Place.

Use this same script for:
- Day12 smoke training
- Day13 Cube-only formal training
- Day14 Goal-only formal training
- Day15 Both curriculum stages

Training reset:
    random XY angles on a fixed ring radius.

Evaluation:
    deterministic actor on fixed directions and matched Reference episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_pickplace"))

from pickplace_week2_common import build_env
from pickplace_week3_common import (
    CONFIG_PATH,
    DEFAULT_TRAJECTORY,
    ReplayBuffer,
    evaluate_paired_policy,
    load_config,
    make_td3,
    radii_for_mode,
    resolve_device,
    summarize_paired,
    write_csv,
    write_json,
)
from rrl.pick_place_evaluation import classify_pick_place_failure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=["cube_only", "goal_only", "both"],
    )
    p.add_argument("--radius-mm", type=float, required=True)
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--max-timesteps", type=int, default=None)
    p.add_argument("--start-timesteps", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--train-seed", type=int, default=None)
    p.add_argument("--eval-seed", type=int, default=None)
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for curriculum continuation. Replay buffer restarts.",
    )
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
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    td3cfg = cfg["week3"]["td3"]

    max_timesteps = int(
        args.max_timesteps
        if args.max_timesteps is not None
        else td3cfg["formal_timesteps"]
    )
    start_timesteps = int(
        args.start_timesteps
        if args.start_timesteps is not None
        else td3cfg["start_timesteps"]
    )
    eval_interval = int(
        args.eval_interval
        if args.eval_interval is not None
        else td3cfg["eval_interval"]
    )
    checkpoint_interval = int(
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else td3cfg["checkpoint_interval"]
    )
    train_seed = int(
        args.train_seed
        if args.train_seed is not None
        else td3cfg["train_seed"]
    )
    eval_seed = int(
        args.eval_seed
        if args.eval_seed is not None
        else td3cfg["eval_seed"]
    )
    batch_size = int(td3cfg["batch_size"])
    max_steps = int(td3cfg["max_steps_per_episode"])
    exploration_noise = float(td3cfg["exploration_noise"])
    eval_dirs = int(td3cfg["eval_num_directions"])
    eval_repeats = int(td3cfg["eval_episodes_per_direction"])
    replay_size = int(td3cfg["replay_size"])
    log_interval = int(td3cfg["log_interval"])
    device = resolve_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cube_radius_mm, goal_radius_mm = radii_for_mode(
        args.mode, args.radius_mm
    )

    # Training angles are independently randomized by the randomization wrapper.
    train_env, _ = build_env(
        config=cfg,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        seed=train_seed,
        randomization_mode=args.mode,
        cube_radius_mm=cube_radius_mm,
        goal_radius_mm=goal_radius_mm,
        fixed_cube_angle_deg=None,
        fixed_goal_angle_deg=None,
        apply_scaling=True,
    )

    try:
        state_dim = int(train_env.observation_space.shape[0])
        action_dim = int(train_env.action_space.shape[0])
        if state_dim != 56 or action_dim != 6:
            raise RuntimeError(
                f"Week3 contract mismatch: state={state_dim}, action={action_dim}"
            )

        policy = make_td3(
            cfg, train_env, device=device, seed=train_seed
        )
        if args.init_checkpoint is not None:
            policy.load(args.init_checkpoint)
            print("loaded init checkpoint:", args.init_checkpoint)

        replay = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=replay_size,
            seed=train_seed,
        )

        rng = np.random.default_rng(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_seed)

        metadata = {
            "mode": args.mode,
            "radius_mm": float(args.radius_mm),
            "cube_radius_mm": cube_radius_mm,
            "goal_radius_mm": goal_radius_mm,
            "trajectory": str(args.trajectory),
            "config_path": str(CONFIG_PATH),
            "observation_dim": state_dim,
            "action_dim": action_dim,
            "observation_scaling": cfg["observation_scaling"],
            "alpha": cfg["residual_pick_place"]["alpha"],
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "device": device,
            "sim_backend": args.sim_backend,
            "max_timesteps": max_timesteps,
            "start_timesteps": start_timesteps,
            "eval_interval": eval_interval,
            "init_checkpoint": (
                None if args.init_checkpoint is None else str(args.init_checkpoint)
            ),
        }
        write_json(args.output_dir / "metadata.json", metadata)

        observation, _ = train_env.reset(seed=train_seed)
        observation = np.asarray(observation, dtype=np.float32)

        episode_return = 0.0
        episode_steps = 0
        episode_index = 0
        episode_rows = []
        loss_rows = []
        eval_rows = []
        best_rank = None
        best_summary = None

        print("=" * 76)
        print("Week3 TD3 Training")
        print("=" * 76)
        print("mode:", args.mode)
        print("radius [mm]:", args.radius_mm)
        print("state/action:", state_dim, action_dim)
        print("device:", device)
        print("timesteps:", max_timesteps)
        print("start random:", start_timesteps)

        for t in range(1, max_timesteps + 1):
            if t <= start_timesteps:
                action = train_env.action_space.sample().astype(np.float32)
            else:
                action = policy.select_action(observation)
                noise = rng.normal(
                    0.0, exploration_noise, size=action_dim
                ).astype(np.float32)
                action = np.clip(action + noise, -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = train_env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            done = bool(terminated or truncated)

            replay.add(
                observation,
                action,
                next_obs,
                float(reward),
                done,
            )

            observation = next_obs
            episode_return += float(reward)
            episode_steps += 1

            if t > start_timesteps and len(replay) >= batch_size:
                train_info = policy.train(replay, batch_size=batch_size)
                if t % log_interval == 0:
                    loss_rows.append(
                        {
                            "timestep": t,
                            **train_info,
                        }
                    )

            if done or episode_steps >= max_steps:
                success = bool(info.get("success", False))
                cause = (
                    classify_pick_place_failure(info)
                    if done
                    else "timeout"
                )
                episode_rows.append(
                    {
                        "episode": episode_index,
                        "end_timestep": t,
                        "seed": train_seed + episode_index,
                        "success": success,
                        "failure_cause": cause,
                        "return": float(episode_return),
                        "steps": int(episode_steps),
                        "max_cube_lift_m": float(
                            info.get("max_cube_lift_m", 0.0)
                        ),
                        "goal_xy_error_m": float(
                            info.get("goal_xy_error_m", np.nan)
                        ),
                    }
                )
                episode_index += 1
                observation, _ = train_env.reset(
                    seed=train_seed + episode_index
                )
                observation = np.asarray(observation, dtype=np.float32)
                episode_return = 0.0
                episode_steps = 0

            if t % checkpoint_interval == 0:
                policy.save(checkpoint_dir / f"step_{t:07d}.pt")

            if t % eval_interval == 0 or t == max_timesteps:
                paired = evaluate_paired_policy(
                    config=cfg,
                    trajectory=args.trajectory,
                    sim_backend=args.sim_backend,
                    mode=args.mode,
                    radius_mm=args.radius_mm,
                    num_directions=eval_dirs,
                    episodes_per_direction=eval_repeats,
                    seed=eval_seed,
                    max_steps=max_steps,
                    policy=policy,
                    both_relation="same",
                    both_grid=False,
                )
                summary = summarize_paired(paired)
                row = {
                    "timestep": t,
                    **summary,
                }
                eval_rows.append(row)

                rank = (
                    int(summary["residual_success_count"]),
                    -int(summary["broken"]),
                    -float(summary["mean_residual_saturation_rate"]),
                    -float(summary["mean_residual_max_l2"]),
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_summary = row
                    policy.save(checkpoint_dir / "best.pt")
                    write_json(
                        args.output_dir / "best_summary.json",
                        best_summary,
                    )

                print(
                    f"eval step={t:7d} "
                    f"ref={summary['reference_success_rate']:.3f} "
                    f"res={summary['residual_success_rate']:.3f} "
                    f"rescued={summary['rescued']} "
                    f"broken={summary['broken']} "
                    f"sat={summary['mean_residual_saturation_rate']:.3f}"
                )

                write_csv(args.output_dir / "latest_eval_pairs.csv", paired)

        policy.save(checkpoint_dir / "final.pt")
        write_csv(args.output_dir / "episodes.csv", episode_rows)
        write_csv(args.output_dir / "losses.csv", loss_rows)
        write_csv(args.output_dir / "evaluations.csv", eval_rows)

        training_failures = Counter(
            r["failure_cause"] for r in episode_rows if not r["success"]
        )
        final_summary = {
            "episodes_completed": len(episode_rows),
            "training_success_count": sum(
                int(r["success"]) for r in episode_rows
            ),
            "training_failure_counts": dict(training_failures),
            "best_evaluation": best_summary,
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "final_checkpoint": str(checkpoint_dir / "final.pt"),
        }
        write_json(args.output_dir / "summary.json", final_summary)

        print("=" * 76)
        print("Training complete")
        print("=" * 76)
        print("best:", checkpoint_dir / "best.pt")
        print("final:", checkpoint_dir / "final.pt")
        print("summary:", args.output_dir / "summary.json")
        return 0
    finally:
        train_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
