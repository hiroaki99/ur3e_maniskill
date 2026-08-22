#!/usr/bin/env python3
"""Day17 final TD3 training with observation-only cube_to_grasp scaling.

Controlled change versus Day13:
    TD3-visible obs[30:33] = raw cube_to_grasp * cube_to_grasp_scale

Everything else (reward, alpha, reference, TD3 architecture/hyperparameters,
20 mm perturbation, gripper script) is kept from Day13/Day11 config.

The run can continue to 50k steps, while step 30k is retained as the
controlled same-budget comparison against Day13.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    ReplayBuffer,
    ResidualPickLiftEnv,
    TD3,
    evaluate_residual_policy,
)
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/day17_td3_obs_scale10_seed1300"),
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

    scaled_env = CubeToGraspObservationScaleWrapper(
        residual_env,
        scale=scale,
    )

    return scaled_env, perturb_env


def paired_counts(reference_rows, residual_rows):
    if len(reference_rows) != len(residual_rows):
        raise RuntimeError("Reference/Residual episode counts differ")

    ref_map = {
        (float(r["angle_deg"]), int(r["repeat"]), int(r["seed"])): bool(r["success"])
        for r in reference_rows
    }

    rescued = broken = kept_success = kept_failure = 0
    for r in residual_rows:
        key = (float(r["angle_deg"]), int(r["repeat"]), int(r["seed"]))
        if key not in ref_map:
            raise RuntimeError(f"Missing paired reference episode: {key}")
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


def score_tuple(row):
    # Success is primary. Then fewer Broken, lower saturation, lower L2.
    return (
        int(row["success_count"]),
        -int(row["broken"]),
        -float(row["residual_saturation_rate"]),
        -float(row["mean_residual_l2"]),
    )


def main():
    args = parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    day11 = config["day11"]
    day12 = config["day12"]
    day17 = config["day17"]

    train_seed = int(day17.get("train_seed", 1300))
    eval_seed = int(day17.get("eval_seed", 1200))
    radius_mm = float(day17.get("train_radius_mm", day12["selected_radius_mm"]))
    scale = float(args.scale if args.scale is not None else day17["cube_to_grasp_scale"])

    max_timesteps = int(
        args.max_timesteps if args.max_timesteps is not None else day17["max_timesteps"]
    )
    start_timesteps = int(day17.get("start_timesteps", day11["start_timesteps"]))
    eval_interval = int(
        args.eval_interval if args.eval_interval is not None else day17["eval_interval"]
    )
    checkpoint_interval = int(day17["checkpoint_interval"])
    log_interval = int(day17["log_interval"])
    max_steps = int(day17["max_steps_per_episode"])
    eval_num_directions = int(day17["periodic_eval_num_directions"])
    eval_episodes_per_direction = int(day17["periodic_eval_episodes_per_direction"])
    lift_threshold_m = float(day17["lift_threshold_m"])
    saturation_threshold = float(day17["residual_saturation_threshold"])
    candidate_count = int(day17.get("candidate_count", 3))
    hypothesis_comparison_step = int(day17.get("hypothesis_comparison_step", 30000))

    device = resolve_device(args.device)

    rng = np.random.default_rng(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_env, _ = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=train_seed,
        scale=scale,
    )
    eval_env, eval_perturb = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
        scale=scale,
    )

    try:
        observation, reset_info = train_env.reset(seed=train_seed)
        observation = np.asarray(observation, dtype=np.float32)

        state_dim = int(train_env.observation_space.shape[0])
        action_dim = int(train_env.action_space.shape[0])
        max_action = float(train_env.action_space.high[0])

        policy = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
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
            seed=train_seed,
        )

        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=int(day11["replay_size"]),
            seed=train_seed,
        )

        exploration_noise = float(day11["exploration_noise"])
        batch_size = int(day11["batch_size"])

        angles_deg = np.linspace(
            0.0, 360.0, eval_num_directions, endpoint=False
        ).tolist()

        # Reference-only is checkpoint-independent: evaluate once and reuse for
        # Broken/Rescued pairing during periodic evaluations.
        reference_summary, reference_rows = evaluate_residual_policy(
            env=eval_env,
            perturbation_wrapper=eval_perturb,
            policy=None,
            angles_deg=angles_deg,
            episodes_per_direction=eval_episodes_per_direction,
            seed=eval_seed,
            max_steps=max_steps,
            lift_threshold_m=lift_threshold_m,
            saturation_threshold=saturation_threshold,
        )

        initial_eval, initial_rows = evaluate_residual_policy(
            env=eval_env,
            perturbation_wrapper=eval_perturb,
            policy=policy,
            angles_deg=angles_deg,
            episodes_per_direction=eval_episodes_per_direction,
            seed=eval_seed,
            max_steps=max_steps,
            lift_threshold_m=lift_threshold_m,
            saturation_threshold=saturation_threshold,
        )
        initial_pairs = paired_counts(reference_rows, initial_rows)

        episode_rows = []
        training_rows = []
        evaluation_rows = []
        evaluation_episode_rows = []
        errors = []

        initial_record = dict(initial_eval)
        initial_record.update(initial_pairs)
        initial_record["global_step"] = 0
        evaluation_rows.append(initial_record)
        for row in initial_rows:
            x = dict(row)
            x["global_step"] = 0
            evaluation_episode_rows.append(x)

        print("=" * 72)
        print("Day17 TD3 Training: cube_to_grasp observation scaling")
        print("=" * 72)
        print("device:", device)
        print("scale:", scale)
        print("training radius:", radius_mm, "mm")
        print("periodic eval:", eval_num_directions, "directions x", eval_episodes_per_direction)
        print("reference success:", reference_summary["success_rate"])
        print("initial TD3 success:", initial_eval["success_rate"])
        print("same-budget comparison checkpoint:", hypothesis_comparison_step)

        episode_number = 0
        episode_return = 0.0
        episode_steps = 0
        episode_grasp_detected = False
        episode_max_lift = 0.0
        episode_residual_l2_sum = 0.0
        episode_residual_abs_sum = 0.0
        episode_residual_elements = 0
        episode_saturation_count = 0
        current_offset = np.asarray(
            reset_info.get("cube_offset_m", [0.0, 0.0, 0.0]), dtype=np.float64
        )

        critic_updates = 0
        actor_updates = 0
        last_critic_loss = None
        last_actor_loss = None
        action_bound_violations = 0
        best_score = None
        best_global_step = None

        for t in range(max_timesteps):
            if t < start_timesteps:
                residual_action = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            else:
                residual_action = policy.select_action(observation)
                residual_action = residual_action + rng.normal(
                    0.0, exploration_noise, size=action_dim
                )
                residual_action = np.clip(
                    residual_action, -max_action, max_action
                ).astype(np.float32)

            if not np.all(np.isfinite(residual_action)):
                errors.append(f"action NaN/inf at {t}")
                break
            if np.any(residual_action < -max_action) or np.any(residual_action > max_action):
                action_bound_violations += 1

            episode_residual_l2_sum += float(np.linalg.norm(residual_action))
            episode_residual_abs_sum += float(np.sum(np.abs(residual_action)))
            episode_residual_elements += residual_action.size
            episode_saturation_count += int(
                np.count_nonzero(np.abs(residual_action) >= saturation_threshold)
            )

            next_observation, reward, terminated, truncated, info = train_env.step(
                residual_action
            )
            next_observation = np.asarray(next_observation, dtype=np.float32)
            reward = float(reward)
            done = bool(terminated or truncated)

            if not np.all(np.isfinite(next_observation)):
                errors.append(f"observation NaN/inf at {t}")
                break
            if not np.isfinite(reward):
                errors.append(f"reward NaN/inf at {t}")
                break

            replay_buffer.add(
                state=observation,
                action=residual_action,
                next_state=next_observation,
                reward=reward,
                done=done,
            )
            observation = next_observation

            episode_return += reward
            episode_steps += 1
            both_contact = bool(info.get("left_contact", False)) and bool(
                info.get("right_contact", False)
            )
            episode_grasp_detected = episode_grasp_detected or both_contact
            episode_max_lift = max(
                episode_max_lift, float(info.get("cube_lift_m", 0.0))
            )

            if t >= start_timesteps and len(replay_buffer) >= batch_size:
                metrics = policy.train(replay_buffer, batch_size=batch_size)
                critic_updates += 1
                last_critic_loss = float(metrics["critic_loss"])
                if not np.isfinite(last_critic_loss):
                    errors.append(f"critic NaN/inf at {t}")
                    break
                if metrics["actor_updated"]:
                    actor_updates += 1
                    last_actor_loss = float(metrics["actor_loss"])
                    if not np.isfinite(last_actor_loss):
                        errors.append(f"actor NaN/inf at {t}")
                        break
                training_rows.append(
                    {
                        "global_step": t + 1,
                        "critic_loss": metrics["critic_loss"],
                        "actor_updated": metrics["actor_updated"],
                        "actor_loss": metrics["actor_loss"] if metrics["actor_loss"] is not None else "",
                        "target_q_mean": metrics["target_q_mean"],
                        "current_q1_mean": metrics["current_q1_mean"],
                        "current_q2_mean": metrics["current_q2_mean"],
                    }
                )

            if done:
                mean_l2 = episode_residual_l2_sum / max(episode_steps, 1)
                mean_abs = episode_residual_abs_sum / max(episode_residual_elements, 1)
                sat = episode_saturation_count / max(episode_residual_elements, 1)
                episode_rows.append(
                    {
                        "episode": episode_number,
                        "global_step": t + 1,
                        "episode_steps": episode_steps,
                        "offset_x_mm": float(current_offset[0] * 1000.0),
                        "offset_y_mm": float(current_offset[1] * 1000.0),
                        "return": float(episode_return),
                        "success": bool(info.get("success", False)),
                        "grasp_detected": bool(episode_grasp_detected),
                        "lift_reached": bool(episode_max_lift >= lift_threshold_m),
                        "max_cube_lift_m": float(episode_max_lift),
                        "mean_residual_l2": float(mean_l2),
                        "mean_abs_residual": float(mean_abs),
                        "residual_saturation_rate": float(sat),
                        "terminal_reason": info.get("terminal_reason"),
                    }
                )

                episode_number += 1
                observation, reset_info = train_env.reset(seed=train_seed + episode_number)
                observation = np.asarray(observation, dtype=np.float32)
                current_offset = np.asarray(
                    reset_info.get("cube_offset_m", [0.0, 0.0, 0.0]), dtype=np.float64
                )
                episode_return = 0.0
                episode_steps = 0
                episode_grasp_detected = False
                episode_max_lift = 0.0
                episode_residual_l2_sum = 0.0
                episode_residual_abs_sum = 0.0
                episode_residual_elements = 0
                episode_saturation_count = 0

            if (t + 1) % log_interval == 0:
                print(
                    f"step={t + 1:6d} episodes={episode_number:4d} "
                    f"buffer={len(replay_buffer):6d} critic={last_critic_loss} "
                    f"actor={last_actor_loss}"
                )

            if (t + 1) % checkpoint_interval == 0:
                policy.save(checkpoint_dir / f"step_{t + 1:08d}.pt")

            if (t + 1) % eval_interval == 0:
                # Every evaluated policy must have a matching checkpoint so
                # ranking remains valid even in short smoke runs where
                # eval_interval != checkpoint_interval.
                eval_checkpoint_path = checkpoint_dir / f"step_{t + 1:08d}.pt"
                if not eval_checkpoint_path.exists():
                    policy.save(eval_checkpoint_path)

                eval_summary, eval_rows = evaluate_residual_policy(
                    env=eval_env,
                    perturbation_wrapper=eval_perturb,
                    policy=policy,
                    angles_deg=angles_deg,
                    episodes_per_direction=eval_episodes_per_direction,
                    seed=eval_seed,
                    max_steps=max_steps,
                    lift_threshold_m=lift_threshold_m,
                    saturation_threshold=saturation_threshold,
                )
                pairs = paired_counts(reference_rows, eval_rows)
                record = dict(eval_summary)
                record.update(pairs)
                record["global_step"] = t + 1
                evaluation_rows.append(record)

                for row in eval_rows:
                    x = dict(row)
                    x["global_step"] = t + 1
                    evaluation_episode_rows.append(x)

                print(
                    "\n[EVAL] "
                    f"step={t + 1} success={eval_summary['success_rate']:.3f} "
                    f"rescued={pairs['rescued']} broken={pairs['broken']} "
                    f"lift={eval_summary['lift_rate']:.3f} "
                    f"res_l2={eval_summary['mean_residual_l2']:.3f} "
                    f"sat={eval_summary['residual_saturation_rate']:.3f}\n"
                )

                current_score = score_tuple(record)
                if best_score is None or current_score > best_score:
                    best_score = current_score
                    best_global_step = t + 1
                    policy.save(checkpoint_dir / "best.pt")
                    print("[BEST] new best checkpoint:", best_global_step)

        policy.save(checkpoint_dir / "final.pt")

        write_csv(args.output_dir / "episodes.csv", episode_rows)
        write_csv(args.output_dir / "training.csv", training_rows)
        write_csv(args.output_dir / "evaluations.csv", evaluation_rows)
        write_csv(args.output_dir / "evaluation_episodes.csv", evaluation_episode_rows)

        # Rank learned periodic checkpoints and materialize top 2-3 candidates.
        learned_records = [r for r in evaluation_rows if int(r["global_step"]) > 0]
        learned_records.sort(key=score_tuple, reverse=True)
        top = learned_records[:candidate_count]
        candidates = []
        for rank, row in enumerate(top, start=1):
            step = int(row["global_step"])
            src = checkpoint_dir / f"step_{step:08d}.pt"
            dst = checkpoint_dir / f"candidate_rank{rank}.pt"
            if src.exists():
                shutil.copy2(src, dst)
            candidates.append(
                {
                    "rank": rank,
                    "global_step": step,
                    "checkpoint": str(dst if dst.exists() else src),
                    "success_count": int(row["success_count"]),
                    "success_rate": float(row["success_rate"]),
                    "rescued": int(row["rescued"]),
                    "broken": int(row["broken"]),
                    "residual_saturation_rate": float(row["residual_saturation_rate"]),
                    "mean_residual_l2": float(row["mean_residual_l2"]),
                }
            )

        with (args.output_dir / "candidate_ranking.json").open("w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)

        technical_passed = bool(
            len(errors) == 0
            and critic_updates > 0
            and actor_updates > 0
            and action_bound_violations == 0
            and best_global_step is not None
        )

        summary = {
            "cube_to_grasp_scale": scale,
            "train_radius_mm": radius_mm,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "max_timesteps": max_timesteps,
            "hypothesis_comparison_step": hypothesis_comparison_step,
            "periodic_eval_num_directions": eval_num_directions,
            "periodic_eval_episodes_per_direction": eval_episodes_per_direction,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "alpha": float(train_env.alpha),
            "reference_periodic_success_rate": float(reference_summary["success_rate"]),
            "initial_policy_success_rate": float(initial_eval["success_rate"]),
            "best_global_step": best_global_step,
            "candidate_checkpoints": candidates,
            "critic_updates": critic_updates,
            "actor_updates": actor_updates,
            "action_bound_violations": action_bound_violations,
            "last_critic_loss": last_critic_loss,
            "last_actor_loss": last_actor_loss,
            "errors": errors,
            "technical_passed": technical_passed,
        }
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 72)
        print("Day17 Training Result")
        print("=" * 72)
        print("technical_passed:", technical_passed)
        print("best_global_step:", best_global_step)
        print("candidate_checkpoints:")
        for item in candidates:
            print(item)
        print("summary:", args.output_dir / "summary.json")

        return 0 if technical_passed else 1

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
