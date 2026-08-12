#!/usr/bin/env python3
"""
Day13 TD3 Main Training.

Training:
    20 mm fixed radius
    random XY direction every episode

Periodic evaluation:
    fixed 8 directions
    deterministic Actor
    no exploration noise

Outputs:
    checkpoints
    episode log
    TD3 loss log
    evaluation log
    best checkpoint
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


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT),
)


from rrl import (
    CubePositionPerturbationWrapper,
    ReplayBuffer,
    ResidualPickLiftEnv,
    TD3,
    evaluate_residual_policy,
)


CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "ur3e_pick_lift.yaml"
)

DEFAULT_TRAJECTORY = (
    REPO_ROOT
    / "trajectories"
    / "day6_pick_lift_reference_v2.json"
)


# ==========================================================
# Utility
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
    )

    parser.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=[
            "physx_cpu",
            "physx_cuda",
        ],
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/"
            "day13_td3_seed1300"
        ),
    )

    return parser.parse_args()


def resolve_device(
    requested,
):

    if requested == "auto":

        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    return requested


def write_csv(
    path: Path,
    rows,
):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def make_env(
    *,
    config,
    trajectory,
    sim_backend,
    radius_mm,
    seed,
):

    import envs.ur3e_pick_lift  # noqa

    day9 = config.get(
        "day9",
        {},
    )

    base_env = gym.make(
        "UR3ePickLift-v0",
        robot_uids=config[
            "robot"
        ][
            "uid"
        ],
        num_envs=1,
        obs_mode="state",
        control_mode=config[
            "project"
        ][
            "control_mode"
        ],
        sim_backend=sim_backend,
        max_episode_steps=int(
            day9.get(
                "base_env_max_episode_steps",
                1000,
            )
        ),
    )

    radius_m = (
        float(
            radius_mm
        )
        / 1000.0
    )

    perturb_env = (
        CubePositionPerturbationWrapper(
            base_env,
            radius_min_m=radius_m,
            radius_max_m=radius_m,
            direction_mode="random_angle",
            fixed_angle_rad=None,
            seed=seed,
        )
    )

    env = ResidualPickLiftEnv(
        env=perturb_env,
        trajectory_path=trajectory,
        config_path=CONFIG_PATH,
    )

    return (
        env,
        perturb_env,
    )


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    day11 = config[
        "day11"
    ]

    day12 = config[
        "day12"
    ]

    day13 = config[
        "day13"
    ]

    train_seed = int(
        day13[
            "train_seed"
        ]
    )

    eval_seed = int(
        day13[
            "eval_seed"
        ]
    )

    radius_mm = float(
        day13.get(
            "train_radius_mm",
            day12[
                "selected_radius_mm"
            ],
        )
    )

    max_timesteps = int(
        args.max_timesteps
        if args.max_timesteps
        is not None
        else day13[
            "max_timesteps"
        ]
    )

    start_timesteps = int(
        day13[
            "start_timesteps"
        ]
    )

    eval_interval = int(
        args.eval_interval
        if args.eval_interval
        is not None
        else day13[
            "eval_interval"
        ]
    )

    checkpoint_interval = int(
        day13[
            "checkpoint_interval"
        ]
    )

    log_interval = int(
        day13[
            "log_interval"
        ]
    )

    max_steps_per_episode = int(
        day13[
            "max_steps_per_episode"
        ]
    )

    eval_episodes_per_direction = int(
        day13[
            "eval_episodes_per_direction"
        ]
    )

    eval_num_directions = int(
        day13[
            "eval_num_directions"
        ]
    )

    lift_threshold_m = float(
        day13[
            "lift_threshold_m"
        ]
    )

    saturation_threshold = float(
        day13[
            "residual_saturation_threshold"
        ]
    )

    reference_baseline = float(
        day13[
            "reference_baseline_success_rate"
        ]
    )

    device = resolve_device(
        args.device
    )

    rng = np.random.default_rng(
        train_seed
    )

    np.random.seed(
        train_seed
    )

    torch.manual_seed(
        train_seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            train_seed
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        args.output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # Training env
    # ======================================================

    train_env, train_perturb = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=train_seed,
    )

    # ======================================================
    # Separate evaluation env
    # ======================================================

    eval_env, eval_perturb = make_env(
        config=config,
        trajectory=args.trajectory,
        sim_backend=args.sim_backend,
        radius_mm=radius_mm,
        seed=eval_seed,
    )

    try:

        observation, reset_info = (
            train_env.reset(
                seed=train_seed
            )
        )

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        state_dim = int(
            train_env.observation_space.shape[
                0
            ]
        )

        action_dim = int(
            train_env.action_space.shape[
                0
            ]
        )

        max_action = float(
            train_env.action_space.high[
                0
            ]
        )

        # ==================================================
        # TD3
        # ==================================================

        policy = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,

            actor_hidden_dim=int(
                day11[
                    "actor_hidden_dim"
                ]
            ),

            critic_hidden_dim=int(
                day11[
                    "critic_hidden_dim"
                ]
            ),

            discount=float(
                day11[
                    "discount"
                ]
            ),

            tau=float(
                day11[
                    "tau"
                ]
            ),

            policy_noise=float(
                day11[
                    "policy_noise"
                ]
            ),

            noise_clip=float(
                day11[
                    "noise_clip"
                ]
            ),

            policy_freq=int(
                day11[
                    "policy_freq"
                ]
            ),

            actor_lr=float(
                day11[
                    "actor_lr"
                ]
            ),

            critic_lr=float(
                day11[
                    "critic_lr"
                ]
            ),

            weight_decay=float(
                day11[
                    "weight_decay"
                ]
            ),

            device=device,
            seed=train_seed,
        )

        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=int(
                day11[
                    "replay_size"
                ]
            ),
            seed=train_seed,
        )

        exploration_noise = float(
            day11[
                "exploration_noise"
            ]
        )

        batch_size = int(
            day11[
                "batch_size"
            ]
        )

        # ==================================================
        # Evaluation directions
        # ==================================================

        angles_deg = (
            np.linspace(
                0.0,
                360.0,
                eval_num_directions,
                endpoint=False,
            )
            .tolist()
        )

        # ==================================================
        # Logs
        # ==================================================

        episode_rows = []
        training_rows = []
        evaluation_rows = []
        evaluation_episode_rows = []

        errors = []

        # ==================================================
        # Initial policy evaluation
        # ==================================================

        initial_eval, initial_rows = (
            evaluate_residual_policy(
                env=eval_env,
                perturbation_wrapper=(
                    eval_perturb
                ),
                policy=policy,
                angles_deg=angles_deg,
                episodes_per_direction=(
                    eval_episodes_per_direction
                ),
                seed=eval_seed,
                max_steps=(
                    max_steps_per_episode
                ),
                lift_threshold_m=(
                    lift_threshold_m
                ),
                saturation_threshold=(
                    saturation_threshold
                ),
            )
        )

        initial_eval[
            "global_step"
        ] = 0

        evaluation_rows.append(
            initial_eval.copy()
        )

        for row in initial_rows:

            row = dict(
                row
            )

            row[
                "global_step"
            ] = 0

            evaluation_episode_rows.append(
                row
            )

        print("=" * 72)
        print("Day 13 TD3 Main Training")
        print("=" * 72)

        print(
            "device:",
            device,
        )

        print(
            "training radius:",
            radius_mm,
            "mm",
        )

        print(
            "reference baseline:",
            reference_baseline,
        )

        print(
            "initial TD3 success:",
            initial_eval[
                "success_rate"
            ],
        )

        print(
            "initial grasp:",
            initial_eval[
                "grasp_rate"
            ],
        )

        print(
            "initial lift:",
            initial_eval[
                "lift_rate"
            ],
        )

        # Bestは「学習後評価」のみ対象
        best_success_rate = -1.0
        best_lift_rate = -1.0
        best_residual_l2 = float(
            "inf"
        )
        best_global_step = None

        # ==================================================
        # Episode accumulators
        # ==================================================

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
            reset_info.get(
                "cube_offset_m",
                [0.0, 0.0, 0.0],
            ),
            dtype=np.float64,
        )

        critic_updates = 0
        actor_updates = 0

        last_critic_loss = None
        last_actor_loss = None

        action_bound_violations = 0

        # ==================================================
        # Training
        # ==================================================

        for t in range(
            max_timesteps
        ):

            # ----------------------------------------------
            # Residual action
            # ----------------------------------------------

            if t < start_timesteps:

                residual_action = (
                    rng.uniform(
                        -1.0,
                        1.0,
                        size=action_dim,
                    )
                    .astype(
                        np.float32
                    )
                )

            else:

                residual_action = (
                    policy.select_action(
                        observation
                    )
                )

                noise = rng.normal(
                    0.0,
                    exploration_noise,
                    size=action_dim,
                )

                residual_action = (
                    residual_action
                    + noise
                )

                residual_action = np.clip(
                    residual_action,
                    -max_action,
                    max_action,
                ).astype(
                    np.float32
                )

            if not np.all(
                np.isfinite(
                    residual_action
                )
            ):

                errors.append(
                    f"action NaN/inf at {t}"
                )
                break

            if (
                np.any(
                    residual_action
                    < -max_action
                )
                or
                np.any(
                    residual_action
                    > max_action
                )
            ):

                action_bound_violations += 1

            # Episode residual stats
            episode_residual_l2_sum += float(
                np.linalg.norm(
                    residual_action
                )
            )

            episode_residual_abs_sum += float(
                np.sum(
                    np.abs(
                        residual_action
                    )
                )
            )

            episode_residual_elements += (
                residual_action.size
            )

            episode_saturation_count += int(
                np.count_nonzero(
                    np.abs(
                        residual_action
                    )
                    >= saturation_threshold
                )
            )

            # ----------------------------------------------
            # Step
            # ----------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = train_env.step(
                residual_action
            )

            next_observation = np.asarray(
                next_observation,
                dtype=np.float32,
            )

            reward = float(
                reward
            )

            done = bool(
                terminated
                or truncated
            )

            if not np.all(
                np.isfinite(
                    next_observation
                )
            ):

                errors.append(
                    f"observation NaN/inf at {t}"
                )
                break

            if not np.isfinite(
                reward
            ):

                errors.append(
                    f"reward NaN/inf at {t}"
                )
                break

            replay_buffer.add(
                state=observation,
                action=residual_action,
                next_state=next_observation,
                reward=reward,
                done=done,
            )

            observation = (
                next_observation
            )

            episode_return += reward
            episode_steps += 1

            both_contact = (
                bool(
                    info.get(
                        "left_contact",
                        False,
                    )
                )
                and
                bool(
                    info.get(
                        "right_contact",
                        False,
                    )
                )
            )

            episode_grasp_detected = (
                episode_grasp_detected
                or both_contact
            )

            episode_max_lift = max(
                episode_max_lift,
                float(
                    info.get(
                        "cube_lift_m",
                        0.0,
                    )
                ),
            )

            # ----------------------------------------------
            # TD3 update
            # ----------------------------------------------

            if (
                t >= start_timesteps
                and
                len(
                    replay_buffer
                ) >= batch_size
            ):

                metrics = policy.train(
                    replay_buffer,
                    batch_size=batch_size,
                )

                critic_updates += 1

                last_critic_loss = float(
                    metrics[
                        "critic_loss"
                    ]
                )

                if not np.isfinite(
                    last_critic_loss
                ):

                    errors.append(
                        f"critic NaN/inf at {t}"
                    )
                    break

                if metrics[
                    "actor_updated"
                ]:

                    actor_updates += 1

                    last_actor_loss = float(
                        metrics[
                            "actor_loss"
                        ]
                    )

                    if not np.isfinite(
                        last_actor_loss
                    ):

                        errors.append(
                            f"actor NaN/inf at {t}"
                        )
                        break

                training_rows.append(
                    {
                        "global_step":
                            t + 1,

                        "critic_loss":
                            metrics[
                                "critic_loss"
                            ],

                        "actor_updated":
                            metrics[
                                "actor_updated"
                            ],

                        "actor_loss":
                            (
                                metrics[
                                    "actor_loss"
                                ]
                                if metrics[
                                    "actor_loss"
                                ]
                                is not None
                                else ""
                            ),

                        "target_q_mean":
                            metrics[
                                "target_q_mean"
                            ],

                        "current_q1_mean":
                            metrics[
                                "current_q1_mean"
                            ],

                        "current_q2_mean":
                            metrics[
                                "current_q2_mean"
                            ],
                    }
                )

            # ----------------------------------------------
            # Episode end
            # ----------------------------------------------

            if done:

                success = bool(
                    info.get(
                        "success",
                        False,
                    )
                )

                lift_reached = (
                    episode_max_lift
                    >= lift_threshold_m
                )

                mean_residual_l2 = (
                    episode_residual_l2_sum
                    / max(
                        episode_steps,
                        1,
                    )
                )

                mean_abs_residual = (
                    episode_residual_abs_sum
                    / max(
                        episode_residual_elements,
                        1,
                    )
                )

                saturation_rate = (
                    episode_saturation_count
                    / max(
                        episode_residual_elements,
                        1,
                    )
                )

                episode_rows.append(
                    {
                        "episode":
                            episode_number,

                        "global_step":
                            t + 1,

                        "episode_steps":
                            episode_steps,

                        "offset_x_mm":
                            float(
                                current_offset[
                                    0
                                ]
                                * 1000.0
                            ),

                        "offset_y_mm":
                            float(
                                current_offset[
                                    1
                                ]
                                * 1000.0
                            ),

                        "return":
                            float(
                                episode_return
                            ),

                        "success":
                            success,

                        "grasp_detected":
                            bool(
                                episode_grasp_detected
                            ),

                        "lift_reached":
                            bool(
                                lift_reached
                            ),

                        "max_cube_lift_m":
                            float(
                                episode_max_lift
                            ),

                        "mean_residual_l2":
                            float(
                                mean_residual_l2
                            ),

                        "mean_abs_residual":
                            float(
                                mean_abs_residual
                            ),

                        "residual_saturation_rate":
                            float(
                                saturation_rate
                            ),

                        "terminal_reason":
                            info.get(
                                "terminal_reason"
                            ),
                    }
                )

                episode_number += 1

                observation, reset_info = (
                    train_env.reset(
                        seed=(
                            train_seed
                            + episode_number
                        )
                    )
                )

                observation = np.asarray(
                    observation,
                    dtype=np.float32,
                )

                current_offset = np.asarray(
                    reset_info.get(
                        "cube_offset_m",
                        [0.0, 0.0, 0.0],
                    ),
                    dtype=np.float64,
                )

                episode_return = 0.0
                episode_steps = 0

                episode_grasp_detected = False
                episode_max_lift = 0.0

                episode_residual_l2_sum = 0.0
                episode_residual_abs_sum = 0.0
                episode_residual_elements = 0
                episode_saturation_count = 0

            # ----------------------------------------------
            # Console log
            # ----------------------------------------------

            if (
                (t + 1)
                % log_interval
                == 0
            ):

                print(
                    f"step={t + 1:6d} "
                    f"episodes={episode_number:4d} "
                    f"buffer={len(replay_buffer):6d} "
                    f"critic={last_critic_loss} "
                    f"actor={last_actor_loss}"
                )

            # ----------------------------------------------
            # Regular checkpoint
            # ----------------------------------------------

            if (
                (t + 1)
                % checkpoint_interval
                == 0
            ):

                path = (
                    checkpoint_dir
                    / (
                        f"step_"
                        f"{t + 1:08d}.pt"
                    )
                )

                policy.save(
                    path
                )

            # ----------------------------------------------
            # Deterministic evaluation
            # ----------------------------------------------

            if (
                (t + 1)
                % eval_interval
                == 0
            ):

                eval_summary, eval_rows = (
                    evaluate_residual_policy(
                        env=eval_env,
                        perturbation_wrapper=(
                            eval_perturb
                        ),
                        policy=policy,
                        angles_deg=angles_deg,
                        episodes_per_direction=(
                            eval_episodes_per_direction
                        ),
                        seed=eval_seed,
                        max_steps=(
                            max_steps_per_episode
                        ),
                        lift_threshold_m=(
                            lift_threshold_m
                        ),
                        saturation_threshold=(
                            saturation_threshold
                        ),
                    )
                )

                eval_summary[
                    "global_step"
                ] = t + 1

                evaluation_rows.append(
                    eval_summary.copy()
                )

                for row in eval_rows:

                    row = dict(
                        row
                    )

                    row[
                        "global_step"
                    ] = t + 1

                    evaluation_episode_rows.append(
                        row
                    )

                print()
                print(
                    "[EVAL] "
                    f"step={t + 1} "
                    f"success="
                    f"{eval_summary['success_rate']:.3f} "
                    f"grasp="
                    f"{eval_summary['grasp_rate']:.3f} "
                    f"lift="
                    f"{eval_summary['lift_rate']:.3f} "
                    f"res_l2="
                    f"{eval_summary['mean_residual_l2']:.3f} "
                    f"sat="
                    f"{eval_summary['residual_saturation_rate']:.3f}"
                )
                print()

                # ------------------------------------------
                # Best checkpoint
                #
                # Primary:
                # success rate
                #
                # tie:
                # lift rate
                #
                # tie:
                # lower residual
                # ------------------------------------------

                score = (
                    eval_summary[
                        "success_rate"
                    ],
                    eval_summary[
                        "lift_rate"
                    ],
                    -eval_summary[
                        "mean_residual_l2"
                    ],
                )

                best_score = (
                    best_success_rate,
                    best_lift_rate,
                    -best_residual_l2,
                )

                if score > best_score:

                    best_success_rate = float(
                        eval_summary[
                            "success_rate"
                        ]
                    )

                    best_lift_rate = float(
                        eval_summary[
                            "lift_rate"
                        ]
                    )

                    best_residual_l2 = float(
                        eval_summary[
                            "mean_residual_l2"
                        ]
                    )

                    best_global_step = (
                        t + 1
                    )

                    policy.save(
                        checkpoint_dir
                        / "best.pt"
                    )

                    print(
                        "[BEST] new best checkpoint:",
                        best_global_step,
                    )

        # ==================================================
        # Final checkpoint
        # ==================================================

        policy.save(
            checkpoint_dir
            / "final.pt"
        )

        # ==================================================
        # Persist logs
        # ==================================================

        write_csv(
            args.output_dir
            / "episodes.csv",
            episode_rows,
        )

        write_csv(
            args.output_dir
            / "training.csv",
            training_rows,
        )

        write_csv(
            args.output_dir
            / "evaluations.csv",
            evaluation_rows,
        )

        write_csv(
            args.output_dir
            / "evaluation_episodes.csv",
            evaluation_episode_rows,
        )

        # ==================================================
        # Day13 criterion
        # ==================================================

        initial_success = float(
            initial_eval[
                "success_rate"
            ]
        )

        trend_improvement = (
            best_success_rate
            - initial_success
        )

        baseline_improvement = (
            best_success_rate
            - reference_baseline
        )

        technical_passed = bool(
            len(errors) == 0
            and critic_updates > 0
            and actor_updates > 0
            and action_bound_violations == 0
            and best_global_step is not None
        )

        trend_passed = bool(
            best_success_rate
            > initial_success
        )

        passed = bool(
            technical_passed
            and trend_passed
        )

        summary = {
            "train_radius_mm":
                radius_mm,

            "train_seed":
                train_seed,

            "eval_seed":
                eval_seed,

            "max_timesteps":
                max_timesteps,

            "state_dim":
                state_dim,

            "action_dim":
                action_dim,

            "alpha":
                float(
                    train_env.alpha
                ),

            "episodes_completed":
                episode_number,

            "buffer_size":
                len(
                    replay_buffer
                ),

            "critic_updates":
                critic_updates,

            "actor_updates":
                actor_updates,

            "reference_baseline_success_rate":
                reference_baseline,

            "initial_policy_success_rate":
                initial_success,

            "best_success_rate":
                best_success_rate,

            "best_lift_rate":
                best_lift_rate,

            "best_residual_l2":
                best_residual_l2,

            "best_global_step":
                best_global_step,

            "trend_improvement":
                float(
                    trend_improvement
                ),

            "baseline_improvement":
                float(
                    baseline_improvement
                ),

            "action_bound_violations":
                action_bound_violations,

            "last_critic_loss":
                last_critic_loss,

            "last_actor_loss":
                last_actor_loss,

            "technical_passed":
                technical_passed,

            "trend_passed":
                trend_passed,

            "errors":
                errors,

            "passed":
                passed,

            "migration_from_yaginuma": {
                "source_residual":
                    "trajectory endpoint [dx,dz]",

                "ported_residual":
                    "6-DoF normalized joint residual",

                "source_state_dim":
                    22,

                "ported_state_dim":
                    44,

                "training_perturbation":
                    "20 mm random XY cube offset",

                "evaluation":
                    "fixed 8 XY directions",
            },
        }

        summary_path = (
            args.output_dir
            / "summary.json"
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print("=" * 72)
        print("Day 13 Result")
        print("=" * 72)

        print(
            "reference baseline:",
            reference_baseline,
        )

        print(
            "initial policy success:",
            initial_success,
        )

        print(
            "best success:",
            best_success_rate,
        )

        print(
            "best step:",
            best_global_step,
        )

        print(
            "best lift rate:",
            best_lift_rate,
        )

        print(
            "best residual L2:",
            best_residual_l2,
        )

        print(
            "baseline improvement:",
            baseline_improvement,
        )

        print(
            "trend improvement:",
            trend_improvement,
        )

        print(
            "errors:",
            errors,
        )

        print(
            "PASSED:",
            passed,
        )

        print(
            "summary:",
            summary_path,
        )

        return (
            0
            if technical_passed
            else 1
        )

    finally:

        train_env.close()
        eval_env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )