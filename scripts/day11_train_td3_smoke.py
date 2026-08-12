#!/usr/bin/env python3
"""
Day11 TD3 Smoke Training.

目的:
- ResidualPickLiftEnv + Reward + TD3を接続
- Replay Bufferへ実遷移を保存
- Critic / Actorを更新
- NaN/Infがないことを確認
- Action [-1,1]を確認
- Checkpointを保存
- 保存後の再読込を確認

性能向上の評価はDay11の目的ではない。
Day12でReferenceへ補正可能な誤差を導入してから行う。
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
    ReplayBuffer,
    ResidualPickLiftEnv,
    TD3,
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
        "--max-timesteps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--start-timesteps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/"
            "day11_td3_smoke_seed0"
        ),
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> str:

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
            "--device cuda was requested "
            "but CUDA is unavailable"
        )

    return requested


def write_csv(
    path: Path,
    rows: list[dict],
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


def main():

    args = parse_args()

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    day9 = config.get(
        "day9",
        {},
    )

    day11 = config[
        "day11"
    ]

    seed = int(
        day11.get(
            "seed",
            0,
        )
    )

    max_timesteps = int(
        args.max_timesteps
        if args.max_timesteps
        is not None
        else day11[
            "max_timesteps"
        ]
    )

    start_timesteps = int(
        args.start_timesteps
        if args.start_timesteps
        is not None
        else day11[
            "start_timesteps"
        ]
    )

    batch_size = int(
        day11[
            "batch_size"
        ]
    )

    replay_size = int(
        day11[
            "replay_size"
        ]
    )

    expl_noise = float(
        day11[
            "exploration_noise"
        ]
    )

    checkpoint_interval = int(
        day11[
            "checkpoint_interval"
        ]
    )

    log_interval = int(
        day11[
            "log_interval"
        ]
    )

    device = resolve_device(
        args.device
    )

    rng = np.random.default_rng(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
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
    # Environment
    # ======================================================

    import envs.ur3e_pick_lift  # noqa

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
        sim_backend=args.sim_backend,
        max_episode_steps=int(
            day9.get(
                "base_env_max_episode_steps",
                1000,
            )
        ),
    )

    env = ResidualPickLiftEnv(
        env=base_env,
        trajectory_path=(
            args.trajectory
        ),
        config_path=CONFIG_PATH,
    )

    try:

        env.action_space.seed(
            seed
        )

        (
            state,
            reset_info,
        ) = env.reset(
            seed=seed
        )

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        state_dim = int(
            env.observation_space.shape[
                0
            ]
        )

        action_dim = int(
            env.action_space.shape[
                0
            ]
        )

        max_action = float(
            env.action_space.high[0]
        )

        # ==================================================
        # TD3 / Buffer
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

            seed=seed,
        )

        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=replay_size,
            seed=seed,
        )

        # ==================================================
        # Logging
        # ==================================================

        errors = []

        episodes = []

        training_rows = []

        episode_return = 0.0
        episode_timesteps = 0
        episode_number = 0

        success_count = 0

        critic_update_count = 0
        actor_update_count = 0

        last_critic_loss = None
        last_actor_loss = None

        max_abs_action = 0.0

        action_bound_violations = 0

        print("=" * 72)
        print("Day 11 TD3 Smoke Training")
        print("=" * 72)

        print(
            "device:",
            device,
        )

        print(
            "state dim:",
            state_dim,
        )

        print(
            "action dim:",
            action_dim,
        )

        print(
            "start timesteps:",
            start_timesteps,
        )

        print(
            "max timesteps:",
            max_timesteps,
        )

        print(
            "discount:",
            policy.discount,
        )

        print(
            "alpha:",
            env.alpha,
        )

        # ==================================================
        # Training loop
        # ==================================================

        for t in range(
            max_timesteps
        ):

            # ----------------------------------------------
            # Action selection
            # ----------------------------------------------

            if t < start_timesteps:

                action = rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=action_dim,
                ).astype(
                    np.float32
                )

            else:

                action = (
                    policy.select_action(
                        state
                    )
                )

                exploration = rng.normal(
                    loc=0.0,
                    scale=expl_noise,
                    size=action_dim,
                )

                action = (
                    action
                    + exploration
                )

                action = np.clip(
                    action,
                    -max_action,
                    max_action,
                ).astype(
                    np.float32
                )

            if not np.all(
                np.isfinite(
                    action
                )
            ):

                errors.append(
                    f"action NaN/inf at step {t}"
                )

                break

            max_abs_action = max(
                max_abs_action,
                float(
                    np.max(
                        np.abs(
                            action
                        )
                    )
                ),
            )

            if (
                np.any(
                    action < -max_action
                )
                or np.any(
                    action > max_action
                )
            ):

                action_bound_violations += 1

            # ----------------------------------------------
            # Environment
            # ----------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            next_state = np.asarray(
                next_state,
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
                    next_state
                )
            ):

                errors.append(
                    f"observation NaN/inf at step {t}"
                )

                break

            if not np.isfinite(
                reward
            ):

                errors.append(
                    f"reward NaN/inf at step {t}"
                )

                break

            # ----------------------------------------------
            # Replay Buffer
            # ----------------------------------------------

            replay_buffer.add(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
            )

            state = next_state

            episode_return += reward

            episode_timesteps += 1

            # ----------------------------------------------
            # TD3 update
            # ----------------------------------------------

            if (
                t >= start_timesteps
                and len(
                    replay_buffer
                ) >= batch_size
            ):

                metrics = policy.train(
                    replay_buffer,
                    batch_size=batch_size,
                )

                critic_update_count += 1

                last_critic_loss = float(
                    metrics[
                        "critic_loss"
                    ]
                )

                if not np.isfinite(
                    last_critic_loss
                ):

                    errors.append(
                        "critic loss NaN/inf "
                        f"at step {t}"
                    )

                    break

                if metrics[
                    "actor_updated"
                ]:

                    actor_update_count += 1

                    last_actor_loss = float(
                        metrics[
                            "actor_loss"
                        ]
                    )

                    if not np.isfinite(
                        last_actor_loss
                    ):

                        errors.append(
                            "actor loss NaN/inf "
                            f"at step {t}"
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

                if success:
                    success_count += 1

                episodes.append(
                    {
                        "episode":
                            episode_number,

                        "global_step":
                            t + 1,

                        "episode_steps":
                            episode_timesteps,

                        "return":
                            episode_return,

                        "success":
                            success,

                        "cube_lift_m":
                            float(
                                info.get(
                                    "cube_lift_m",
                                    0.0,
                                )
                            ),

                        "terminal_reason":
                            info.get(
                                "terminal_reason"
                            ),

                        "buffer_size":
                            len(
                                replay_buffer
                            ),
                    }
                )

                print(
                    f"episode="
                    f"{episode_number:3d} "
                    f"step={t + 1:5d} "
                    f"return="
                    f"{episode_return:+.3f} "
                    f"success={success} "
                    f"lift="
                    f"{info.get('cube_lift_m', 0.0):.4f}"
                )

                episode_number += 1

                (
                    state,
                    reset_info,
                ) = env.reset(
                    seed=(
                        seed
                        + episode_number
                    )
                )

                state = np.asarray(
                    state,
                    dtype=np.float32,
                )

                episode_return = 0.0

                episode_timesteps = 0

            # ----------------------------------------------
            # Console log
            # ----------------------------------------------

            if (
                (t + 1)
                % log_interval
                == 0
            ):

                print(
                    f"step={t + 1:5d} "
                    f"buffer="
                    f"{len(replay_buffer):5d} "
                    f"critic="
                    f"{last_critic_loss} "
                    f"actor="
                    f"{last_actor_loss} "
                    f"max|a|="
                    f"{max_abs_action:.3f}"
                )

            # ----------------------------------------------
            # Checkpoint
            # ----------------------------------------------

            if (
                (t + 1)
                % checkpoint_interval
                == 0
            ):

                checkpoint_path = (
                    checkpoint_dir
                    / (
                        f"step_"
                        f"{t + 1:07d}.pt"
                    )
                )

                policy.save(
                    checkpoint_path
                )

                print(
                    "checkpoint:",
                    checkpoint_path,
                )

        # ==================================================
        # Final save
        # ==================================================

        final_checkpoint = (
            checkpoint_dir
            / "final.pt"
        )

        policy.save(
            final_checkpoint
        )

        # ==================================================
        # Save / Load integration check
        # ==================================================

        probe_state = (
            state.copy()
        )

        action_before_load = (
            policy.select_action(
                probe_state
            )
        )

        loaded_policy = TD3(
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
            seed=seed + 999,
        )

        loaded_policy.load(
            final_checkpoint
        )

        action_after_load = (
            loaded_policy.select_action(
                probe_state
            )
        )

        load_difference = float(
            np.max(
                np.abs(
                    action_before_load
                    - action_after_load
                )
            )
        )

        if (
            load_difference > 1e-7
        ):

            errors.append(
                "save/load action mismatch: "
                f"{load_difference}"
            )

        if critic_update_count <= 0:

            errors.append(
                "Critic was never updated"
            )

        if actor_update_count <= 0:

            errors.append(
                "Actor was never updated"
            )

        if action_bound_violations != 0:

            errors.append(
                "Residual action exceeded bounds"
            )

        passed = (
            len(errors) == 0
        )

        # ==================================================
        # CSV
        # ==================================================

        write_csv(
            args.output_dir
            / "episodes.csv",
            episodes,
        )

        write_csv(
            args.output_dir
            / "training.csv",
            training_rows,
        )

        # ==================================================
        # Summary
        # ==================================================

        summary = {
            "trajectory":
                str(
                    args.trajectory
                ),

            "device":
                device,

            "seed":
                seed,

            "state_dim":
                state_dim,

            "action_dim":
                action_dim,

            "alpha":
                float(
                    env.alpha
                ),

            "max_timesteps":
                max_timesteps,

            "start_timesteps":
                start_timesteps,

            "batch_size":
                batch_size,

            "replay_size":
                replay_size,

            "discount":
                float(
                    day11[
                        "discount"
                    ]
                ),

            "tau":
                float(
                    day11[
                        "tau"
                    ]
                ),

            "exploration_noise":
                expl_noise,

            "policy_noise":
                float(
                    day11[
                        "policy_noise"
                    ]
                ),

            "noise_clip":
                float(
                    day11[
                        "noise_clip"
                    ]
                ),

            "policy_freq":
                int(
                    day11[
                        "policy_freq"
                    ]
                ),

            "critic_update_count":
                critic_update_count,

            "actor_update_count":
                actor_update_count,

            "last_critic_loss":
                last_critic_loss,

            "last_actor_loss":
                last_actor_loss,

            "episodes_completed":
                len(
                    episodes
                ),

            "success_count":
                success_count,

            "max_abs_residual_action":
                max_abs_action,

            "action_bound_violations":
                action_bound_violations,

            "buffer_size":
                len(
                    replay_buffer
                ),

            "final_checkpoint":
                str(
                    final_checkpoint
                ),

            "save_load_max_difference":
                load_difference,

            "migration_from_yaginuma": {
                "source_state_dim":
                    22,

                "ported_state_dim":
                    44,

                "source_action_dim":
                    2,

                "ported_action_dim":
                    6,

                "source_discount":
                    0.0,

                "ported_discount":
                    float(
                        day11[
                            "discount"
                        ]
                    ),

                "source_actor":
                    "13 expert heads",

                "ported_actor":
                    (
                        "single head; "
                        "phase encoded in observation"
                    ),

                "source_action_semantics":
                    (
                        "expert endpoint "
                        "[dx,dz]"
                    ),

                "ported_action_semantics":
                    (
                        "normalized 6-DoF "
                        "joint residual"
                    ),
            },

            "errors":
                errors,

            "passed":
                passed,
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

        # ==================================================
        # Final print
        # ==================================================

        print()
        print("=" * 72)
        print("Day 11 Result")
        print("=" * 72)

        print(
            "timesteps:",
            max_timesteps,
        )

        print(
            "buffer size:",
            len(
                replay_buffer
            ),
        )

        print(
            "critic updates:",
            critic_update_count,
        )

        print(
            "actor updates:",
            actor_update_count,
        )

        print(
            "episodes:",
            len(
                episodes
            ),
        )

        print(
            "successes:",
            success_count,
        )

        print(
            "last critic loss:",
            last_critic_loss,
        )

        print(
            "last actor loss:",
            last_actor_loss,
        )

        print(
            "max |residual action|:",
            max_abs_action,
        )

        print(
            "bound violations:",
            action_bound_violations,
        )

        print(
            "save/load difference:",
            load_difference,
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
            if passed
            else 1
        )

    finally:

        env.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )