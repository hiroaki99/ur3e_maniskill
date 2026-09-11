#!/usr/bin/env python3
"""
Day11 TD3 Core Test.

ManiSkillを動かす前に、
純粋なPyTorch TD3について以下を確認する。

- Actor output shape = (6,)
- Actionが[-1,1]
- Replay Buffer sample
- Twin Critic update
- delayed Actor update
- loss finite
- save/load後のActor出力完全一致
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(REPO_ROOT),
)


from rrl import (
    ReplayBuffer,
    TD3,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--updates",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/"
            "day11_td3_core_test.json"
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    state_dim = 44
    action_dim = 6

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    rng = np.random.default_rng(
        args.seed
    )

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=2000,
        seed=args.seed,
    )

    # ------------------------------------------------------
    # Synthetic transitions
    # ------------------------------------------------------

    for _ in range(512):

        state = rng.normal(
            0.0,
            0.5,
            size=state_dim,
        ).astype(
            np.float32
        )

        action = rng.uniform(
            -1.0,
            1.0,
            size=action_dim,
        ).astype(
            np.float32
        )

        next_state = (
            state
            + rng.normal(
                0.0,
                0.02,
                size=state_dim,
            )
        ).astype(
            np.float32
        )

        reward = float(
            rng.normal(
                0.0,
                1.0,
            )
        )

        done = bool(
            rng.random()
            < 0.02
        )

        buffer.add(
            state,
            action,
            next_state,
            reward,
            done,
        )

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        discount=0.99,
        tau=0.05,
        policy_noise=0.05,
        noise_clip=0.10,
        policy_freq=2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        device=device,
        seed=args.seed,
    )

    errors = []

    # ------------------------------------------------------
    # Actor
    # ------------------------------------------------------

    probe_state = np.zeros(
        state_dim,
        dtype=np.float32,
    )

    initial_action = (
        policy.select_action(
            probe_state
        )
    )

    print("=" * 72)
    print("Day 11 TD3 Core Test")
    print("=" * 72)

    print(
        "device:",
        device,
    )

    print(
        "actor action:",
        initial_action.tolist(),
    )

    if initial_action.shape != (
        action_dim,
    ):

        errors.append(
            "Actor output shape mismatch"
        )

    if (
        np.any(
            initial_action < -1.0
        )
        or np.any(
            initial_action > 1.0
        )
    ):

        errors.append(
            "Actor output outside [-1,1]"
        )

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    critic_losses = []
    actor_losses = []

    actor_update_count = 0

    for update in range(
        args.updates
    ):

        metrics = policy.train(
            buffer,
            batch_size=32,
        )

        critic_loss = (
            metrics[
                "critic_loss"
            ]
        )

        if not np.isfinite(
            critic_loss
        ):

            errors.append(
                "Critic loss NaN/inf"
            )

            break

        critic_losses.append(
            critic_loss
        )

        if metrics[
            "actor_updated"
        ]:

            actor_update_count += 1

            actor_loss = metrics[
                "actor_loss"
            ]

            if (
                actor_loss is None
                or not np.isfinite(
                    actor_loss
                )
            ):

                errors.append(
                    "Actor loss NaN/inf"
                )

                break

            actor_losses.append(
                actor_loss
            )

    expected_actor_updates = (
        args.updates
        // 2
    )

    if (
        actor_update_count
        != expected_actor_updates
    ):

        errors.append(
            "Delayed Actor update count "
            "mismatch: "
            f"{actor_update_count} != "
            f"{expected_actor_updates}"
        )

    # ------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------

    checkpoint = (
        args.output.parent
        / "day11_td3_core_test.pt"
    )

    policy.save(
        checkpoint
    )

    action_before_load = (
        policy.select_action(
            probe_state
        )
    )

    loaded_policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        discount=0.99,
        tau=0.05,
        policy_noise=0.05,
        noise_clip=0.10,
        policy_freq=2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        device=device,
        seed=args.seed + 999,
    )

    loaded_policy.load(
        checkpoint
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

    if load_difference > 1e-7:

        errors.append(
            "save/load Actor action mismatch: "
            f"{load_difference}"
        )

    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    report = {
        "device":
            device,

        "state_dim":
            state_dim,

        "action_dim":
            action_dim,

        "buffer_size":
            len(
                buffer
            ),

        "updates":
            args.updates,

        "actor_update_count":
            actor_update_count,

        "expected_actor_updates":
            expected_actor_updates,

        "critic_loss_first":
            (
                critic_losses[0]
                if critic_losses
                else None
            ),

        "critic_loss_last":
            (
                critic_losses[-1]
                if critic_losses
                else None
            ),

        "actor_loss_first":
            (
                actor_losses[0]
                if actor_losses
                else None
            ),

        "actor_loss_last":
            (
                actor_losses[-1]
                if actor_losses
                else None
            ),

        "initial_action":
            initial_action.tolist(),

        "save_load_max_difference":
            load_difference,

        "checkpoint":
            str(
                checkpoint
            ),

        "errors":
            errors,

        "passed":
            len(errors) == 0,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "updates:",
        args.updates,
    )

    print(
        "actor updates:",
        actor_update_count,
    )

    print(
        "critic loss:",
        (
            critic_losses[-1]
            if critic_losses
            else None
        ),
    )

    print(
        "actor loss:",
        (
            actor_losses[-1]
            if actor_losses
            else None
        ),
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
        len(errors) == 0,
    )

    print(
        "report:",
        args.output,
    )

    return (
        0
        if len(errors) == 0
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )