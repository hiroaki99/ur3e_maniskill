#!/usr/bin/env python3
"""
Replay Buffer for UR3e Residual TD3.

保存する遷移:
    state
    residual_action
    next_state
    reward
    not_done

元yaginuma-trackerのReplay Bufferと同じ基本契約を使う。

今回の違い:
- state:  22 -> 44
- action: 2  -> 6
- expert別sampleは行わない
  （現在は単一Reference trajectoryのため）
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 100000,
        seed: int = 0,
    ):

        self.state_dim = int(
            state_dim
        )

        self.action_dim = int(
            action_dim
        )

        self.max_size = int(
            max_size
        )

        if self.state_dim <= 0:
            raise ValueError(
                "state_dim must be > 0"
            )

        if self.action_dim <= 0:
            raise ValueError(
                "action_dim must be > 0"
            )

        if self.max_size <= 0:
            raise ValueError(
                "max_size must be > 0"
            )

        self.state = np.zeros(
            (
                self.max_size,
                self.state_dim,
            ),
            dtype=np.float32,
        )

        self.action = np.zeros(
            (
                self.max_size,
                self.action_dim,
            ),
            dtype=np.float32,
        )

        self.next_state = np.zeros(
            (
                self.max_size,
                self.state_dim,
            ),
            dtype=np.float32,
        )

        self.reward = np.zeros(
            (
                self.max_size,
                1,
            ),
            dtype=np.float32,
        )

        self.not_done = np.zeros(
            (
                self.max_size,
                1,
            ),
            dtype=np.float32,
        )

        self.ptr = 0

        self.size = 0

        self.rng = np.random.default_rng(
            seed
        )

    def __len__(
        self,
    ) -> int:

        return self.size

    def add(
        self,
        state,
        action,
        next_state,
        reward: float,
        done: bool,
    ):

        state = np.asarray(
            state,
            dtype=np.float32,
        ).reshape(-1)

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        ).reshape(-1)

        if state.shape != (
            self.state_dim,
        ):
            raise ValueError(
                "state shape mismatch: "
                f"{state.shape}"
            )

        if next_state.shape != (
            self.state_dim,
        ):
            raise ValueError(
                "next_state shape mismatch: "
                f"{next_state.shape}"
            )

        if action.shape != (
            self.action_dim,
        ):
            raise ValueError(
                "action shape mismatch: "
                f"{action.shape}"
            )

        if not np.all(
            np.isfinite(
                state
            )
        ):
            raise ValueError(
                "state contains NaN/inf"
            )

        if not np.all(
            np.isfinite(
                next_state
            )
        ):
            raise ValueError(
                "next_state contains NaN/inf"
            )

        if not np.all(
            np.isfinite(
                action
            )
        ):
            raise ValueError(
                "action contains NaN/inf"
            )

        if not np.isfinite(
            reward
        ):
            raise ValueError(
                "reward is NaN/inf"
            )

        index = self.ptr

        self.state[
            index
        ] = state

        self.action[
            index
        ] = action

        self.next_state[
            index
        ] = next_state

        self.reward[
            index,
            0,
        ] = float(
            reward
        )

        self.not_done[
            index,
            0,
        ] = (
            0.0
            if bool(done)
            else 1.0
        )

        self.ptr = (
            self.ptr + 1
        ) % self.max_size

        self.size = min(
            self.size + 1,
            self.max_size,
        )

    def sample(
        self,
        batch_size: int,
    ):

        batch_size = int(
            batch_size
        )

        if self.size < batch_size:

            raise ValueError(
                "Replay Bufferに十分な"
                "サンプルがありません: "
                f"size={self.size}, "
                f"batch={batch_size}"
            )

        indices = self.rng.integers(
            low=0,
            high=self.size,
            size=batch_size,
        )

        return (
            self.state[
                indices
            ],
            self.action[
                indices
            ],
            self.next_state[
                indices
            ],
            self.reward[
                indices
            ],
            self.not_done[
                indices
            ],
        )