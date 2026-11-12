#!/usr/bin/env python3
"""Replay buffer for Yaginuma-style TD3 residual learning."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class YaginumaReplayBuffer:
    def __init__(
        self,
        *,
        state_dim: int = 22,
        action_dim: int = 2,
        max_size: int = 1_000_000,
        seed: int = 0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_size = int(max_size)
        self.rng = np.random.default_rng(int(seed))

        self.state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state,
        action,
        next_state,
        reward: float,
        done_bool: float,
    ) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)

        if state.shape != (self.state_dim,):
            raise ValueError(f"state shape {state.shape} != ({self.state_dim},)")
        if action.shape != (self.action_dim,):
            raise ValueError(f"action shape {action.shape} != ({self.action_dim},)")
        if next_state.shape != (self.state_dim,):
            raise ValueError(
                f"next_state shape {next_state.shape} != ({self.state_dim},)"
            )

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr, 0] = float(reward)
        self.not_done[self.ptr, 0] = 1.0 - float(done_bool)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def expert_ids(self) -> np.ndarray:
        if self.size == 0:
            return np.empty((0,), dtype=np.int64)
        return np.argmax(self.state[: self.size, 9:22], axis=1).astype(np.int64)

    def count_for_expert(self, expert_id: int) -> int:
        ids = self.expert_ids()
        return int(np.count_nonzero(ids == int(expert_id)))

    def counts(self) -> Dict[int, int]:
        ids = self.expert_ids()
        return {ep: int(np.count_nonzero(ids == ep)) for ep in range(13)}

    def sample_for_expert(
        self,
        *,
        expert_id: int,
        batch_size: int,
        device,
    ):
        expert_id = int(expert_id)
        batch_size = int(batch_size)

        if not 0 <= expert_id <= 12:
            raise ValueError("expert_id must be in [0,12].")
        if self.size <= 0:
            raise RuntimeError("Replay buffer is empty.")

        ids = self.expert_ids()
        candidate = np.flatnonzero(ids == expert_id)
        if candidate.size == 0:
            raise RuntimeError(f"No replay transitions for expert {expert_id}.")

        # Source-style learning is expert-conditioned.  Sampling with
        # replacement keeps the contract usable during smoke tests; after the
        # original 500 random warm-up steps, each fixed expert normally has
        # many more than batch_size samples.
        indices = self.rng.choice(candidate, size=batch_size, replace=True)

        def tensor(array):
            return torch.as_tensor(array[indices], dtype=torch.float32, device=device)

        return (
            tensor(self.state),
            tensor(self.action),
            tensor(self.next_state),
            tensor(self.reward),
            tensor(self.not_done),
        )
