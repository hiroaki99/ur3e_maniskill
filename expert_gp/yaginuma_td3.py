#!/usr/bin/env python3
"""Source-faithful TD3 core for the Yaginuma-style multi-expert RRL path.

Contract reproduced from REPRODUCTION_SPECIFICATION.md:
- state: pose9 + onehot13 = 22 dims
- action: endpoint residual [dx, dz] = 2 dims
- action range: each component in [-0.1, 0.1]
- Actor: shared Linear(22,32) + ReLU, then 13 expert-specific heads
- Critic: twin Q networks, (22+2)->128->128->1
- SmoothL1 critic loss, AdamW optimizers
- delayed actor update, target smoothing, soft target update
- default discount = 0.0

This module intentionally does not import ManiSkill.  It can be unit-tested on CPU.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


STATE_DIM = 22
POSE_DIM = 9
ONEHOT_DIM = 13
ACTION_DIM = 2
NUM_HEADS = 13
MAX_ACTION = 0.1


class MultiHeadActor(nn.Module):
    """Yaginuma TD3 actor with one shared layer and 13 expert heads."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        max_action: float = MAX_ACTION,
        hidden_dim: int = 32,
        num_heads: int = NUM_HEADS,
        pose_dim: int = POSE_DIM,
    ) -> None:
        super().__init__()

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.pose_dim = int(pose_dim)

        if self.state_dim != self.pose_dim + self.num_heads:
            raise ValueError(
                "state_dim must equal pose_dim + num_heads: "
                f"{self.state_dim} != {self.pose_dim}+{self.num_heads}"
            )

        self.shared = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.head_hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(self.num_heads)
            ]
        )
        self.head_output = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.action_dim) for _ in range(self.num_heads)]
        )

    def head_index(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape (batch,{self.state_dim}), got {tuple(state.shape)}"
            )
        onehot = state[:, self.pose_dim : self.pose_dim + self.num_heads]
        return torch.argmax(onehot, dim=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        shared = self.shared(state)
        indices = self.head_index(state)
        out = torch.empty(
            (state.shape[0], self.action_dim),
            dtype=state.dtype,
            device=state.device,
        )

        # Route each row to its selected expert head.  This is equivalent to
        # selecting head argmax(state[:, 9:]) in the source contract.
        for head in range(self.num_heads):
            mask = indices == head
            if torch.any(mask):
                hidden = self.head_hidden[head](shared[mask])
                out[mask] = self.head_output[head](hidden)

        return self.max_action * torch.tanh(out)


class TwinCritic(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        input_dim = int(state_dim) + int(action_dim)

        def make_q() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1_net = make_q()
        self.q2_net = make_q()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa), self.q2_net(sa)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa)


class YaginumaTD3:
    """TD3 implementation for the 22D/2D multi-head RRL contract."""

    def __init__(
        self,
        *,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        max_action: float = MAX_ACTION,
        actor_hidden_dim: int = 32,
        critic_hidden_dim: int = 128,
        discount: float = 0.0,
        tau: float = 0.05,
        policy_noise: float = 0.005,
        noise_clip: float = 0.01,
        policy_freq: int = 2,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_freq = int(policy_freq)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)

        if self.state_dim != STATE_DIM:
            raise ValueError(f"Phase 6 source-faithful state_dim must be {STATE_DIM}.")
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"Phase 6 source-faithful action_dim must be {ACTION_DIM}.")
        if not np.isclose(self.max_action, MAX_ACTION):
            raise ValueError(f"Phase 6 source-faithful max_action must be {MAX_ACTION}.")

        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.actor = MultiHeadActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            hidden_dim=actor_hidden_dim,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
        )

        self.critic = TwinCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=critic_hidden_dim,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
        )

        self.critic_loss_fn = nn.SmoothL1Loss()
        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if state_np.shape[1] != self.state_dim:
            raise ValueError(
                f"state dimension mismatch: {state_np.shape[1]} != {self.state_dim}"
            )
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        action = self.actor(state_t)
        return action.cpu().numpy().reshape(-1).astype(np.float64)

    def train_for_expert(
        self,
        expert_id: int,
        replay_buffer,
        batch_size: int = 32,
    ) -> Dict[str, Optional[float]]:
        """One source-style TD3 update using only transitions for expert_id."""
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample_for_expert(
            expert_id=int(expert_id),
            batch_size=int(batch_size),
            device=self.device,
        )

        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.minimum(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = self.critic_loss_fn(current_q1, target_q) + self.critic_loss_fn(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value: Optional[float] = None

        if self.total_it % self.policy_freq == 0:
            actor_action = self.actor(state)
            actor_loss = -self.critic.q1(state, actor_action).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu().item())

            with torch.no_grad():
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.mul_(1.0 - self.tau)
                    target_param.data.add_(self.tau * param.data)

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.mul_(1.0 - self.tau)
                    target_param.data.add_(self.tau * param.data)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": actor_loss_value,
            "total_it": int(self.total_it),
            "expert_id": int(expert_id),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "max_action": self.max_action,
                "discount": self.discount,
                "tau": self.tau,
                "policy_noise": self.policy_noise,
                "noise_clip": self.noise_clip,
                "policy_freq": self.policy_freq,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str | Path, *, map_location: Optional[str] = None) -> None:
        checkpoint = torch.load(
            Path(path),
            map_location=map_location if map_location is not None else self.device,
            weights_only=False,
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_it = int(checkpoint.get("total_it", 0))
