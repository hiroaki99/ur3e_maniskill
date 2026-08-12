#!/usr/bin/env python3
"""
TD3 for UR3e Residual Reinforcement Learning.

Source concept:
    yaginuma-tracker / TD3.py

Ported contract:
    state  = 44
    action = 6 normalized residual joints

Maintained from source:
- Twin Critic
- target policy smoothing
- minimum target Q
- delayed Actor update
- soft target update
- SmoothL1 critic loss
- AdamW

Intentional port changes:
- state 22 -> 44
- action 2 -> 6
- max_action 0.1 -> normalized 1.0
- 13 expert heads -> single Actor
- discount 0.0 -> configurable (default 0.99)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ==========================================================
# Actor
# ==========================================================

class Actor(
    nn.Module
):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        max_action: float = 1.0,
    ):

        super().__init__()

        self.max_action = float(
            max_action
        )

        self.net = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                action_dim,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        state,
    ):

        return (
            self.max_action
            * self.net(
                state
            )
        )


# ==========================================================
# Twin Critic
# ==========================================================

class Critic(
    nn.Module
):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):

        super().__init__()

        input_dim = (
            state_dim
            + action_dim
        )

        self.q1_net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        self.q2_net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        state,
        action,
    ):

        state_action = torch.cat(
            [
                state,
                action,
            ],
            dim=1,
        )

        q1 = self.q1_net(
            state_action
        )

        q2 = self.q2_net(
            state_action
        )

        return (
            q1,
            q2,
        )

    def q1(
        self,
        state,
        action,
    ):

        state_action = torch.cat(
            [
                state,
                action,
            ],
            dim=1,
        )

        return self.q1_net(
            state_action
        )


# ==========================================================
# TD3
# ==========================================================

class TD3:

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        actor_hidden_dim: int = 32,
        critic_hidden_dim: int = 128,
        discount: float = 0.99,
        tau: float = 0.05,
        policy_noise: float = 0.05,
        noise_clip: float = 0.10,
        policy_freq: int = 2,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = "cpu",
        seed: int = 0,
    ):

        self.state_dim = int(
            state_dim
        )

        self.action_dim = int(
            action_dim
        )

        self.max_action = float(
            max_action
        )

        self.discount = float(
            discount
        )

        self.tau = float(
            tau
        )

        self.policy_noise = float(
            policy_noise
        )

        self.noise_clip = float(
            noise_clip
        )

        self.policy_freq = int(
            policy_freq
        )

        self.actor_hidden_dim = int(
            actor_hidden_dim
        )

        self.critic_hidden_dim = int(
            critic_hidden_dim
        )

        self.actor_lr = float(
            actor_lr
        )

        self.critic_lr = float(
            critic_lr
        )

        self.weight_decay = float(
            weight_decay
        )

        self.seed = int(
            seed
        )

        self.device = torch.device(
            device
        )

        torch.manual_seed(
            self.seed
        )

        if (
            self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.manual_seed_all(
                self.seed
            )

        # --------------------------------------------------
        # Actor
        # --------------------------------------------------

        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=(
                self.actor_hidden_dim
            ),
            max_action=self.max_action,
        ).to(
            self.device
        )

        self.actor_target = copy.deepcopy(
            self.actor
        )

        self.actor_optimizer = (
            torch.optim.AdamW(
                self.actor.parameters(),
                lr=self.actor_lr,
                weight_decay=(
                    self.weight_decay
                ),
            )
        )

        # --------------------------------------------------
        # Critic
        # --------------------------------------------------

        self.critic = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=(
                self.critic_hidden_dim
            ),
        ).to(
            self.device
        )

        self.critic_target = copy.deepcopy(
            self.critic
        )

        self.critic_optimizer = (
            torch.optim.AdamW(
                self.critic.parameters(),
                lr=self.critic_lr,
                weight_decay=(
                    self.weight_decay
                ),
            )
        )

        self.critic_loss_function = (
            nn.SmoothL1Loss()
        )

        self.total_it = 0

    # ======================================================
    # Action
    # ======================================================

    def select_action(
        self,
        state,
    ) -> np.ndarray:

        state = np.asarray(
            state,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        if state.shape[
            1
        ] != self.state_dim:

            raise ValueError(
                "state dimension mismatch: "
                f"{state.shape}"
            )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        self.actor.eval()

        with torch.no_grad():

            action = self.actor(
                state_tensor
            )

        self.actor.train()

        action = (
            action
            .cpu()
            .numpy()[0]
        )

        return np.clip(
            action,
            -self.max_action,
            self.max_action,
        ).astype(
            np.float32
        )

    # ======================================================
    # Train
    # ======================================================

    def train(
        self,
        replay_buffer,
        batch_size: int,
    ) -> dict:

        self.total_it += 1

        (
            state,
            action,
            next_state,
            reward,
            not_done,
        ) = replay_buffer.sample(
            batch_size
        )

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        action = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=self.device,
        )

        next_state = torch.as_tensor(
            next_state,
            dtype=torch.float32,
            device=self.device,
        )

        reward = torch.as_tensor(
            reward,
            dtype=torch.float32,
            device=self.device,
        )

        not_done = torch.as_tensor(
            not_done,
            dtype=torch.float32,
            device=self.device,
        )

        # --------------------------------------------------
        # Target
        # --------------------------------------------------

        with torch.no_grad():

            noise = (
                torch.randn_like(
                    action
                )
                * self.policy_noise
            )

            noise = noise.clamp(
                -self.noise_clip,
                self.noise_clip,
            )

            next_action = (
                self.actor_target(
                    next_state
                )
                + noise
            )

            next_action = next_action.clamp(
                -self.max_action,
                self.max_action,
            )

            (
                target_q1,
                target_q2,
            ) = self.critic_target(
                next_state,
                next_action,
            )

            target_q = torch.minimum(
                target_q1,
                target_q2,
            )

            target_q = (
                reward
                + not_done
                * self.discount
                * target_q
            )

        # --------------------------------------------------
        # Critic update
        # --------------------------------------------------

        (
            current_q1,
            current_q2,
        ) = self.critic(
            state,
            action,
        )

        critic_loss_q1 = (
            self.critic_loss_function(
                current_q1,
                target_q,
            )
        )

        critic_loss_q2 = (
            self.critic_loss_function(
                current_q2,
                target_q,
            )
        )

        critic_loss = (
            critic_loss_q1
            + critic_loss_q2
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        self.critic_optimizer.step()

        actor_updated = False
        actor_loss_value = None

        # --------------------------------------------------
        # Delayed Actor update
        # --------------------------------------------------

        if (
            self.total_it
            % self.policy_freq
            == 0
        ):

            actor_action = self.actor(
                state
            )

            actor_loss = -self.critic.q1(
                state,
                actor_action,
            ).mean()

            self.actor_optimizer.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            self.actor_optimizer.step()

            actor_loss_value = float(
                actor_loss.detach().cpu()
            )

            actor_updated = True

            # ----------------------------------------------
            # Soft target update
            # ----------------------------------------------

            self._soft_update(
                self.critic,
                self.critic_target,
            )

            self._soft_update(
                self.actor,
                self.actor_target,
            )

        return {
            "total_it":
                int(
                    self.total_it
                ),

            "critic_loss":
                float(
                    critic_loss
                    .detach()
                    .cpu()
                ),

            "critic_loss_q1":
                float(
                    critic_loss_q1
                    .detach()
                    .cpu()
                ),

            "critic_loss_q2":
                float(
                    critic_loss_q2
                    .detach()
                    .cpu()
                ),

            "actor_updated":
                bool(
                    actor_updated
                ),

            "actor_loss":
                actor_loss_value,

            "target_q_mean":
                float(
                    target_q
                    .mean()
                    .detach()
                    .cpu()
                ),

            "current_q1_mean":
                float(
                    current_q1
                    .mean()
                    .detach()
                    .cpu()
                ),

            "current_q2_mean":
                float(
                    current_q2
                    .mean()
                    .detach()
                    .cpu()
                ),
        }

    def _soft_update(
        self,
        source,
        target,
    ):

        with torch.no_grad():

            for (
                source_parameter,
                target_parameter,
            ) in zip(
                source.parameters(),
                target.parameters(),
            ):

                target_parameter.data.copy_(
                    self.tau
                    * source_parameter.data
                    + (
                        1.0
                        - self.tau
                    )
                    * target_parameter.data
                )

    # ======================================================
    # Save / Load
    # ======================================================

    def save(
        self,
        path,
    ):

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "state_dim":
                self.state_dim,

            "action_dim":
                self.action_dim,

            "max_action":
                self.max_action,

            "actor_hidden_dim":
                self.actor_hidden_dim,

            "critic_hidden_dim":
                self.critic_hidden_dim,

            "discount":
                self.discount,

            "tau":
                self.tau,

            "policy_noise":
                self.policy_noise,

            "noise_clip":
                self.noise_clip,

            "policy_freq":
                self.policy_freq,

            "actor_lr":
                self.actor_lr,

            "critic_lr":
                self.critic_lr,

            "weight_decay":
                self.weight_decay,

            "seed":
                self.seed,

            "total_it":
                self.total_it,

            "actor":
                self.actor.state_dict(),

            "actor_target":
                self.actor_target.state_dict(),

            "actor_optimizer":
                self.actor_optimizer.state_dict(),

            "critic":
                self.critic.state_dict(),

            "critic_target":
                self.critic_target.state_dict(),

            "critic_optimizer":
                self.critic_optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            path,
        )

    def load(
        self,
        path,
    ):

        path = Path(
            path
        )

        try:

            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )

        except TypeError:

            # Older PyTorch compatibility
            checkpoint = torch.load(
                path,
                map_location=self.device,
            )

        if int(
            checkpoint[
                "state_dim"
            ]
        ) != self.state_dim:

            raise RuntimeError(
                "checkpoint state_dim mismatch"
            )

        if int(
            checkpoint[
                "action_dim"
            ]
        ) != self.action_dim:

            raise RuntimeError(
                "checkpoint action_dim mismatch"
            )

        self.actor.load_state_dict(
            checkpoint[
                "actor"
            ]
        )

        self.actor_target.load_state_dict(
            checkpoint[
                "actor_target"
            ]
        )

        self.actor_optimizer.load_state_dict(
            checkpoint[
                "actor_optimizer"
            ]
        )

        self.critic.load_state_dict(
            checkpoint[
                "critic"
            ]
        )

        self.critic_target.load_state_dict(
            checkpoint[
                "critic_target"
            ]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint[
                "critic_optimizer"
            ]
        )

        self.total_it = int(
            checkpoint.get(
                "total_it",
                0,
            )
        )