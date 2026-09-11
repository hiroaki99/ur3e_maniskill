#!/usr/bin/env python3
"""Add Week3 TD3/sweep settings without changing validated Week1/Week2 values."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ur3e_pick_place.yaml"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    path = args.config
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backup = path.with_suffix(path.suffix + ".week2_backup")
    if not backup.exists():
        shutil.copy2(path, backup)

    cfg["week3"] = {
        "reference_sweep": {
            "radii_mm": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            "num_directions": 8,
            "target_success_rate_min": 0.20,
            "target_success_rate_max": 0.60,
            "selection_target_success_rate": 0.40,
            "both_relation": "same",
        },
        "td3": {
            "actor_hidden_dim": 32,
            "critic_hidden_dim": 128,
            "discount": 0.99,
            "tau": 0.05,
            "exploration_noise": 0.10,
            "policy_noise": 0.05,
            "noise_clip": 0.10,
            "policy_freq": 2,
            "actor_lr": 0.0001,
            "critic_lr": 0.001,
            "weight_decay": 0.01,
            "batch_size": 128,
            "replay_size": 200000,
            "start_timesteps": 1000,
            "train_seed": 3100,
            "eval_seed": 3200,
            "max_steps_per_episode": 1200,
            "smoke_timesteps": 5000,
            "formal_timesteps": 30000,
            "curriculum_stage_timesteps": 15000,
            "eval_interval": 5000,
            "smoke_eval_interval": 1000,
            "checkpoint_interval": 5000,
            "log_interval": 500,
            "eval_num_directions": 8,
            "eval_episodes_per_direction": 1,
        },
    }

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print("updated:", path)
    print("backup :", backup)
    print("Week2 scales preserved:", cfg.get("observation_scaling"))
    print("alpha preserved:", cfg.get("residual_pick_place", {}).get("alpha"))
    print("Week3 TD3 settings added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
