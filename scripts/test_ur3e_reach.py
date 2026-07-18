from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e  # noqa: F401
import envs.ur3e_reach  # noqa: F401


def main():
    num_envs = 4

    env = gym.make(
        "UR3eReach-v0",
        num_envs=num_envs,
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_joint_delta_pos",
    )

    obs, info = env.reset(seed=0)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Observation shape:", obs.shape)

    for step in range(200):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            print(
                f"step={step:3d}",
                f"reward_mean={reward.mean().item():.4f}",
                f"distance_mean="
                f"{info['tcp_to_goal_dist'].mean().item():.4f}",
                f"success={info['success'].sum().item()}/{num_envs}",
            )

        assert torch.isfinite(obs).all()
        assert torch.isfinite(reward).all()

    env.close()
    print("SUCCESS: UR3eReach-v0 ran for 200 steps.")


if __name__ == "__main__":
    main()