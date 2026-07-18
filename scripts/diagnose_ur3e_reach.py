from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e  # noqa: F401
import envs.ur3e_reach  # noqa: F401


ENV_ID = "UR3eReach-v0"
NUM_ENVS = 16
NUM_STEPS = 100


def get_distance(env):
    base_env = env.unwrapped
    tcp_pos = base_env.agent.tcp.pose.p
    goal_pos = base_env.goal_site.pose.p
    return torch.linalg.norm(tcp_pos - goal_pos, dim=1)


def evaluate_baseline(name, action_type):
    env = gym.make(
        ENV_ID,
        num_envs=NUM_ENVS,
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_joint_target_delta_pos",
    )

    obs, info = env.reset(seed=0)

    initial_dist = get_distance(env).clone()
    min_dist = initial_dist.clone()
    episode_return = torch.zeros(NUM_ENVS, device=initial_dist.device)
    success_once = torch.zeros(
        NUM_ENVS,
        dtype=torch.bool,
        device=initial_dist.device,
    )

    for _ in range(NUM_STEPS):
        if action_type == "zero":
            action = torch.zeros(
                env.action_space.shape,
                dtype=torch.float32,
                device=initial_dist.device,
            )
        elif action_type == "random":
            action = env.action_space.sample()
        else:
            raise ValueError(action_type)

        obs, reward, terminated, truncated, info = env.step(action)

        distance = get_distance(env)
        min_dist = torch.minimum(min_dist, distance)
        episode_return += reward
        success_once |= info["success"]

    final_dist = get_distance(env)

    print(f"\n[{name}]")
    print(f"initial distance mean : {initial_dist.mean().item():.4f}")
    print(f"minimum distance mean : {min_dist.mean().item():.4f}")
    print(f"final distance mean   : {final_dist.mean().item():.4f}")
    print(f"return mean           : {episode_return.mean().item():.4f}")
    print(
        f"success once          : "
        f"{success_once.sum().item()}/{NUM_ENVS}"
    )

    env.close()


def main():
    evaluate_baseline("Zero action", "zero")
    evaluate_baseline("Random action", "random")


if __name__ == "__main__":
    main()