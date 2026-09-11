from pathlib import Path
import sys

import gymnasium as gym
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e  # noqa: F401
import envs.ur3e_reach  # noqa: F401


def main() -> None:
    num_envs = 8

    env = gym.make(
        "UR3eReach-v0",
        num_envs=num_envs,
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_joint_delta_pos",
    )

    obs, _ = env.reset(seed=0)
    base_env = env.unwrapped

    initial_info = base_env.evaluate()

    initial_distance = initial_info[
        "tcp_to_goal_dist"
    ].clone()

    initial_tcp = base_env.agent.tcp.pose.p.clone()
    goal_position = base_env.goal_site.pose.p.clone()
    initial_qpos = base_env.agent.robot.get_qpos().clone()

    min_distance = initial_distance.clone()
    total_reward = torch.zeros_like(initial_distance)

    zero_action = torch.zeros(
        env.action_space.shape,
        dtype=torch.float32,
        device=initial_distance.device,
    )

    # 100step目でTimeLimitによるリセットが発生する可能性があるため99step
    for _ in range(99):
        obs, reward, terminated, truncated, info = env.step(
            zero_action
        )

        current_distance = info["tcp_to_goal_dist"]

        min_distance = torch.minimum(
            min_distance,
            current_distance,
        )

        total_reward += reward

    final_info = base_env.evaluate()
    final_distance = final_info["tcp_to_goal_dist"]
    final_qpos = base_env.agent.robot.get_qpos()

    qpos_drift = torch.max(
        torch.abs(final_qpos - initial_qpos),
        dim=1,
    ).values

    print("=" * 70)
    print("UR3eReach zero-action diagnostic")
    print("=" * 70)

    print("\nInitial TCP:")
    print(initial_tcp)

    print("\nGoal:")
    print(goal_position)

    print("\nInitial distance:")
    print(initial_distance)
    print("mean:", initial_distance.mean().item())

    print("\nMinimum distance:")
    print(min_distance)
    print("mean:", min_distance.mean().item())

    print("\nFinal distance:")
    print(final_distance)
    print("mean:", final_distance.mean().item())

    print("\nZero-action return:")
    print(total_reward)
    print("mean:", total_reward.mean().item())

    print("\nMaximum qpos drift:")
    print(qpos_drift)
    print("mean:", qpos_drift.mean().item())

    env.close()


if __name__ == "__main__":
    main()