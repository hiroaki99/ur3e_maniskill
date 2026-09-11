from pathlib import Path
import sys

import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import agents.ur3e  # noqa: F401
import envs.ur3e_reach  # noqa: F401


def check(control_mode=None):
    kwargs = {
        "num_envs": 1,
        "obs_mode": "state",
        "reward_mode": "normalized_dense",
        "sim_backend": "physx_cuda",
    }

    if control_mode is not None:
        kwargs["control_mode"] = control_mode

    env = gym.make("UR3eReach-v0", **kwargs)
    env.reset(seed=0)

    print("指定値:", control_mode)
    print("実際のcontrol_mode:", env.unwrapped.agent.control_mode)
    print("action space:", env.action_space)

    env.close()


print("=== control_mode省略 ===")
check()

print("\n=== pd_joint_delta_pos指定 ===")
check("pd_joint_delta_pos")