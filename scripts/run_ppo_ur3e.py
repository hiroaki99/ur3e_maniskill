from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Agentと環境の登録
import agents.ur3e  # noqa: F401
import envs.ur3e_reach  # noqa: F401

PPO_SCRIPT = (
    Path.home()
    / "ManiSkill"
    / "examples"
    / "baselines"
    / "ppo"
    / "ppo.py"
)

if not PPO_SCRIPT.exists():
    raise FileNotFoundError(PPO_SCRIPT)

runpy.run_path(str(PPO_SCRIPT), run_name="__main__")