from pathlib import Path
import sys

import tyro


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# import時に UR3e Agent が登録される
import agents.ur3e
import agents.ur3e_ezgripper

from mani_skill.examples.demo_robot import Args, main


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)