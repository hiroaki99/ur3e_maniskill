from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.ur3e import UR3e


print("=" * 60)
print("UR3e Agent Registration Test")
print("=" * 60)

print("Class:", UR3e)
print("UID:", UR3e.uid)
print("URDF:", UR3e.urdf_path)
print("Joint names:")

for i, name in enumerate(UR3e.arm_joint_names):
    print(f"{i}: {name}")

print()
print("SUCCESS: UR3e Agent class imported successfully.")