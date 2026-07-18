from pathlib import Path

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import URDFLoader


URDF_PATH = (
    Path.home()
    / "ur3e_maniskill"
    / "assets"
    / "ur3e"
    / "ur3e_maniskill.urdf"
)


def main() -> None:
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    print("=" * 60)
    print("URDF load test")
    print("=" * 60)

    print("URDF:", URDF_PATH)

    loader = URDFLoader()
    loader.set_scene(ManiSkillScene())

    robot = loader.load(str(URDF_PATH))

    print("\n[Active joints]")
    for name in robot.active_joints_map.keys():
        print(" -", name)

    print("\n[Links]")
    for name in robot.links_map.keys():
        print(" -", name)

    print("\n[Summary]")
    print("Active joints:", len(robot.active_joints_map))
    print("Links:", len(robot.links_map))

    expected_joints = {
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    }

    actual_joints = set(robot.active_joints_map.keys())

    missing = expected_joints - actual_joints

    if missing:
        print("\nFAILED")
        print("Missing active joints:")
        for name in sorted(missing):
            print(" -", name)
        raise RuntimeError("UR3e active joint check failed")

    print("\nSUCCESS")
    print("UR3e loaded successfully.")
    print("All six expected active joints were found.")


if __name__ == "__main__":
    main()