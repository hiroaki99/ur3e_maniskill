from pathlib import Path
import sys

import sapien


URDF_PATH = (
    Path.home()
    / "ur3e_maniskill"
    / "assets"
    / "ur3e"
    / "ur3e_maniskill.urdf"
)


EXPECTED_JOINTS = {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
}


def main() -> None:
    print("=" * 70)
    print("UR3e URDF Load Test")
    print("=" * 70)

    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found: {URDF_PATH}")
        sys.exit(1)

    print(f"URDF: {URDF_PATH}")
    print()

    # SAPIEN Sceneを作成
    scene = sapien.Scene()

    # Sceneに紐づいたURDF Loaderを作成
    loader = scene.create_urdf_loader()

    # 産業用固定アームなので、rootを固定する
    loader.fix_root_link = True

    print("[1] Loading URDF...")

    robot = loader.load(str(URDF_PATH))

    if robot is None:
        print("FAILED: loader.load() returned None")
        sys.exit(1)

    print("URDF loaded successfully.")
    print()

    # Active joints
    active_joints = robot.get_active_joints()

    print("[2] Active joints")
    print("-" * 70)

    active_joint_names = []

    for i, joint in enumerate(active_joints):
        name = joint.name
        active_joint_names.append(name)
        print(f"{i:2d}: {name}")

    print()
    print(f"Active joint count: {len(active_joints)}")
    print()

    # Links
    links = robot.get_links()

    print("[3] Links")
    print("-" * 70)

    for i, link in enumerate(links):
        print(f"{i:2d}: {link.name}")

    print()
    print(f"Link count: {len(links)}")
    print()

    # Joint validation
    actual_joints = set(active_joint_names)

    missing = EXPECTED_JOINTS - actual_joints
    unexpected = actual_joints - EXPECTED_JOINTS

    print("[4] Validation")
    print("-" * 70)

    if missing:
        print("Missing expected joints:")
        for name in sorted(missing):
            print(f"  - {name}")

    if unexpected:
        print("Unexpected active joints:")
        for name in sorted(unexpected):
            print(f"  - {name}")

    if len(active_joints) != 6:
        print()
        print(
            f"FAILED: expected 6 active joints, "
            f"but found {len(active_joints)}"
        )
        sys.exit(1)

    if missing:
        print()
        print("FAILED: expected UR3e joints are missing.")
        sys.exit(1)

    print("Active joint count: OK")
    print("Expected joint names: OK")
    print()

    print("=" * 70)
    print("SUCCESS")
    print("UR3e URDF was loaded successfully.")
    print("All six expected active joints were found.")
    print("=" * 70)


if __name__ == "__main__":
    main()