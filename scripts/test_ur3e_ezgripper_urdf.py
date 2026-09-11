from pathlib import Path
import sys

import numpy as np
import sapien


URDF_PATH = (
    Path.home()
    / "ur3e_maniskill"
    / "assets"
    / "ur3e_ezgripper"
    / "ur3e_ezgripper.urdf"
)


EXPECTED_ARM_JOINTS = {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
}

EXPECTED_GRIPPER_LINKS = {
    "gripper_ezgripper_mount",
    "gripper_ezgripper_palm_link",
    "gripper_ezgripper_finger_L1_1",
    "gripper_ezgripper_finger_L1_2",
    "gripper_ezgripper_finger_L2_1",
    "gripper_ezgripper_finger_L2_2",
    "gripper_ezgripper_finger_pad_1",
    "gripper_ezgripper_finger_pad_2",
    "grasp_tcp",
}


def main() -> None:
    print("=" * 80)
    print("UR3e + EZGripper SAPIEN load test")
    print("=" * 80)
    print("URDF:", URDF_PATH)

    if not URDF_PATH.exists():
        raise FileNotFoundError(URDF_PATH)

    scene = sapien.Scene()

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True

    robot = loader.load(str(URDF_PATH))

    if robot is None:
        print("FAILED: loader.load() returned None")
        sys.exit(1)

    active_joints = robot.get_active_joints()
    qlimits = robot.get_qlimits()

    print()
    print("[Active joints]")
    print("-" * 80)

    active_joint_names = []

    for index, joint in enumerate(active_joints):
        active_joint_names.append(joint.name)

        limit = np.asarray(qlimits[index])

        print(
            f"{index:2d}: {joint.name}\n"
            f"    lower={limit[0]: .6f}, "
            f"upper={limit[1]: .6f}"
        )

    print()
    print("Active joint count:", len(active_joints))

    links = robot.get_links()
    link_names = {link.name for link in links}

    print()
    print("[Links]")
    print("-" * 80)

    for index, link in enumerate(links):
        print(f"{index:2d}: {link.name}")

    print()
    print("Link count:", len(links))

    actual_joint_names = set(active_joint_names)

    missing_arm_joints = (
        EXPECTED_ARM_JOINTS
        - actual_joint_names
    )

    missing_gripper_links = (
        EXPECTED_GRIPPER_LINKS
        - link_names
    )

    print()
    print("[Validation]")
    print("-" * 80)

    if missing_arm_joints:
        print("Missing arm joints:")
        for name in sorted(missing_arm_joints):
            print(" -", name)
    else:
        print("UR3e arm joints: OK")

    if missing_gripper_links:
        print("Missing gripper links:")
        for name in sorted(missing_gripper_links):
            print(" -", name)
    else:
        print("EZGripper links: OK")

    gripper_active_joints = [
        name
        for name in active_joint_names
        if name.startswith("gripper_")
    ]

    print()
    print("[EZGripper active joints]")

    for name in gripper_active_joints:
        print(" -", name)

    print()
    print(
        "EZGripper active joint count:",
        len(gripper_active_joints),
    )

    if missing_arm_joints or missing_gripper_links:
        print()
        print("FAILED")
        sys.exit(1)

    print()
    print("=" * 80)
    print("SUCCESS")
    print("UR3e + EZGripper loaded successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()