from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


URDF_PATH = (
    Path.home()
    / "ur3e_maniskill"
    / "assets"
    / "ezgripper_gen2"
    / "ezgripper_fragment.urdf"
)


def main() -> None:
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    # robot直下だけを取得する
    links = root.findall("./link")
    joints = root.findall("./joint")

    print("=" * 80)
    print("EZGripper URDF structure")
    print("=" * 80)
    print("URDF:", URDF_PATH)

    print("\n[Top-level links]")
    for link in links:
        print(" -", link.attrib["name"])

    print("\n[Top-level joints]")
    for joint in joints:
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "unknown")

        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        mimic_elem = joint.find("mimic")

        parent = (
            parent_elem.attrib.get("link", "unknown")
            if parent_elem is not None
            else "none"
        )
        child = (
            child_elem.attrib.get("link", "unknown")
            if child_elem is not None
            else "none"
        )

        print(
            f" - {name}\n"
            f"     type   : {joint_type}\n"
            f"     parent : {parent}\n"
            f"     child  : {child}"
        )

        if mimic_elem is not None:
            print(
                "     mimic  :",
                mimic_elem.attrib,
            )

    names = [joint.attrib["name"] for joint in joints]
    duplicates = [
        name
        for name, count in Counter(names).items()
        if count > 1
    ]

    print("\n[Top-level duplicate joints]")
    if duplicates:
        for name in duplicates:
            print(" -", name)
    else:
        print(" none")

    active_joints = [
        joint
        for joint in joints
        if joint.attrib.get("type") not in {
            "fixed",
            "floating",
        }
    ]

    print("\n[Non-fixed joints]")
    for joint in active_joints:
        print(
            " -",
            joint.attrib["name"],
            f"({joint.attrib.get('type')})",
        )

    print("\n[Summary]")
    print("Top-level links:", len(links))
    print("Top-level joints:", len(joints))
    print("Non-fixed joints:", len(active_joints))

    ros2_control_joints = root.findall(
        "./ros2_control/joint"
    )

    print("\n[ros2_control metadata joints]")
    for joint in ros2_control_joints:
        print(" -", joint.attrib["name"])


if __name__ == "__main__":
    main()