from pathlib import Path
import xml.etree.ElementTree as ET


URDF_PATH = (
    Path.home()
    / "ur3e_maniskill"
    / "assets"
    / "ur3e_ezgripper"
    / "ur3e_ezgripper.urdf"
)


def main() -> None:
    if not URDF_PATH.exists():
        raise FileNotFoundError(URDF_PATH)

    root = ET.parse(URDF_PATH).getroot()

    print("=" * 80)
    print("UR3e + EZGripper URDF inspection")
    print("=" * 80)
    print("URDF:", URDF_PATH)

    non_fixed_joints = []

    for joint in root.findall("./joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "unknown")

        parent_element = joint.find("parent")
        child_element = joint.find("child")
        limit_element = joint.find("limit")
        mimic_element = joint.find("mimic")

        parent = (
            parent_element.attrib["link"]
            if parent_element is not None
            else "none"
        )

        child = (
            child_element.attrib["link"]
            if child_element is not None
            else "none"
        )

        if joint_type != "fixed":
            non_fixed_joints.append(name)

        print()
        print(f"Joint: {name}")
        print(f"  type   : {joint_type}")
        print(f"  parent : {parent}")
        print(f"  child  : {child}")

        if limit_element is not None:
            print(
                "  limit  :",
                {
                    "lower": limit_element.attrib.get("lower"),
                    "upper": limit_element.attrib.get("upper"),
                    "effort": limit_element.attrib.get("effort"),
                    "velocity": limit_element.attrib.get("velocity"),
                },
            )

        if mimic_element is not None:
            print(
                "  mimic  :",
                {
                    "joint": mimic_element.attrib.get("joint"),
                    "multiplier": mimic_element.attrib.get(
                        "multiplier",
                        "1.0",
                    ),
                    "offset": mimic_element.attrib.get(
                        "offset",
                        "0.0",
                    ),
                },
            )

    print()
    print("=" * 80)
    print("Non-fixed joints")
    print("=" * 80)

    for index, name in enumerate(non_fixed_joints):
        print(f"{index:2d}: {name}")

    print()
    print("Non-fixed joint count:", len(non_fixed_joints))


if __name__ == "__main__":
    main()