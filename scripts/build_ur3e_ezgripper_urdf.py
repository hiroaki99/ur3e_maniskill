from __future__ import annotations

import copy
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path.home() / "ur3e_maniskill"

UR3E_URDF = (
    PROJECT_ROOT
    / "assets"
    / "ur3e"
    / "ur3e_maniskill.urdf"
)

EZGRIPPER_FRAGMENT = (
    PROJECT_ROOT
    / "assets"
    / "ezgripper_gen2"
    / "ezgripper_fragment.urdf"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "assets"
    / "ur3e_ezgripper"
)

OUTPUT_URDF = OUTPUT_DIR / "ur3e_ezgripper.urdf"

UR3E_MESH_SOURCE = (
    PROJECT_ROOT
    / "assets"
    / "ur3e"
    / "meshes"
)

UR3E_MESH_DESTINATION = (
    OUTPUT_DIR
    / "meshes"
    / "ur3e"
)

EZGRIPPER_MESH_DESTINATION = (
    OUTPUT_DIR
    / "meshes"
    / "ezgripper"
)

# 公式EZGripperリポジトリを配置した場所
EZGRIPPER_DESCRIPTION_DIR = (
    PROJECT_ROOT
    / "third_party"
    / "EZGripper_ros2"
    / "ezgripper_description"
)


def local_tag(element: ET.Element) -> str:
    """XML名前空間を除いたタグ名を返す。"""
    return element.tag.split("}")[-1]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def copy_ur3e_meshes() -> None:
    """既存UR3eのmeshを統合先へコピーする。"""
    require_directory(UR3E_MESH_SOURCE)

    UR3E_MESH_DESTINATION.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        UR3E_MESH_SOURCE,
        UR3E_MESH_DESTINATION,
        dirs_exist_ok=True,
    )


def rewrite_ur3e_mesh_paths(robot_root: ET.Element) -> None:
    """
    UR3e URDF内のmeshパスを、

    meshes/visual/base.dae
        ↓
    meshes/ur3e/visual/base.dae

    のように変更する。
    """
    package_prefix = (
        "package://ur_description/meshes/ur3e/"
    )

    for mesh in robot_root.findall(".//mesh"):
        filename = mesh.get("filename")

        if not filename:
            continue

        if filename.startswith(package_prefix):
            suffix = filename[len(package_prefix):]

            mesh.set(
                "filename",
                f"meshes/ur3e/{suffix}",
            )

        elif filename.startswith("meshes/"):
            suffix = filename[len("meshes/"):]

            # すでに書き換え済みの場合は変更しない
            if suffix.startswith("ur3e/"):
                continue

            mesh.set(
                "filename",
                f"meshes/ur3e/{suffix}",
            )


def resolve_ezgripper_mesh_path(
    filename: str,
) -> Path:
    """EZGripper URDF中のmesh参照を実ファイルへ解決する。"""
    package_prefix = "package://ezgripper_description/"

    if filename.startswith(package_prefix):
        suffix = filename[len(package_prefix):]

        return (
            EZGRIPPER_DESCRIPTION_DIR
            / suffix
        ).resolve()

    if filename.startswith("file://"):
        return Path(
            filename[len("file://"):]
        ).expanduser().resolve()

    path = Path(filename).expanduser()

    if path.is_absolute():
        return path.resolve()

    candidates = [
        (
            EZGRIPPER_FRAGMENT.parent
            / path
        ).resolve(),
        (
            EZGRIPPER_DESCRIPTION_DIR
            / path
        ).resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # エラーメッセージ用に最初の候補を返す
    return candidates[0]


def ezgripper_mesh_suffix(
    filename: str,
    source_path: Path,
) -> Path:
    """
    mesh以下の相対構造を維持する。

    .../meshes/visual/example.dae
        ↓
    visual/example.dae
    """
    normalized = filename.replace("\\", "/")

    if "/meshes/" in normalized:
        suffix = normalized.split(
            "/meshes/",
            1,
        )[1]

        return Path(suffix)

    if normalized.startswith("meshes/"):
        return Path(
            normalized[len("meshes/"):]
        )

    try:
        return source_path.relative_to(
            EZGRIPPER_DESCRIPTION_DIR
            / "meshes"
        )
    except ValueError:
        return Path(source_path.name)


def rewrite_and_copy_ezgripper_meshes(
    element: ET.Element,
) -> None:
    """
    EZGripper要素内のmeshを統合ディレクトリへコピーし、
    URDF内の参照を相対パスへ変更する。
    """
    for mesh in element.findall(".//mesh"):
        filename = mesh.get("filename")

        if not filename:
            continue

        source_path = resolve_ezgripper_mesh_path(
            filename
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                "EZGripper mesh not found:\n"
                f"  URDF reference: {filename}\n"
                f"  Resolved path : {source_path}"
            )

        suffix = ezgripper_mesh_suffix(
            filename,
            source_path,
        )

        destination = (
            EZGRIPPER_MESH_DESTINATION
            / suffix
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_path,
            destination,
        )

        mesh.set(
            "filename",
            (
                Path("meshes")
                / "ezgripper"
                / suffix
            ).as_posix(),
        )


def append_grasp_tcp(robot_root: ET.Element) -> None:
    """
    左右指の間の把持中心を表す仮TCPを追加する。

    xyzは暫定値。
    GUI表示後に実際の把持中心へ調整する。
    """
    existing_links = {
        link.get("name")
        for link in robot_root.findall("./link")
    }

    existing_joints = {
        joint.get("name")
        for joint in robot_root.findall("./joint")
    }

    if "grasp_tcp" not in existing_links:
        tcp_link = ET.Element(
            "link",
            {"name": "grasp_tcp"},
        )

        visual = ET.SubElement(
            tcp_link,
            "visual",
        )

        ET.SubElement(
            visual,
            "origin",
            {
                "xyz": "0 0 0",
                "rpy": "0 0 0",
            },
        )

        geometry = ET.SubElement(
            visual,
            "geometry",
        )

        ET.SubElement(
            geometry,
            "sphere",
            {"radius": "0.008"},
        )

        material = ET.SubElement(
            visual,
            "material",
            {"name": "grasp_tcp_green"},
        )

        ET.SubElement(
            material,
            "color",
            {"rgba": "0 1 0 1"},
        )

        robot_root.append(tcp_link)

    tcp_joint_name = "gripper_grasp_tcp_joint"

    if tcp_joint_name not in existing_joints:
        tcp_joint = ET.Element(
            "joint",
            {
                "name": tcp_joint_name,
                "type": "fixed",
            },
        )

        ET.SubElement(
            tcp_joint,
            "parent",
            {
                "link":
                    "gripper_ezgripper_palm_link"
            },
        )

        ET.SubElement(
            tcp_joint,
            "child",
            {"link": "grasp_tcp"},
        )

        # EZGripperのpalm座標系から見た暫定位置
        ET.SubElement(
            tcp_joint,
            "origin",
            {
                "xyz": "0.11 0 0",
                "rpy": "0 0 0",
            },
        )

        robot_root.append(tcp_joint)


def merge_ezgripper(
    ur3e_root: ET.Element,
    ezgripper_root: ET.Element,
) -> None:
    """
    EZGripper fragmentから、ManiSkillに必要な要素だけを追加する。
    """
    allowed_tags = {
        "material",
        "link",
        "joint",
    }

    existing_link_names = {
        element.get("name")
        for element in ur3e_root.findall("./link")
    }

    existing_joint_names = {
        element.get("name")
        for element in ur3e_root.findall("./joint")
    }

    existing_material_names = {
        element.get("name")
        for element in ur3e_root.findall("./material")
    }

    for child in list(ezgripper_root):
        tag = local_tag(child)

        if tag not in allowed_tags:
            print(
                f"Skipping top-level tag: {tag}"
            )
            continue

        name = child.get("name")

        if not name:
            continue

        if tag == "link":
            if name in existing_link_names:
                raise ValueError(
                    f"Duplicate link name: {name}"
                )

        elif tag == "joint":
            if name in existing_joint_names:
                raise ValueError(
                    f"Duplicate joint name: {name}"
                )

        elif tag == "material":
            # UR3e側と同名materialなら重複追加しない
            if name in existing_material_names:
                print(
                    "Skipping duplicate material:",
                    name,
                )
                continue

        copied = copy.deepcopy(child)

        rewrite_and_copy_ezgripper_meshes(
            copied
        )

        ur3e_root.append(copied)

        if tag == "link":
            existing_link_names.add(name)

        elif tag == "joint":
            existing_joint_names.add(name)

        elif tag == "material":
            existing_material_names.add(name)


def validate_robot(robot_root: ET.Element) -> None:
    links = robot_root.findall("./link")
    joints = robot_root.findall("./joint")

    link_names = [
        link.get("name")
        for link in links
    ]

    joint_names = [
        joint.get("name")
        for joint in joints
    ]

    duplicate_links = [
        name
        for name, count in Counter(
            link_names
        ).items()
        if count > 1
    ]

    duplicate_joints = [
        name
        for name, count in Counter(
            joint_names
        ).items()
        if count > 1
    ]

    if duplicate_links:
        raise ValueError(
            "Duplicate links: "
            f"{duplicate_links}"
        )

    if duplicate_joints:
        raise ValueError(
            "Duplicate joints: "
            f"{duplicate_joints}"
        )

    link_name_set = set(link_names)

    missing_parent_child = []

    for joint in joints:
        joint_name = joint.get("name")

        parent_element = joint.find("parent")
        child_element = joint.find("child")

        if (
            parent_element is None
            or child_element is None
        ):
            missing_parent_child.append(
                (
                    joint_name,
                    "parent/child element missing",
                )
            )
            continue

        parent_name = parent_element.get("link")
        child_name = child_element.get("link")

        if parent_name not in link_name_set:
            missing_parent_child.append(
                (
                    joint_name,
                    f"parent link missing: {parent_name}",
                )
            )

        if child_name not in link_name_set:
            missing_parent_child.append(
                (
                    joint_name,
                    f"child link missing: {child_name}",
                )
            )

    if missing_parent_child:
        lines = [
            f"{joint}: {reason}"
            for joint, reason
            in missing_parent_child
        ]

        raise ValueError(
            "Invalid joint connections:\n"
            + "\n".join(lines)
        )

    required_links = {
        "tool0",
        "gripper_ezgripper_mount",
        "gripper_ezgripper_palm_link",
        "gripper_ezgripper_finger_pad_1",
        "gripper_ezgripper_finger_pad_2",
        "grasp_tcp",
    }

    missing_required_links = (
        required_links
        - link_name_set
    )

    if missing_required_links:
        raise ValueError(
            "Required links missing: "
            f"{sorted(missing_required_links)}"
        )

    print()
    print("[Validation]")
    print("Links:", len(links))
    print("Joints:", len(joints))
    print("Duplicate links: none")
    print("Duplicate joints: none")
    print("Joint parent/child connections: OK")
    print("Required gripper links: OK")


def main() -> None:
    require_file(UR3E_URDF)
    require_file(EZGRIPPER_FRAGMENT)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    copy_ur3e_meshes()

    ur3e_tree = ET.parse(UR3E_URDF)
    ur3e_root = ur3e_tree.getroot()

    ezgripper_tree = ET.parse(
        EZGRIPPER_FRAGMENT
    )
    ezgripper_root = (
        ezgripper_tree.getroot()
    )

    ur3e_root.set(
        "name",
        "ur3e_ezgripper",
    )

    rewrite_ur3e_mesh_paths(
        ur3e_root
    )

    merge_ezgripper(
        ur3e_root,
        ezgripper_root,
    )

    append_grasp_tcp(
        ur3e_root
    )

    validate_robot(
        ur3e_root
    )

    ET.indent(
        ur3e_tree,
        space="  ",
    )

    ur3e_tree.write(
        OUTPUT_URDF,
        encoding="utf-8",
        xml_declaration=True,
    )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("Generated:", OUTPUT_URDF)
    print("=" * 70)


if __name__ == "__main__":
    main()