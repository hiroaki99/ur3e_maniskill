from pathlib import Path


PROJECT_ROOT = Path.home() / "ur3e_maniskill"

SOURCE_DIR = (
    PROJECT_ROOT
    / "assets"
    / "ezgripper_gen2"
    / "source_urdf"
)

EZGRIPPER_REPO = (
    PROJECT_ROOT
    / "third_party"
    / "EZGripper_ros2"
)

DESCRIPTION_DIR = (
    EZGRIPPER_REPO
    / "ezgripper_description"
)

CONTROL_DIR = (
    EZGRIPPER_REPO
    / "ezgripper_control"
)


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(SOURCE_DIR)

    for path in SOURCE_DIR.glob("*.xacro"):
        text = path.read_text()

        text = text.replace(
            "$(find ezgripper_description)",
            str(DESCRIPTION_DIR),
        )

        text = text.replace(
            "$(find ezgripper_control)",
            str(CONTROL_DIR),
        )

        path.write_text(text)

        print("Updated:", path)


if __name__ == "__main__":
    main()