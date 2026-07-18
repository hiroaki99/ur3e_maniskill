from pathlib import Path
import re
import sys


urdf_path = (
    Path.home()
    / "ur3e_maniskill"
    / "assets"
    / "ur3e"
    / "ur3e_maniskill.urdf"
)

if not urdf_path.exists():
    print(f"ERROR: URDF not found: {urdf_path}")
    sys.exit(1)

text = urdf_path.read_text()

mesh_paths = re.findall(r'filename="([^"]+)"', text)

print(f"URDF: {urdf_path}")
print(f"Mesh references: {len(mesh_paths)}")
print()

missing = []

for mesh_path_str in mesh_paths:
    mesh_path = Path(mesh_path_str)

    # 相対パスならURDFファイルの場所を基準にする
    if not mesh_path.is_absolute():
        mesh_path = urdf_path.parent / mesh_path

    exists = mesh_path.exists()

    print(
        f"{'OK' if exists else 'MISSING':8s} "
        f"{mesh_path_str}"
    )

    if not exists:
        missing.append(mesh_path_str)

print()
print("=" * 60)

if missing:
    print(f"FAILED: {len(missing)} mesh file(s) missing")
    sys.exit(1)
else:
    print("SUCCESS: all mesh files exist")