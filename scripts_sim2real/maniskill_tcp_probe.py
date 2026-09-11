#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np

# プロジェクトルートを Python import path に追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import mani_skill.envs  # noqa: F401

# UR3eReach-v0 を登録
import envs.ur3e_reach  # noqa: F401


TCP_LINK_CANDIDATES = [
    "tool0",
    "tool0_controller",
    "ee_link",
    "tcp",
    "flange",
]

# ============================================================
# 実機 TCP 設定
#
# tf2_echo tool0 tool0_controller で確認した値:
#   translation = [0, 0, 0.150] m
#   rotation    = identity
#
# tool0 座標系の +Z 方向へ 150 mm
# ============================================================

TCP_OFFSET_TOOL0 = np.array(
    [0.0, 0.0, 0.150],
    dtype=np.float64,
)


# REAL_TCP_POSITION = np.array(
#     [
#         0.3738615227529578,
#         0.02002188285652715,
#         0.6927836305889686,
#     ],
#     dtype=np.float64,
# )

REAL_TCP_POSITION = np.array(
    [
        0.37598653516400243,
        -0.12128360891503201,
        0.6593910311634118,
    ],
    dtype=np.float64,
)


# REAL_TCP_QUATERNION_XYZW = np.array(
#     [
#         -0.5252031983687698,
#          0.47506574657280315,
#         -0.5258079669394721,
#          0.47116888560198755,
#     ],
#     dtype=np.float64,
# )

REAL_TCP_QUATERNION_XYZW = np.array(
    [
        -0.5896537134524702,
         0.39106275186111983,
        -0.5897716487268977,
         0.389291439254034,
    ],
    dtype=np.float64,
)

# ============================================================
# 実機 UR3e の /joint_states から取得した関節角
# 2026-08-31 FK/TCP 整合確認用
# ============================================================

# REAL_JOINT_POSITIONS = {
#     "shoulder_pan_joint": 1.584212064743042,
#     "shoulder_lift_joint": -1.5772816143431605,
#     "elbow_joint": -0.0369114875793457,
#     "wrist_1_joint": -1.6347819767394007,
#     "wrist_2_joint": -0.012888256703512013,
#     "wrist_3_joint": 18.852199408023644,
# }

REAL_JOINT_POSITIONS = {
    "shoulder_pan_joint": 1.5842311382293701,
    "shoulder_lift_joint": -1.2043504875949402,
    "elbow_joint": -0.04070211574435234,
    "wrist_1_joint": -2.300890108148092,
    "wrist_2_joint": -0.011884991322652638,
    "wrist_3_joint": 18.852176539863486,
}


def load_validation_pose(path):
    """
    実機UR3eで取得した関節角とTCP姿勢をJSONから読み込む。
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    joint_positions = data["joint_positions"]

    tcp_position = np.array(
        [
            data["tcp_pose"]["position"]["x"],
            data["tcp_pose"]["position"]["y"],
            data["tcp_pose"]["position"]["z"],
        ],
        dtype=np.float64,
    )

    tcp_quaternion_xyzw = np.array(
        [
            data["tcp_pose"]["orientation_xyzw"]["x"],
            data["tcp_pose"]["orientation_xyzw"]["y"],
            data["tcp_pose"]["orientation_xyzw"]["z"],
            data["tcp_pose"]["orientation_xyzw"]["w"],
        ],
        dtype=np.float64,
    )

    return (
        joint_positions,
        tcp_position,
        tcp_quaternion_xyzw,
    )



def wrap_to_pi(angle):
    """
    角度を [-pi, pi] に変換する。

    実機 wrist_3_joint は約 18.85 rad だが、
    FK 上は約 0.00264 rad と等価。
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def set_real_joint_pose(robot, real_joint_positions):
    """
    実機 UR3e から取得した関節角を、
    関節名に基づいて ManiSkill UR3e に設定する。
    """

    active_joints = robot.get_active_joints()

    print("\n=== ManiSkill active joint order ===")
    for i, joint in enumerate(active_joints):
        print(f"  {i}: {joint.name}")
    print("====================================")

    # ManiSkill の現在 qpos を取得
    qpos = robot.get_qpos()

    # Torch Tensor の場合
    if hasattr(qpos, "clone"):
        target_qpos = qpos.clone()

        # num_envs=1 の場合は通常 [1, num_joints]
        batched = target_qpos.ndim == 2

        for i, joint in enumerate(active_joints):
            name = joint.name

            if name not in real_joint_positions:
                continue

            value = wrap_to_pi(real_joint_positions[name])

            if batched:
                target_qpos[0, i] = value
            else:
                target_qpos[i] = value

    # NumPy 等の場合
    else:
        target_qpos = np.asarray(qpos).copy()

        batched = target_qpos.ndim == 2

        for i, joint in enumerate(active_joints):
            name = joint.name

            if name not in real_joint_positions:
                continue

            value = wrap_to_pi(real_joint_positions[name])

            if batched:
                target_qpos[0, i] = value
            else:
                target_qpos[i] = value

    print("\n=== Target ManiSkill qpos ===")
    print(target_qpos)

    robot.set_qpos(target_qpos)

    print("\n=== Actual qpos after set_qpos ===")
    print(robot.get_qpos())


def quaternion_wxyz_to_rotation_matrix(q):
    """
    Quaternion [w, x, y, z] を 3x3 回転行列へ変換する。
    """

    q = np.asarray(q, dtype=np.float64)

    q = q / np.linalg.norm(q)

    w, x, y, z = q

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_angle_error_deg(q1_xyzw, q2_xyzw):
    """
    2つのQuaternion間の最短回転角 [deg] を求める。

    q と -q が同一姿勢であることを考慮する。
    """

    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q2 = np.asarray(q2_xyzw, dtype=np.float64)

    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    dot = abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)

    angle_rad = 2.0 * np.arccos(dot)

    return np.degrees(angle_rad)


def to_list(x):
    """
    Torch tensor / numpy array / ManiSkill batched tensor を
    Python list に変換する。
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    if hasattr(x, "tolist"):
        x = x.tolist()

    # num_envs=1 の場合 [ [x,y,z] ] になることがある
    if len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]

    return x


def find_tcp_link(robot):
    links = robot.get_links()

    print("=== Available ManiSkill robot links ===")
    for link in links:
        print(f"  {link.name}")

    print("======================================")

    link_map = {link.name: link for link in links}

    for name in TCP_LINK_CANDIDATES:
        if name in link_map:
            print(f"[INFO] TCP candidate selected: {name}")
            return link_map[name]

    raise RuntimeError(
        "TCP link could not be detected automatically. "
        "Check the link list printed above and add the correct "
        "link name to TCP_LINK_CANDIDATES."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="ManiSkill environment id, e.g. UR3eReach-v0",
    )

    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_joint_delta_pos",
    )


    parser.add_argument(
        "--validation-pose",
        type=str,
        default=None,
        help=(
            "JSON file containing real UR3e joint positions "
            "and TCP pose for FK validation."
        ),
    )

    args = parser.parse_args()

    if args.validation_pose is not None:
        (
            real_joint_positions,
            real_tcp_position,
            real_tcp_quaternion_xyzw,
        ) = load_validation_pose(args.validation_pose)

    else:
        # 従来値を利用
        real_joint_positions = REAL_JOINT_POSITIONS
        real_tcp_position = REAL_TCP_POSITION
        real_tcp_quaternion_xyzw = REAL_TCP_QUATERNION_XYZW

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        control_mode=args.control_mode,
    )

    obs, info = env.reset()

    base_env = env.unwrapped
    robot = base_env.agent.robot

    set_real_joint_pose(robot, real_joint_positions)

    tcp_link = find_tcp_link(robot)

    links = robot.get_links()
    link_map = {link.name: link for link in links}

    if "base_link" not in link_map:
        raise RuntimeError("base_link was not found in robot links.")

    if "base" not in link_map:
        raise RuntimeError("base was not found in robot links.")

    base_link = link_map["base_link"]
    base = link_map["base"]

    # world -> base_link
    T_world_base_link = base_link.pose

    # world -> base
    T_world_base = base.pose

    # world -> tcp
    T_world_tcp = tcp_link.pose

    # base_link -> tcp
    T_base_link_tcp = T_world_base_link.inv() * T_world_tcp

    # base -> tcp
    T_base_tcp = T_world_base.inv() * T_world_tcp

    # base_link -> base
    T_base_link_base = T_world_base_link.inv() * T_world_base


    # -----------------------------
    # World frame pose
    # -----------------------------
    world_position = to_list(T_world_tcp.p)
    world_quaternion_wxyz = to_list(T_world_tcp.q)

    world_quaternion_xyzw = [
        world_quaternion_wxyz[1],
        world_quaternion_wxyz[2],
        world_quaternion_wxyz[3],
        world_quaternion_wxyz[0],
    ]

    # -----------------------------
    # base_link -> TCP
    # -----------------------------
    base_link_position = to_list(T_base_link_tcp.p)
    base_link_quaternion_wxyz = to_list(T_base_link_tcp.q)

    base_link_quaternion_xyzw = [
        base_link_quaternion_wxyz[1],
        base_link_quaternion_wxyz[2],
        base_link_quaternion_wxyz[3],
        base_link_quaternion_wxyz[0],
    ]


    # -----------------------------
    # base -> tool0
    # -----------------------------
    base_tool0_position = np.asarray(
        to_list(T_base_tcp.p),
        dtype=np.float64,
    )

    base_tool0_quaternion_wxyz = np.asarray(
        to_list(T_base_tcp.q),
        dtype=np.float64,
    )

    base_tool0_quaternion_xyzw = np.array(
        [
            base_tool0_quaternion_wxyz[1],
            base_tool0_quaternion_wxyz[2],
            base_tool0_quaternion_wxyz[3],
            base_tool0_quaternion_wxyz[0],
        ],
        dtype=np.float64,
    )


    # ============================================================
    # tool0 -> virtual TCP
    #
    # 実機:
    #   tool0 -> tool0_controller
    #   [0, 0, 0.150] m
    #
    # ManiSkillでも同じTCP offsetを再現する。
    # ============================================================

    R_base_tool0 = quaternion_wxyz_to_rotation_matrix(base_tool0_quaternion_wxyz)

    tcp_offset_base = R_base_tool0 @ TCP_OFFSET_TOOL0

    virtual_tcp_position = (base_tool0_position + tcp_offset_base)

    # tool0 -> TCP に回転オフセットは無いので、
    # TCP姿勢はtool0姿勢と同一
    virtual_tcp_quaternion_xyzw = (base_tool0_quaternion_xyzw.copy())


    # -----------------------------
    # Sim2Real 誤差
    # -----------------------------

    position_error_vector = (virtual_tcp_position - real_tcp_position)

    position_error_m = np.linalg.norm(position_error_vector)

    orientation_error_deg = quaternion_angle_error_deg(
        virtual_tcp_quaternion_xyzw,
        real_tcp_quaternion_xyzw,
    )


    # -----------------------------
    # base_link -> base
    # -----------------------------
    base_link_base_position = to_list(T_base_link_base.p)
    base_link_base_quaternion_wxyz = to_list(T_base_link_base.q)

    base_link_base_quaternion_xyzw = [
        base_link_base_quaternion_wxyz[1],
        base_link_base_quaternion_wxyz[2],
        base_link_base_quaternion_wxyz[3],
        base_link_base_quaternion_wxyz[0],
    ]


    result = {
        "source": "maniskill",
        "tcp_link": tcp_link.name,

        "world_frame": {
            "frame_id": "maniskill_world",
            "position": {
                "x": float(world_position[0]),
                "y": float(world_position[1]),
                "z": float(world_position[2]),
            },
            "orientation_xyzw": {
                "x": float(world_quaternion_xyzw[0]),
                "y": float(world_quaternion_xyzw[1]),
                "z": float(world_quaternion_xyzw[2]),
                "w": float(world_quaternion_xyzw[3]),
            },
        },

        "base_link_frame": {
            "frame_id": "base_link",
            "position": {
                "x": float(base_link_position[0]),
                "y": float(base_link_position[1]),
                "z": float(base_link_position[2]),
            },
            "orientation_xyzw": {
                "x": float(base_link_quaternion_xyzw[0]),
                "y": float(base_link_quaternion_xyzw[1]),
                "z": float(base_link_quaternion_xyzw[2]),
                "w": float(base_link_quaternion_xyzw[3]),
            },
        },

        "base_to_tool0": {
            "frame_id": "base",
            "position": {
                "x": float(base_tool0_position[0]),
                "y": float(base_tool0_position[1]),
                "z": float(base_tool0_position[2]),
            },
            "orientation_xyzw": {
                "x": float(base_tool0_quaternion_xyzw[0]),
                "y": float(base_tool0_quaternion_xyzw[1]),
                "z": float(base_tool0_quaternion_xyzw[2]),
                "w": float(base_tool0_quaternion_xyzw[3]),
            },
        },

        "base_to_virtual_tcp": {
            "frame_id": "base",
            "tcp_offset_tool0_m": {
                "x": float(TCP_OFFSET_TOOL0[0]),
                "y": float(TCP_OFFSET_TOOL0[1]),
                "z": float(TCP_OFFSET_TOOL0[2]),
            },
            "position": {
                "x": float(virtual_tcp_position[0]),
                "y": float(virtual_tcp_position[1]),
                "z": float(virtual_tcp_position[2]),
            },
            "orientation_xyzw": {
                "x": float(virtual_tcp_quaternion_xyzw[0]),
                "y": float(virtual_tcp_quaternion_xyzw[1]),
                "z": float(virtual_tcp_quaternion_xyzw[2]),
                "w": float(virtual_tcp_quaternion_xyzw[3]),
            },
        },

        "real_tcp_reference": {
            "frame_id": "base",
            "position": {
                "x": float(real_tcp_position[0]),
                "y": float(real_tcp_position[1]),
                "z": float(real_tcp_position[2]),
            },
            "orientation_xyzw": {
                "x": float(real_tcp_quaternion_xyzw[0]),
                "y": float(real_tcp_quaternion_xyzw[1]),
                "z": float(real_tcp_quaternion_xyzw[2]),
                "w": float(real_tcp_quaternion_xyzw[3]),
            },
        },

        "sim2real_error": {
            "position_error_vector_m": {
                "x": float(position_error_vector[0]),
                "y": float(position_error_vector[1]),
                "z": float(position_error_vector[2]),
            },
            "position_error_m": float(position_error_m),
            "position_error_mm": float(position_error_m * 1000.0),
            "orientation_error_deg": float(orientation_error_deg),
        },

        "base_link_to_base": {
            "position": {
                "x": float(base_link_base_position[0]),
                "y": float(base_link_base_position[1]),
                "z": float(base_link_base_position[2]),
            },
            "orientation_xyzw": {
                "x": float(base_link_base_quaternion_xyzw[0]),
                "y": float(base_link_base_quaternion_xyzw[1]),
                "z": float(base_link_base_quaternion_xyzw[2]),
                "w": float(base_link_base_quaternion_xyzw[3]),
            },
        },
    }

    print("\n=== ManiSkill TCP Pose ===")
    print(json.dumps(result, indent=2))

    env.close()


if __name__ == "__main__":
    main()