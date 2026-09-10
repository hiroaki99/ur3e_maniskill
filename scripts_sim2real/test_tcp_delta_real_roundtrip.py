#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import gymnasium as gym
import mani_skill.envs  # noqa: F401
import envs.ur3e_reach  # noqa: F401


from scripts_sim2real.maniskill_backend import (
    ManiSkillBackend,
)

from scripts_sim2real.ur3e_ros2_backend import (
    UR3eROS2Backend,
)

# 既にdry-runで検証済みの関数を再利用
from scripts_sim2real.test_tcp_delta_ik_dryrun import (
    JOINT_NAMES,
    compute_numerical_jacobian,
    get_pose_arrays,
    quaternion_angle_error_deg,
)


# ============================================================
# 実験設定
# ============================================================

# base座標系 X方向へ +1 mm
# TARGET_DELTA_POSITION = np.array(
#     [0.001, 0.0, 0.0],
#     dtype=np.float64,
# )

DAMPING = 1.0e-3


# Bridge側は0.05 radだが、
# このTCP試験ではさらに厳しくする。
MAX_IK_DQ_RAD = 0.01


COMMAND_DURATION_SEC = 2.0


# ------------------------------------------------------------
# 実機TCP評価基準
# ------------------------------------------------------------

# +1 mmに対する実移動誤差
FORWARD_POSITION_TOLERANCE_M = (
    0.0005
)

# TCP姿勢変化
FORWARD_ORIENTATION_TOLERANCE_DEG = (
    0.5
)

FORWARD_JOINT_TARGET_TOLERANCE_RAD = (
    0.001
)

# 想定外の大移動検知
MAX_REAL_TCP_DISPLACEMENT_M = (
    0.003
)

# 復帰後TCP
RETURN_TCP_TOLERANCE_M = (
    0.0005
)

# 復帰後joint
RETURN_JOINT_TOLERANCE_RAD = (
    0.001
)


def angular_difference(
    a,
    b,
):
    return float(
        np.arctan2(
            np.sin(a - b),
            np.cos(a - b),
        )
    )


def print_joint_dict(
    title,
    joints,
):
    print(
        f"\n=== {title} ==="
    )

    for name in JOINT_NAMES:
        print(
            f"{name}: "
            f"{joints[name]:+.9f} rad"
        )


def compute_ik_target(
    sim_backend,
    q_current,
    target_delta_position,
):
    # --------------------------------------------
    # ManiSkillを現在実機姿勢へ
    # --------------------------------------------

    sim_backend.command_joint_positions(
        q_current
    )

    p0, quat0, _ = (
        get_pose_arrays(
            sim_backend
        )
    )


    # --------------------------------------------
    # Numerical Jacobian
    # --------------------------------------------

    J = (
        compute_numerical_jacobian(
            sim_backend,
            q_current,
        )
    )


    print(
        "\n=== Jacobian ==="
    )

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print(J)

    condition_number = float(
        np.linalg.cond(J)
    )

    print(
        "\nJacobian condition number:",
        condition_number,
    )


    # --------------------------------------------
    # Cartesian target
    # --------------------------------------------

    desired_twist = np.zeros(
        6,
        dtype=np.float64,
    )

    desired_twist[0:3] = (
        target_delta_position
    )


    # --------------------------------------------
    # Damped Least Squares
    # --------------------------------------------

    regularized = (
        J @ J.T
        + (
            DAMPING ** 2
        )
        * np.eye(6)
    )

    dq = (
        J.T
        @ np.linalg.solve(
            regularized,
            desired_twist,
        )
    )


    max_abs_dq = float(
        np.max(
            np.abs(dq)
        )
    )


    if max_abs_dq > MAX_IK_DQ_RAD:
        raise RuntimeError(
            "IK joint delta exceeded "
            "the local safety limit: "
            f"{max_abs_dq:.6f} rad "
            f"> {MAX_IK_DQ_RAD:.6f} rad"
        )


    q_target = (
        q_current.copy()
    )

    for name, delta in zip(
        JOINT_NAMES,
        dq,
    ):
        q_target[name] += (
            float(delta)
        )


    # --------------------------------------------
    # ManiSkill内でtargetを検証
    # --------------------------------------------

    sim_backend.command_joint_positions(
        q_target
    )

    p1, quat1, _ = (
        get_pose_arrays(
            sim_backend
        )
    )

    predicted_delta = (
        p1 - p0
    )

    predicted_error = (
        predicted_delta
        - target_delta_position
    )

    predicted_orientation_change = (
        quaternion_angle_error_deg(
            quat0,
            quat1,
        )
    )


    return {
        "J": J,
        "condition_number":
            condition_number,

        "dq": dq,

        "q_target":
            q_target,

        "predicted_delta":
            predicted_delta,

        "predicted_error_m":
            float(
                np.linalg.norm(
                    predicted_error
                )
            ),

        "predicted_orientation_change_deg":
            predicted_orientation_change,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually send the +1 mm "
            "motion to the real UR3e. "
            "Without this flag the script "
            "is dry-run only."
        ),
    )

    parser.add_argument(
        "--dx-mm",
        type=float,
        default=1.0,
        help="TCP delta X in base frame [mm].",
    )

    parser.add_argument(
        "--dy-mm",
        type=float,
        default=0.0,
        help="TCP delta Y in base frame [mm].",
    )

    parser.add_argument(
        "--dz-mm",
        type=float,
        default=0.0,
        help="TCP delta Z in base frame [mm].",
    )

    args = parser.parse_args()

    target_delta_position = np.array(
        [
            args.dx_mm,
            args.dy_mm,
            args.dz_mm,
        ],
        dtype=np.float64,
    ) / 1000.0


    # ========================================================
    # 1. Backends
    # ========================================================

    real_backend = (
        UR3eROS2Backend(
            timeout=15.0
        )
    )

    print(
        "=== Bridge ==="
    )

    print(
        real_backend.ping()
    )

    print(
        "\n=== Waiting for ROS2 state ==="
    )

    real_backend.wait_until_ready(
        timeout_sec=10.0,
    )

    print(
        "ROS2 state: READY"
    )


    env = gym.make(
        "UR3eReach-v0",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )

    env.reset()

    sim_backend = (
        ManiSkillBackend(env)
    )


    # ========================================================
    # 2. 現在実機状態
    # ========================================================

    q_original = (
        real_backend
        .get_joint_positions()
    )

    tcp_original = (
        real_backend
        .get_tcp_pose()
    )


    print_joint_dict(
        "Original Real Joint Positions",
        q_original,
    )

    print(
        "\n=== Original Real TCP ==="
    )

    print(tcp_original)


    # ========================================================
    # 3. ManiSkill IK
    # ========================================================

    result = (
        compute_ik_target(
            sim_backend,
            q_original,
            target_delta_position,
        )
    )

    dq = result["dq"]

    q_target = (
        result["q_target"]
    )


    print(
        "\n=== IK Joint Delta ==="
    )

    for name, delta in zip(
        JOINT_NAMES,
        dq,
    ):
        print(
            f"{name}: "
            f"{delta:+.9f} rad"
        )


    print_joint_dict(
        "Canonical Target q",
        q_target,
    )


    print(
        "\n=== ManiSkill Prediction ==="
    )

    print(
        "requested delta [m]:",
        target_delta_position.tolist(),
    )

    print(
        "predicted delta [m]:",
        result[
            "predicted_delta"
        ].tolist(),
    )

    print(
        "predicted position error [mm]:",
        result[
            "predicted_error_m"
        ] * 1000.0,
    )

    print(
        "predicted orientation change [deg]:",
        result[
            "predicted_orientation_change_deg"
        ],
    )


    # ========================================================
    # 4. Bridgeに送信直前targetを問い合わせ
    #
    # ここでは実機は動かない。
    # ========================================================

    preview = (
        real_backend
        .preview_joint_positions(
            q_target
        )
    )


    print(
        "\n=== REAL COMMAND PREVIEW ==="
    )

    print(
        "NO MOTION HAS BEEN SENT."
    )


    print(
        "\nCurrent raw q:"
    )

    for name in JOINT_NAMES:
        print(
            f"{name}: "
            f"{preview['current_raw'][name]:+.9f}"
        )


    print(
        "\nResolved target q:"
    )

    for name in JOINT_NAMES:
        print(
            f"{name}: "
            f"{preview['resolved_targets'][name]:+.9f}"
        )


    print(
        "\nResolved delta q:"
    )

    for name in JOINT_NAMES:
        print(
            f"{name}: "
            f"{preview['delta_rad'][name]:+.9f}"
        )


    # ========================================================
    # 5. dry-runの場合はここで終了
    # ========================================================

    if not args.execute:

        print(
            "\n===================================="
        )

        print(
            "DRY RUN COMPLETE."
        )

        print(
            "The real robot was NOT moved."
        )

        print(
            "\nIf all values are reasonable, "
            "run again with:"
        )

        print(
            "\npython "
            "scripts_sim2real/"
            "test_tcp_delta_real_roundtrip.py "
            "--execute"
        )

        print(
            "===================================="
        )

        sim_backend.close()
        real_backend.close()

        return


    # ========================================================
    # 6. 明示確認
    # ========================================================

    print(
        "\n===================================="
    )

    print(
        "REAL ROBOT MOTION REQUESTED"
    )

    print(
        "Target TCP delta [mm]: "
        f"dx={args.dx_mm:+.3f}, "
        f"dy={args.dy_mm:+.3f}, "
        f"dz={args.dz_mm:+.3f}"
    )

    print(
        "Then return to original q."
    )

    print(
        "===================================="
    )


    confirmation = input(
        "\nType MOVE to continue: "
    )

    if confirmation != "MOVE":

        print(
            "Cancelled. "
            "No motion was sent."
        )

        sim_backend.close()
        real_backend.close()

        return


    # ========================================================
    # 7. 送信直前の状態再確認
    # ========================================================

    q_before = (
        real_backend
        .get_joint_positions()
    )

    tcp_before = (
        real_backend
        .get_tcp_pose()
    )


    max_joint_drift = max(
        abs(
            angular_difference(
                q_before[name],
                q_original[name],
            )
        )
        for name in JOINT_NAMES
    )


    if max_joint_drift > 0.002:

        raise RuntimeError(
            "Robot moved after the IK preview. "
            f"max joint drift="
            f"{max_joint_drift:.6f} rad. "
            "Aborting; rerun the script."
        )


    print(
        "\n=== FINAL PRE-MOTION CHECK ==="
    )

    print(
        "max joint drift since preview [rad]:",
        max_joint_drift,
    )


    # ========================================================
    # 8. +1 mm相当のjoint targetを実機へ
    # ========================================================

    print(
        "\n=== EXECUTING +1 mm TCP TEST ==="
    )


    forward_result = (
        real_backend
        .command_joint_positions(
            q_target,
            duration_sec=
                COMMAND_DURATION_SEC,
        )
    )


    print(
        forward_result
    )


    time.sleep(0.3)


    # ========================================================
    # 9. 実機TCP検証
    # ========================================================

    q_after = (
        real_backend
        .get_joint_positions()
    )

    tcp_after = (
        real_backend
        .get_tcp_pose()
    )


    print_joint_dict(
        "Real Joint Positions After Forward",
        q_after,
    )


    print(
        "\n=== REAL JOINT TRACKING RESULT ==="
    )

    actual_joint_delta = {}
    target_joint_error = {}

    for name in JOINT_NAMES:

        actual_joint_delta[name] = (
            angular_difference(
                q_after[name],
                q_before[name],
            )
        )

        target_joint_error[name] = (
            angular_difference(
                q_after[name],
                q_target[name],
            )
        )

        print(
            f"{name}: "
            f"actual_delta="
            f"{actual_joint_delta[name]:+.9f} rad, "
            f"target_error="
            f"{target_joint_error[name]:+.9f} rad"
        )


    max_target_joint_error = max(
        abs(value)
        for value
        in target_joint_error.values()
    )


    print(
        "\nmax target joint error [rad]:",
        max_target_joint_error,
    )


    p_before = np.asarray(
        tcp_before.position,
        dtype=np.float64,
    )

    p_after = np.asarray(
        tcp_after.position,
        dtype=np.float64,
    )


    actual_real_delta = (
        p_after
        - p_before
    )


    real_delta_error = (
        actual_real_delta
        - target_delta_position
    )


    real_position_error_m = float(
        np.linalg.norm(
            real_delta_error
        )
    )


    real_displacement_m = float(
        np.linalg.norm(
            actual_real_delta
        )
    )


    real_orientation_change_deg = (
        quaternion_angle_error_deg(
            tcp_before.orientation_xyzw,
            tcp_after.orientation_xyzw,
        )
    )


    print(
        "\n=== REAL TCP FORWARD RESULT ==="
    )

    print(
        "TCP before:"
    )

    print(
        tcp_before
    )

    print(
        "\nTCP after:"
    )

    print(
        tcp_after
    )

    print(
        "\nrequested delta [m]:",
        target_delta_position.tolist(),
    )

    print(
        "actual real delta [m]:",
        actual_real_delta.tolist(),
    )

    print(
        "delta error [mm]:",
        (
            real_delta_error
            * 1000.0
        ).tolist(),
    )

    print(
        "position error norm [mm]:",
        real_position_error_m
        * 1000.0,
    )

    print(
        "orientation change [deg]:",
        real_orientation_change_deg,
    )


    forward_position_pass = (
        real_position_error_m
        < FORWARD_POSITION_TOLERANCE_M
    )

    forward_orientation_pass = (
        real_orientation_change_deg
        < FORWARD_ORIENTATION_TOLERANCE_DEG
    )

    forward_joint_pass = (
        max_target_joint_error
        < FORWARD_JOINT_TARGET_TOLERANCE_RAD
    )

    displacement_safe = (
        real_displacement_m
        < MAX_REAL_TCP_DISPLACEMENT_M
    )


    print(
        "\nForward position:",
        "PASS"
        if forward_position_pass
        else "FAIL",
    )

    print(
        "Forward orientation:",
        "PASS"
        if forward_orientation_pass
        else "FAIL",
    )

    print(
        "Forward joints:",
        "PASS"
        if forward_joint_pass
        else "FAIL",
    )

    print(
        "Displacement safety:",
        "PASS"
        if displacement_safe
        else "FAIL",
    )


    # ========================================================
    # 10. 復帰前に少し待つ
    #
    # 異常動作を目視した場合はここでCtrl+Cし、
    # ロボット側のStopを使用する。
    # ========================================================

    print(
        "\nReturning to original joint pose "
        "in 2 seconds."
    )

    print(
        "If physical motion is abnormal, "
        "stop the robot instead of relying "
        "on automatic return."
    )

    time.sleep(2.0)


    # ========================================================
    # 11. 元のjoint姿勢へ復帰
    # ========================================================

    print(
        "\n=== RETURN TO ORIGINAL q ==="
    )


    return_result = (
        real_backend
        .command_joint_positions(
            q_original,
            duration_sec=
                COMMAND_DURATION_SEC,
        )
    )


    print(
        return_result
    )


    time.sleep(0.3)


    # ========================================================
    # 12. 復帰後検証
    # ========================================================

    q_final = (
        real_backend
        .get_joint_positions()
    )

    tcp_final = (
        real_backend
        .get_tcp_pose()
    )


    p_final = np.asarray(
        tcp_final.position,
        dtype=np.float64,
    )


    return_tcp_error_m = float(
        np.linalg.norm(
            p_final
            - p_before
        )
    )


    max_return_joint_error = max(
        abs(
            angular_difference(
                q_final[name],
                q_original[name],
            )
        )
        for name in JOINT_NAMES
    )


    print_joint_dict(
        "Final Joint Positions",
        q_final,
    )


    print(
        "\n=== Final TCP ==="
    )

    print(
        tcp_final
    )


    print(
        "\n=== RETURN RESULT ==="
    )

    print(
        "TCP return error [mm]:",
        return_tcp_error_m
        * 1000.0,
    )

    print(
        "max joint return error [rad]:",
        max_return_joint_error,
    )


    return_tcp_pass = (
        return_tcp_error_m
        < RETURN_TCP_TOLERANCE_M
    )

    return_joint_pass = (
        max_return_joint_error
        < RETURN_JOINT_TOLERANCE_RAD
    )


    print(
        "return TCP:",
        "PASS"
        if return_tcp_pass
        else "FAIL",
    )

    print(
        "return joints:",
        "PASS"
        if return_joint_pass
        else "FAIL",
    )


    overall = (
        forward_result.get(
            "success",
            False,
        )
        and forward_position_pass
        and forward_orientation_pass
        and forward_joint_pass
        and displacement_safe
        and return_result.get(
            "success",
            False,
        )
        and return_tcp_pass
        and return_joint_pass
    )


    print(
        "\n===================================="
    )

    print(
        "FINAL RESULT:",
        "PASS"
        if overall
        else "FAIL",
    )

    print(
        "===================================="
    )


    sim_backend.close()
    real_backend.close()


if __name__ == "__main__":
    main()