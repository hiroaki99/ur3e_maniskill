#!/usr/bin/env python3

from pathlib import Path
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts_pickplace")

from pickplace_week2_common import build_env, load_config


EXPECTED = {
    "to_pregrasp": 1.0,
    "descend": 1.0,
    "grasp": 1.0,
    "stable_hold": 1.0,
    "lift": 1.0,
    "transport": 1.0,
    "descend_place": 1.0,
    "release": 0.0,
    "final_settle": 0.0,
}


def find_residual_env(env):
    current = env

    while current is not None:
        if hasattr(current, "_residual_gate"):
            return current

        current = getattr(current, "env", None)

    raise RuntimeError(
        "ResidualPickPlaceEnv could not be found."
    )


def main():
    config = load_config()

    env, _ = build_env(
        config=config,
        trajectory=Path(
            "trajectories/pick_place_reference_v1.json"
        ),
        sim_backend="physx_cpu",
        seed=3600,
        randomization_mode="both",
        cube_radius_mm=10.0,
        goal_radius_mm=10.0,
        fixed_cube_angle_deg=0.0,
        fixed_goal_angle_deg=180.0,
        apply_scaling=True,
    )

    try:
        residual_env = find_residual_env(env)

        all_passed = True

        print("=" * 70)
        print("Day15 Both Phase-Gating Test")
        print("=" * 70)

        for phase_index, phase_name in enumerate(
            residual_env.PHASE_NAMES
        ):
            if phase_name == "terminal":
                continue

            gate = float(
                residual_env._residual_gate(
                    phase_index
                )
            )

            expected = EXPECTED[phase_name]

            passed = gate == expected

            all_passed &= passed

            print(
                f"{phase_index:2d} "
                f"{phase_name:16s} "
                f"gate={gate:.0f} "
                f"expected={expected:.0f} "
                f"passed={passed}"
            )

        print()
        print("Gate:", all_passed)

        return 0 if all_passed else 1

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())