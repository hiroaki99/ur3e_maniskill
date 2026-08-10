#!/usr/bin/env python3
"""
Day 7:
Day 6で完成したReference Pick-and-Liftを複数回実行し、
Residual RLの基準軌道として十分安定しているか評価する。

評価項目
--------
- success rate
- cube lift height
- grasp command
- first bilateral contact force
- final hold contact steps
- final contact peak force
- runner error rate

このスクリプト自身はロボット制御を再実装しない。
Day 6で動作確認済みの day6_run_reference.py を
subprocessとして反復実行する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ==========================================================
# Path
# ==========================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]


DEFAULT_RUNNER = (
    REPO_ROOT
    / "scripts"
    / "day6_run_reference.py"
)


DEFAULT_TRAJECTORY = (
    REPO_ROOT
    / "trajectories"
    / "day6_pick_lift_reference_v2.json"
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "reports"
    / "day7_reference_eval"
)


# ==========================================================
# Arguments
# ==========================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Reference Pick-and-Liftの実行回数",
    )

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
        help="Day 6で生成したReference trajectory",
    )

    parser.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_RUNNER,
        help="Day 6の実行スクリプト",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--success-rate-threshold",
        type=float,
        default=0.80,
        help="Day 7合格成功率",
    )

    parser.add_argument(
        "--lift-height-threshold",
        type=float,
        default=0.05,
        help="成功とみなす最低Cube Lift高さ[m]",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="1試行あたりの最大実行時間[sec]",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="runnerが例外終了したら即停止する",
    )

    return parser.parse_args()


# ==========================================================
# Utility
# ==========================================================

def safe_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:

        result = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def safe_int(
    value: Any,
) -> int | None:

    if value is None:
        return None

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def mean_or_none(
    values,
):

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(
        statistics.mean(values)
    )


def std_or_none(
    values,
):

    values = [
        value
        for value in values
        if value is not None
    ]

    if len(values) < 2:
        return 0.0 if values else None

    return float(
        statistics.stdev(values)
    )


def min_or_none(
    values,
):

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(min(values))


def max_or_none(
    values,
):

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(max(values))


def get_report_value(
    report: dict,
    *names,
    default=None,
):
    """
    Day 6 reportのキー名が多少変わっても
    評価スクリプトを壊れにくくする。
    """

    for name in names:

        if name in report:
            return report[name]

    return default


# ==========================================================
# Trial execution
# ==========================================================

def run_one_trial(
    trial_index: int,
    runner: Path,
    trajectory: Path,
    output_dir: Path,
    timeout: float,
) -> dict:

    trial_number = (
        trial_index + 1
    )

    report_path = (
        output_dir
        / f"trial_{trial_number:03d}.json"
    )

    log_path = (
        output_dir
        / f"trial_{trial_number:03d}.log"
    )

    command = [
        sys.executable,
        str(runner),
        "--trajectory",
        str(trajectory),
        "--output",
        str(report_path),
    ]

    print()
    print(
        f"[{trial_number:03d}] "
        "Running reference..."
    )

    start_time = (
        time.perf_counter()
    )

    timed_out = False

    try:

        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return_code = (
            process.returncode
        )

        stdout = (
            process.stdout
            or ""
        )

        stderr = (
            process.stderr
            or ""
        )

    except subprocess.TimeoutExpired as exc:

        timed_out = True

        return_code = -999

        stdout = (
            exc.stdout
            if isinstance(
                exc.stdout,
                str,
            )
            else ""
        )

        stderr = (
            exc.stderr
            if isinstance(
                exc.stderr,
                str,
            )
            else ""
        )

        stderr += (
            "\n"
            f"TIMEOUT after "
            f"{timeout} sec"
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ------------------------------------------------------
    # Log
    # ------------------------------------------------------

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "COMMAND:\n"
        )

        file.write(
            " ".join(command)
        )

        file.write(
            "\n\n"
        )

        file.write(
            "STDOUT:\n"
        )

        file.write(
            stdout
        )

        file.write(
            "\n\nSTDERR:\n"
        )

        file.write(
            stderr
        )

    # ------------------------------------------------------
    # JSON Report
    # ------------------------------------------------------

    report = {}

    report_read_error = None

    if report_path.exists():

        try:

            with report_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                report = json.load(
                    file
                )

        except Exception as exc:

            report_read_error = (
                repr(exc)
            )

    else:

        report_read_error = (
            "report file was not created"
        )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    success = bool(
        get_report_value(
            report,
            "success",
            default=False,
        )
    )

    cube_lift = safe_float(
        get_report_value(
            report,
            "cube_lift_m",
        )
    )

    grasp_command = safe_float(
        get_report_value(
            report,
            "grasp_command",
        )
    )

    first_left_force = safe_float(
        get_report_value(
            report,
            "first_both_contact_left_force_n",
        )
    )

    first_right_force = safe_float(
        get_report_value(
            report,
            "first_both_contact_right_force_n",
        )
    )

    final_peak_left = safe_float(
        get_report_value(
            report,
            "final_peak_left_force_n",
        )
    )

    final_peak_right = safe_float(
        get_report_value(
            report,
            "final_peak_right_force_n",
        )
    )

    final_hold_steps = safe_int(
        get_report_value(
            report,
            "final_hold_steps",
        )
    )

    final_both_steps = safe_int(
        get_report_value(
            report,
            "final_both_contact_steps",
        )
    )

    # ------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------

    print(
        f"[{trial_number:03d}] "
        f"returncode={return_code} "
        f"success={success} "
        f"lift="
        f"{cube_lift if cube_lift is not None else 'N/A'} "
        f"time={elapsed:.2f}s"
    )

    return {
        "trial":
            trial_number,

        "return_code":
            return_code,

        "timed_out":
            timed_out,

        "success":
            success,

        "cube_lift_m":
            cube_lift,

        "grasp_command":
            grasp_command,

        "first_both_left_force_n":
            first_left_force,

        "first_both_right_force_n":
            first_right_force,

        "final_peak_left_force_n":
            final_peak_left,

        "final_peak_right_force_n":
            final_peak_right,

        "final_hold_steps":
            final_hold_steps,

        "final_both_contact_steps":
            final_both_steps,

        "elapsed_sec":
            float(elapsed),

        "report_path":
            str(report_path),

        "log_path":
            str(log_path),

        "report_read_error":
            report_read_error,
    }


# ==========================================================
# CSV
# ==========================================================

def save_trials_csv(
    trials: list[dict],
    path: Path,
):

    if not trials:
        return

    fieldnames = list(
        trials[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
        )

        writer.writeheader()

        writer.writerows(
            trials
        )


# ==========================================================
# Main
# ==========================================================

def main() -> int:

    args = parse_args()

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    if args.runs <= 0:

        raise ValueError(
            "--runs は1以上にしてください"
        )

    if not (
        0.0
        <= args.success_rate_threshold
        <= 1.0
    ):

        raise ValueError(
            "--success-rate-threshold は"
            "0～1で指定してください"
        )

    if not args.runner.exists():

        raise FileNotFoundError(
            f"runnerがありません: "
            f"{args.runner}"
        )

    if not args.trajectory.exists():

        raise FileNotFoundError(
            f"trajectoryがありません: "
            f"{args.trajectory}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Day 7 Reference Evaluation")
    print("=" * 72)

    print(
        "runner     :",
        args.runner,
    )

    print(
        "trajectory :",
        args.trajectory,
    )

    print(
        "runs       :",
        args.runs,
    )

    print(
        "gate       :",
        args.success_rate_threshold,
    )

    # ------------------------------------------------------
    # Trials
    # ------------------------------------------------------

    trials = []

    for trial_index in range(
        args.runs
    ):

        result = run_one_trial(
            trial_index=trial_index,
            runner=args.runner,
            trajectory=args.trajectory,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )

        trials.append(
            result
        )

        if (
            args.stop_on_error
            and result[
                "return_code"
            ] not in (
                0,
                1,
            )
        ):

            print(
                "[STOP] runner error detected"
            )

            break

    # ------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------

    executed_runs = len(
        trials
    )

    success_count = sum(
        bool(
            trial["success"]
        )
        for trial in trials
    )

    failure_count = (
        executed_runs
        - success_count
    )

    success_rate = (
        success_count
        / executed_runs
        if executed_runs > 0
        else 0.0
    )

    runner_error_count = sum(
        trial[
            "return_code"
        ] not in (
            0,
            1,
        )
        for trial in trials
    )

    timeout_count = sum(
        bool(
            trial["timed_out"]
        )
        for trial in trials
    )

    lift_values = [
        trial[
            "cube_lift_m"
        ]
        for trial in trials
    ]

    grasp_commands = [
        trial[
            "grasp_command"
        ]
        for trial in trials
    ]

    left_first_forces = [
        trial[
            "first_both_left_force_n"
        ]
        for trial in trials
    ]

    right_first_forces = [
        trial[
            "first_both_right_force_n"
        ]
        for trial in trials
    ]

    final_left_forces = [
        trial[
            "final_peak_left_force_n"
        ]
        for trial in trials
    ]

    final_right_forces = [
        trial[
            "final_peak_right_force_n"
        ]
        for trial in trials
    ]

    elapsed_values = [
        trial[
            "elapsed_sec"
        ]
        for trial in trials
    ]

    # Lift条件を独立にも確認
    lift_pass_count = sum(
        (
            value is not None
            and value
            >= args.lift_height_threshold
        )
        for value in lift_values
    )

    lift_pass_rate = (
        lift_pass_count
        / executed_runs
        if executed_runs > 0
        else 0.0
    )

    gate_passed = (
        success_rate
        >= args.success_rate_threshold
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = {
        "runs_requested":
            int(
                args.runs
            ),

        "runs_executed":
            executed_runs,

        "success_count":
            success_count,

        "failure_count":
            failure_count,

        "success_rate":
            float(
                success_rate
            ),

        "success_rate_threshold":
            float(
                args.success_rate_threshold
            ),

        "gate_passed":
            bool(
                gate_passed
            ),

        "runner_error_count":
            runner_error_count,

        "timeout_count":
            timeout_count,

        "lift_height_threshold_m":
            float(
                args.lift_height_threshold
            ),

        "lift_pass_count":
            lift_pass_count,

        "lift_pass_rate":
            float(
                lift_pass_rate
            ),

        "cube_lift_m": {
            "mean":
                mean_or_none(
                    lift_values
                ),

            "std":
                std_or_none(
                    lift_values
                ),

            "min":
                min_or_none(
                    lift_values
                ),

            "max":
                max_or_none(
                    lift_values
                ),
        },

        "grasp_command": {
            "mean":
                mean_or_none(
                    grasp_commands
                ),

            "std":
                std_or_none(
                    grasp_commands
                ),

            "min":
                min_or_none(
                    grasp_commands
                ),

            "max":
                max_or_none(
                    grasp_commands
                ),
        },

        "first_bilateral_contact_force_n": {
            "left_mean":
                mean_or_none(
                    left_first_forces
                ),

            "right_mean":
                mean_or_none(
                    right_first_forces
                ),

            "left_max":
                max_or_none(
                    left_first_forces
                ),

            "right_max":
                max_or_none(
                    right_first_forces
                ),
        },

        "final_peak_contact_force_n": {
            "left_mean":
                mean_or_none(
                    final_left_forces
                ),

            "right_mean":
                mean_or_none(
                    final_right_forces
                ),

            "left_max":
                max_or_none(
                    final_left_forces
                ),

            "right_max":
                max_or_none(
                    final_right_forces
                ),
        },

        "elapsed_sec": {
            "mean":
                mean_or_none(
                    elapsed_values
                ),

            "total":
                float(
                    sum(
                        elapsed_values
                    )
                ),
        },

        "trajectory":
            str(
                args.trajectory
            ),

        "runner":
            str(
                args.runner
            ),
    }

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    csv_path = (
        args.output_dir
        / "trials.csv"
    )

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    save_trials_csv(
        trials,
        csv_path,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------
    # Terminal Result
    # ------------------------------------------------------

    print()
    print("=" * 72)
    print("Day 7 Result")
    print("=" * 72)

    print(
        "executed runs :",
        executed_runs,
    )

    print(
        "success       :",
        f"{success_count}/"
        f"{executed_runs}",
    )

    print(
        "success rate  :",
        f"{success_rate * 100:.2f} %",
    )

    print(
        "lift pass     :",
        f"{lift_pass_count}/"
        f"{executed_runs}",
    )

    print(
        "lift mean     :",
        summary[
            "cube_lift_m"
        ]["mean"],
        "m",
    )

    print(
        "lift min      :",
        summary[
            "cube_lift_m"
        ]["min"],
        "m",
    )

    print(
        "grasp cmd mean:",
        summary[
            "grasp_command"
        ]["mean"],
    )

    print(
        "runner errors :",
        runner_error_count,
    )

    print(
        "timeouts      :",
        timeout_count,
    )

    print(
        "gate          :",
        (
            "PASS"
            if gate_passed
            else "FAIL"
        ),
    )

    print(
        "trials CSV    :",
        csv_path,
    )

    print(
        "summary       :",
        summary_path,
    )

    return (
        0
        if gate_passed
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )