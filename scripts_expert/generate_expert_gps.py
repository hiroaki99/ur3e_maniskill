#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate 12 Yaginuma expert GP trajectories from class000.npy ... class011.npy.

Pipeline:
    load_data()
      -> get_mean_trajectory()
      -> upsampling()
      -> decoder()

Outputs:
    reports/expert_gp/
        expert_00.npz
        ...
        expert_11.npz
        summary.csv
        metadata.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Trajectory_Generator / GaussianProcessMultiDim が
# matplotlib.pyplot をimportするため、GUIなし環境でも動くようにする。
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


NUM_EXPERTS = 12
DIM = 2
DEFAULT_SEED = 0


def parse_args():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Generate 12 Yaginuma expert GP trajectories."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "expert_gp" / "data",
        help="Directory containing class000.npy ... class011.npy.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "reports" / "expert_gp",
        help="Directory where generated expert trajectories are saved.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "NumPy random seed. "
            "The original GaussianProcess.GP.predict() injects "
            "NumPy random noise."
        ),
    )

    return parser.parse_args()


def validate_input_files(data_dir: Path):
    missing = []

    for expert_id in range(NUM_EXPERTS):
        path = data_dir / f"class{expert_id:03d}.npy"

        if not path.is_file():
            missing.append(path)

    if missing:
        message = "\n".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            "The following expert data files are missing:\n"
            + message
        )


def import_trajectory_generator(
    project_root: Path,
):
    expert_gp_dir = (
        project_root / "expert_gp"
    )

    if str(expert_gp_dir) not in sys.path:
        sys.path.insert(
            0,
            str(expert_gp_dir),
        )

    try:
        from Trajectory_Generator import (
            Trajectory_Generator,
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to import "
            "Trajectory_Generator. "
            "Confirm that Step 3 "
            "import test passes."
        ) from exc

    return Trajectory_Generator


def convert_expert(
    decoded_expert,
    expert_id: int,
):
    """
    Original format:

        decoded_expert[axis][point]
            = [mean, sigma]

    Convert into:

        mean  : (N, 2)
        sigma : (N, 2)
    """

    if len(decoded_expert) != DIM:
        raise RuntimeError(
            f"Expert {expert_id:02d}: "
            f"expected {DIM} axes, "
            f"got {len(decoded_expert)}."
        )

    axis_arrays = []

    for axis_id in range(DIM):

        arr = np.asarray(
            decoded_expert[axis_id],
            dtype=np.float64,
        )

        if (
            arr.ndim != 2
            or arr.shape[1] != 2
        ):
            raise RuntimeError(
                f"Expert {expert_id:02d}, "
                f"axis {axis_id}: "
                f"expected shape (N, 2), "
                f"got {arr.shape}."
            )

        if not np.all(
            np.isfinite(arr)
        ):
            raise RuntimeError(
                f"Expert {expert_id:02d}, "
                f"axis {axis_id}: "
                "NaN or Inf detected."
            )

        axis_arrays.append(
            arr
        )

    lengths = [
        arr.shape[0]
        for arr in axis_arrays
    ]

    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"Expert {expert_id:02d}: "
            f"axis lengths differ: "
            f"{lengths}."
        )

    mean = np.column_stack(
        [
            axis_arrays[axis][:, 0]
            for axis in range(DIM)
        ]
    )

    sigma = np.column_stack(
        [
            axis_arrays[axis][:, 1]
            for axis in range(DIM)
        ]
    )

    return mean, sigma


def save_expert(
    output_dir: Path,
    expert_id: int,
    mean: np.ndarray,
    sigma: np.ndarray,
):
    output_path = (
        output_dir
        / f"expert_{expert_id:02d}.npz"
    )

    np.savez_compressed(
        output_path,
        mean=mean,
        sigma=sigma,
        axis0_mean=mean[:, 0],
        axis1_mean=mean[:, 1],
        axis0_sigma=sigma[:, 0],
        axis1_sigma=sigma[:, 1],
    )

    return output_path


def main():
    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    data_dir = (
        args.data_dir.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    print(
        "=== Yaginuma 12 Expert "
        "GP Generation ==="
    )

    print(
        "project root :",
        project_root,
    )

    print(
        "data dir     :",
        data_dir,
    )

    print(
        "output dir   :",
        output_dir,
    )

    print(
        "seed         :",
        args.seed,
    )


    validate_input_files(
        data_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Original GaussianProcess.GP.predict()
    # uses np.random.normal().
    #
    # GP本体は変更せず、
    # ラッパー側だけseed固定する。
    np.random.seed(
        args.seed
    )


    Trajectory_Generator = (
        import_trajectory_generator(
            project_root
        )
    )


    # 元コードは
    #
    # self.dir + "class%03d.npy"
    #
    # と文字列連結するため、
    # 最後の "/" が必要。
    dirname = (
        str(data_dir)
        + os.sep
    )


    tg = Trajectory_Generator(
        dim=DIM,
        numclass=NUM_EXPERTS,
        dirname=dirname,
    )


    print(
        "\nGenerating expert "
        "trajectories..."
    )

    print(
        "  load_data()"
    )

    print(
        "  -> get_mean_trajectory()"
    )

    print(
        "  -> upsampling("
        "multiple=100)"
    )

    print(
        "  -> decoder()"
    )


    decoded_experts = (
        tg.generate_expert_trajectory()
    )


    if (
        len(decoded_experts)
        != NUM_EXPERTS
    ):
        raise RuntimeError(
            f"Expected "
            f"{NUM_EXPERTS} experts, "
            f"got "
            f"{len(decoded_experts)}."
        )


    summary_rows = []

    negative_sigma_count = 0


    print(
        "\n=== Generated Experts ==="
    )


    for (
        expert_id,
        decoded_expert,
    ) in enumerate(
        decoded_experts
    ):

        mean, sigma = (
            convert_expert(
                decoded_expert,
                expert_id,
            )
        )


        output_path = (
            save_expert(
                output_dir,
                expert_id,
                mean,
                sigma,
            )
        )


        min_sigma = float(
            np.min(sigma)
        )

        max_sigma = float(
            np.max(sigma)
        )


        if (
            min_sigma
            < -1.0e-10
        ):
            negative_sigma_count += 1


        start = mean[0]
        end = mean[-1]


        summary_rows.append(
            {
                "expert_id":
                    expert_id,

                "num_points":
                    mean.shape[0],

                "start_axis0":
                    float(start[0]),

                "start_axis1":
                    float(start[1]),

                "end_axis0":
                    float(end[0]),

                "end_axis1":
                    float(end[1]),

                "mean_min_axis0":
                    float(
                        np.min(
                            mean[:, 0]
                        )
                    ),

                "mean_max_axis0":
                    float(
                        np.max(
                            mean[:, 0]
                        )
                    ),

                "mean_min_axis1":
                    float(
                        np.min(
                            mean[:, 1]
                        )
                    ),

                "mean_max_axis1":
                    float(
                        np.max(
                            mean[:, 1]
                        )
                    ),

                "sigma_min":
                    min_sigma,

                "sigma_max":
                    max_sigma,

                "file":
                    output_path.name,
            }
        )


        print(
            f"Expert "
            f"{expert_id:02d}: "
            f"points="
            f"{mean.shape[0]}, "
            f"start=["
            f"{start[0]:+.6f}, "
            f"{start[1]:+.6f}], "
            f"end=["
            f"{end[0]:+.6f}, "
            f"{end[1]:+.6f}], "
            f"sigma=["
            f"{min_sigma:.6e}, "
            f"{max_sigma:.6e}] "
            f"-> "
            f"{output_path.name}"
        )


    summary_path = (
        output_dir
        / "summary.csv"
    )


    fieldnames = list(
        summary_rows[0].keys()
    )


    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )


    metadata_path = (
        output_dir
        / "metadata.txt"
    )


    metadata_path.write_text(
        "\n".join(
            [
                "Yaginuma 12 Expert "
                "GP generation",

                f"num_experts="
                f"{NUM_EXPERTS}",

                f"dim={DIM}",

                f"seed={args.seed}",

                f"data_dir="
                f"{data_dir}",

                "gp_prediction_points="
                "1000",

                "upsampling_multiple="
                "100",

                "decoded_axes=2",

                (
                    "note=Original "
                    "GaussianProcess."
                    "GP.predict() "
                    "uses NumPy "
                    "random noise; "
                    "seed fixed only "
                    "in this wrapper."
                ),
            ]
        )
        + "\n",

        encoding="utf-8",
    )


    print(
        "\n=== Validation Summary ==="
    )

    print(
        "experts generated :",
        f"{len(summary_rows)}"
        f"/{NUM_EXPERTS}",
    )

    print(
        "summary            :",
        summary_path,
    )

    print(
        "metadata           :",
        metadata_path,
    )


    if negative_sigma_count > 0:

        print(
            "WARNING: "
            f"{negative_sigma_count} "
            "expert(s) contained "
            "sigma < 0."
        )

    else:

        print(
            "sigma check        : "
            "no negative values "
            "detected"
        )


    print(
        "\n=============================="
    )

    print(
        "EXPERT GP GENERATION: PASS"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()