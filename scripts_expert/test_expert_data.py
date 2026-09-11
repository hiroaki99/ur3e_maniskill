#!/usr/bin/env python3

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "expert_gp"
    / "data"
)

NUM_EXPERTS = 12
EXPECTED_DIM = 2


def inspect_expert_file(path: Path):

    print(
        f"\n=== {path.name} ==="
    )

    if not path.exists():
        print("FAIL: file does not exist.")
        return False

    try:
        data = np.load(
            path,
            allow_pickle=True,
        )
    except Exception as exc:
        print(
            "FAIL: np.load failed:"
        )
        print(exc)
        return False

    print(
        "container type:",
        type(data),
    )

    print(
        "container dtype:",
        data.dtype,
    )

    print(
        "number of segments:",
        len(data),
    )

    if len(data) == 0:
        print(
            "FAIL: no segments."
        )
        return False

    lengths = []

    all_values = []

    for index, segment in enumerate(
        data
    ):
        try:
            segment_array = np.asarray(
                segment,
                dtype=np.float64,
            )
        except Exception as exc:
            print(
                f"FAIL: segment {index} "
                f"cannot be converted."
            )
            print(exc)
            return False

        if segment_array.ndim != 2:
            print(
                f"FAIL: segment {index} "
                f"ndim={segment_array.ndim}"
            )
            return False

        if (
            segment_array.shape[1]
            != EXPECTED_DIM
        ):
            print(
                f"FAIL: segment {index} "
                f"shape="
                f"{segment_array.shape}"
            )
            return False

        if not np.all(
            np.isfinite(
                segment_array
            )
        ):
            print(
                f"FAIL: segment {index} "
                "contains NaN or Inf."
            )
            return False

        lengths.append(
            segment_array.shape[0]
        )

        all_values.append(
            segment_array
        )

    merged = np.concatenate(
        all_values,
        axis=0,
    )

    print(
        "segment length min:",
        min(lengths),
    )

    print(
        "segment length max:",
        max(lengths),
    )

    print(
        "segment length mean:",
        float(np.mean(lengths)),
    )

    print(
        "value min:",
        merged.min(
            axis=0
        ),
    )

    print(
        "value max:",
        merged.max(
            axis=0
        ),
    )

    print(
        "PASS"
    )

    return True


def main():

    print(
        "=== Expert Data Validation ==="
    )

    print(
        "data directory:",
        DATA_DIR,
    )

    passed = 0

    for expert_id in range(
        NUM_EXPERTS
    ):
        path = (
            DATA_DIR
            / f"class{expert_id:03d}.npy"
        )

        if inspect_expert_file(
            path
        ):
            passed += 1

    print(
        "\n=============================="
    )

    print(
        f"TOTAL: "
        f"{passed}/{NUM_EXPERTS} PASS"
    )

    print(
        "=============================="
    )

    if passed != NUM_EXPERTS:
        raise RuntimeError(
            "Expert data validation failed."
        )


if __name__ == "__main__":
    main()