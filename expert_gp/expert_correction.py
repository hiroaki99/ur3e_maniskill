#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict Yaginuma-style GP begin-correction utilities.

This module reproduces residual_ddpg.py::correct_traj_begin() together with
Trajectory_Generator.py::dummy_trajectory() and merge_trajectory().

Important original behavior
---------------------------
For the generated expert arrays:

    d_data[ep][axis]           -> 99,900 points
    d_data[ep][axis][::100]    -> 999 points

correct_traj_begin() passes:

    segm_len = len(d_data[ep][axis][::100]) = 999

to dummy_trajectory().  The original dummy trajectory length is:

    (segm_len - 1) * 100 = 99,800 points

and dummy[::100] therefore contains 998 points.

merge_trajectory() uses Python zip(), so the 999-point expert side and
998-point dummy side are truncated to the shorter length: 998 points.

That one-point truncation is intentional because it is the behavior of the
original implementation.

This module contains no ManiSkill or ROS2 command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ORIGINAL_STRIDE = 100
ORIGINAL_BEGIN_SIGMA_SCALE = 1000.0
ORIGINAL_END_SIGMA_SCALE = 2000.0


@dataclass(frozen=True)
class CorrectedExpert:
    expert_id: int
    requested_start_logical: np.ndarray
    prototype_mean_sliced: np.ndarray
    prototype_sigma_sliced: np.ndarray
    dummy_mean_sliced: np.ndarray
    dummy_sigma_sliced: np.ndarray
    corrected_mean: np.ndarray
    corrected_sigma: np.ndarray
    expert_sliced_length: int
    dummy_sliced_length: int
    corrected_length: int
    source_stride: int
    sigma_scale: float


@dataclass(frozen=True)
class CorrectedExpertEnd:
    ddpg_action: np.ndarray
    requested_end_logical: np.ndarray
    input_mean: np.ndarray
    input_sigma: np.ndarray
    dummy_mean_sliced: np.ndarray
    dummy_sigma_sliced: np.ndarray
    corrected_mean: np.ndarray
    corrected_sigma: np.ndarray
    input_length: int
    dummy_sliced_length: int
    corrected_length: int
    source_stride: int
    sigma_scale: float


def _validate_mean_sigma(
    mean: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if mean.ndim != 2 or mean.shape[1] != 2:
        raise ValueError(
            f"mean must have shape (N,2), got {mean.shape}."
        )

    if sigma.shape != mean.shape:
        raise ValueError(
            "sigma must have the same shape as mean: "
            f"mean={mean.shape}, sigma={sigma.shape}."
        )

    if mean.shape[0] < 2:
        raise ValueError(
            "trajectory must contain at least two points."
        )

    if not np.all(np.isfinite(mean)):
        raise ValueError("mean contains NaN/Inf.")

    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma contains NaN/Inf.")

    if np.any(sigma < 0.0):
        raise ValueError("sigma contains negative values.")

    return mean, sigma


def load_expert_mean_sigma(
    expert_dir: str | Path,
    expert_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    expert_id = int(expert_id)

    if not 0 <= expert_id <= 11:
        raise ValueError(
            f"expert_id must be in [0,11], got {expert_id}."
        )

    path = (
        Path(expert_dir).expanduser().resolve()
        / f"expert_{expert_id:02d}.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Expert file not found: {path}"
        )

    with np.load(path) as data:
        if "mean" not in data:
            raise KeyError(
                f"{path} does not contain key 'mean'."
            )
        if "sigma" not in data:
            raise KeyError(
                f"{path} does not contain key 'sigma'."
            )

        mean = np.asarray(
            data["mean"],
            dtype=np.float64,
        )
        sigma = np.asarray(
            data["sigma"],
            dtype=np.float64,
        )

    return _validate_mean_sigma(mean, sigma)


def make_original_begin_dummy_sliced(
    current_logical_xz,
    *,
    segm_len: int,
    source_stride: int = ORIGINAL_STRIDE,
    sigma_scale: float = ORIGINAL_BEGIN_SIGMA_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce:

        rng, begin_pos_traj = dummy_trajectory(pos, segm_len)
        begin_pos_traj[axis][::100]

    from the original code.
    """
    current = np.asarray(
        current_logical_xz,
        dtype=np.float64,
    ).reshape(-1)

    if current.shape != (2,):
        raise ValueError(
            "current_logical_xz must have shape (2,)."
        )

    if not np.all(np.isfinite(current)):
        raise ValueError(
            "current_logical_xz contains NaN/Inf."
        )

    segm_len = int(segm_len)
    source_stride = int(source_stride)
    sigma_scale = float(sigma_scale)

    if segm_len < 2:
        raise ValueError(
            "segm_len must be >= 2."
        )

    if source_stride <= 0:
        raise ValueError(
            "source_stride must be > 0."
        )

    if (
        not np.isfinite(sigma_scale)
        or sigma_scale <= 0.0
    ):
        raise ValueError(
            "sigma_scale must be finite and > 0."
        )

    # Exact structure of Trajectory_Generator.dummy_trajectory():
    #
    # rng = [
    #   segm_len * i / ((segm_len - 1) * 100)
    #   for i in range((segm_len - 1) * 100)
    # ]
    #
    # We retain `source_stride` as an explicit parameter, but strict
    # reproduction uses source_stride=100.
    full_length = (
        (segm_len - 1)
        * ORIGINAL_STRIDE
    )

    i = np.arange(
        full_length,
        dtype=np.float64,
    )

    rng = (
        segm_len
        * i
        / float(full_length)
    )

    dummy_sigma_full = (
        rng
        / sigma_scale
    )

    dummy_sigma_sliced = (
        dummy_sigma_full[
            ::source_stride
        ]
    )

    dummy_mean_sliced = np.repeat(
        current[None, :],
        dummy_sigma_sliced.shape[0],
        axis=0,
    )

    dummy_sigma_sliced_2d = np.repeat(
        dummy_sigma_sliced[:, None],
        2,
        axis=1,
    )

    return (
        dummy_mean_sliced,
        dummy_sigma_sliced_2d,
    )



def make_original_end_dummy_sliced(
    end_logical_xz,
    *,
    segm_len: int,
    source_stride: int = ORIGINAL_STRIDE,
    sigma_scale: float = ORIGINAL_END_SIGMA_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strictly reproduce:

        rng, end_params = end_dummy_trajectory(
            end_point,
            len(traj_x),
        )
        end_params[axis][::100]

    from Yaginuma's original code.

    Original end_dummy_trajectory():

        rng = [
            segm_len * i / ((segm_len - 1) * 100)
            for i in range((segm_len - 1) * 100)
        ]

        tmp = [
            [pos[data], r / sigma_scale]
            for r in rng
        ]

        tmp = tmp[::-1]

    Important:
    - reverse the FULL dummy list first
    - only then apply [::100]
    - production correct_traj_end() does not override sigma_scale, so the
      original default 2000.0 is used
    """
    endpoint = np.asarray(
        end_logical_xz,
        dtype=np.float64,
    ).reshape(-1)

    if endpoint.shape != (2,):
        raise ValueError(
            "end_logical_xz must have shape (2,)."
        )

    if not np.all(np.isfinite(endpoint)):
        raise ValueError(
            "end_logical_xz contains NaN/Inf."
        )

    segm_len = int(segm_len)
    source_stride = int(source_stride)
    sigma_scale = float(sigma_scale)

    if segm_len < 2:
        raise ValueError(
            "segm_len must be >= 2."
        )

    if source_stride <= 0:
        raise ValueError(
            "source_stride must be > 0."
        )

    if (
        not np.isfinite(sigma_scale)
        or sigma_scale <= 0.0
    ):
        raise ValueError(
            "sigma_scale must be finite and > 0."
        )

    full_length = (
        (segm_len - 1)
        * ORIGINAL_STRIDE
    )

    i = np.arange(
        full_length,
        dtype=np.float64,
    )

    rng = (
        segm_len
        * i
        / float(full_length)
    )

    dummy_sigma_full = (
        rng
        / sigma_scale
    )

    # Exact original order:
    #     tmp = tmp[::-1]
    # and only afterwards:
    #     end_params[axis][::100]
    dummy_sigma_sliced = (
        dummy_sigma_full[
            ::-1
        ][
            ::source_stride
        ]
    )

    dummy_mean_sliced = np.repeat(
        endpoint[None, :],
        dummy_sigma_sliced.shape[0],
        axis=0,
    )

    dummy_sigma_sliced_2d = np.repeat(
        dummy_sigma_sliced[:, None],
        2,
        axis=1,
    )

    return (
        dummy_mean_sliced,
        dummy_sigma_sliced_2d,
    )


def merge_gaussian_trajectory_zip_semantics(
    mean1: np.ndarray,
    sigma1: np.ndarray,
    mean2: np.ndarray,
    sigma2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce merge_trajectory() including Python zip() truncation.
    """
    mean1 = np.asarray(mean1, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    mean2 = np.asarray(mean2, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    if mean1.shape != sigma1.shape:
        raise ValueError(
            "mean1/sigma1 shape mismatch."
        )

    if mean2.shape != sigma2.shape:
        raise ValueError(
            "mean2/sigma2 shape mismatch."
        )

    if (
        mean1.ndim != 2
        or mean2.ndim != 2
        or mean1.shape[1] != 2
        or mean2.shape[1] != 2
    ):
        raise ValueError(
            "All mean/sigma arrays must be (N,2)."
        )

    length = min(
        mean1.shape[0],
        mean2.shape[0],
    )

    m1 = mean1[:length]
    s1 = sigma1[:length]
    m2 = mean2[:length]
    s2 = sigma2[:length]

    denominator = s1 + s2

    if np.any(denominator <= 0.0):
        bad = np.argwhere(
            denominator <= 0.0
        )[0]
        raise RuntimeError(
            "Non-positive fusion denominator at "
            f"index={tuple(int(x) for x in bad)}."
        )

    merged_mean = (
        s2 * m1
        + s1 * m2
    ) / denominator

    merged_sigma = (
        s1 * s2
    ) / denominator

    return (
        merged_mean,
        merged_sigma,
    )


def correct_expert_begin(
    *,
    expert_id: int,
    mean: np.ndarray,
    sigma: np.ndarray,
    current_logical_xz,
    source_stride: int = ORIGINAL_STRIDE,
    sigma_scale: float = ORIGINAL_BEGIN_SIGMA_SCALE,
) -> CorrectedExpert:
    """
    Strict reproduction of residual_ddpg.py::correct_traj_begin().

    Original:
        expert = d_data[ep][axis][::100]
        segm_len = len(expert)
        dummy = dummy_trajectory(pos, segm_len)[axis][::100]
        merge_trajectory(expert, dummy)

    Because expert and dummy lengths differ by one point, the merged result
    has the dummy-side length due to zip() truncation.
    """
    mean, sigma = _validate_mean_sigma(
        mean,
        sigma,
    )

    source_stride = int(source_stride)

    if source_stride <= 0:
        raise ValueError(
            "source_stride must be > 0."
        )

    expert_mean_sliced = (
        mean[::source_stride]
        .copy()
    )
    expert_sigma_sliced = (
        sigma[::source_stride]
        .copy()
    )

    segm_len = int(
        expert_mean_sliced.shape[0]
    )

    (
        dummy_mean_sliced,
        dummy_sigma_sliced,
    ) = make_original_begin_dummy_sliced(
        current_logical_xz,
        segm_len=segm_len,
        source_stride=source_stride,
        sigma_scale=sigma_scale,
    )

    (
        corrected_mean,
        corrected_sigma,
    ) = merge_gaussian_trajectory_zip_semantics(
        expert_mean_sliced,
        expert_sigma_sliced,
        dummy_mean_sliced,
        dummy_sigma_sliced,
    )

    return CorrectedExpert(
        expert_id=int(expert_id),
        requested_start_logical=np.asarray(
            current_logical_xz,
            dtype=np.float64,
        ).copy(),
        prototype_mean_sliced=(
            expert_mean_sliced
        ),
        prototype_sigma_sliced=(
            expert_sigma_sliced
        ),
        dummy_mean_sliced=(
            dummy_mean_sliced
        ),
        dummy_sigma_sliced=(
            dummy_sigma_sliced
        ),
        corrected_mean=(
            corrected_mean
        ),
        corrected_sigma=(
            corrected_sigma
        ),
        expert_sliced_length=int(
            expert_mean_sliced.shape[0]
        ),
        dummy_sliced_length=int(
            dummy_mean_sliced.shape[0]
        ),
        corrected_length=int(
            corrected_mean.shape[0]
        ),
        source_stride=source_stride,
        sigma_scale=float(sigma_scale),
    )


def correct_expert_end(
    *,
    mean: np.ndarray,
    sigma: np.ndarray,
    ddpg_action,
    source_stride: int = ORIGINAL_STRIDE,
    sigma_scale: float = ORIGINAL_END_SIGMA_SCALE,
) -> CorrectedExpertEnd:
    """
    Strict reproduction of residual_ddpg.py::correct_traj_end().

    Original logic:

        end_point = mus[-1][:2] + ddpg_action

        traj_x = np.array([mus[:,0], sigmas[:,0]]).T
        traj_z = np.array([mus[:,1], sigmas[:,1]]).T

        rng, end_params = self.tg.end_dummy_trajectory(
            end_point,
            len(traj_x),
        )

        mu1, sigma1 = merge_trajectory(
            traj_x,
            end_params[0][::100],
        )
        mu2, sigma2 = merge_trajectory(
            traj_z,
            end_params[1][::100],
        )

    For the strict begin-corrected trajectory:
        input length = 998
        end dummy full length = 99,700
        end dummy [::100] length = 997
        zip-style fusion therefore returns 997 points.

    Even ddpg_action == [0, 0] is NOT an identity operation because the
    endpoint dummy GP is still fused with the incoming trajectory.
    """
    mean, sigma = _validate_mean_sigma(
        mean,
        sigma,
    )

    action = np.asarray(
        ddpg_action,
        dtype=np.float64,
    ).reshape(-1)

    if action.shape != (2,):
        raise ValueError(
            "ddpg_action must have shape (2,)."
        )

    if not np.all(np.isfinite(action)):
        raise ValueError(
            "ddpg_action contains NaN/Inf."
        )

    source_stride = int(source_stride)

    if source_stride <= 0:
        raise ValueError(
            "source_stride must be > 0."
        )

    requested_end = (
        mean[-1, :2]
        + action
    )

    (
        dummy_mean_sliced,
        dummy_sigma_sliced,
    ) = make_original_end_dummy_sliced(
        requested_end,
        segm_len=int(
            mean.shape[0]
        ),
        source_stride=source_stride,
        sigma_scale=sigma_scale,
    )

    (
        corrected_mean,
        corrected_sigma,
    ) = merge_gaussian_trajectory_zip_semantics(
        mean,
        sigma,
        dummy_mean_sliced,
        dummy_sigma_sliced,
    )

    return CorrectedExpertEnd(
        ddpg_action=action.copy(),
        requested_end_logical=(
            requested_end.copy()
        ),
        input_mean=mean.copy(),
        input_sigma=sigma.copy(),
        dummy_mean_sliced=(
            dummy_mean_sliced
        ),
        dummy_sigma_sliced=(
            dummy_sigma_sliced
        ),
        corrected_mean=(
            corrected_mean
        ),
        corrected_sigma=(
            corrected_sigma
        ),
        input_length=int(
            mean.shape[0]
        ),
        dummy_sliced_length=int(
            dummy_mean_sliced.shape[0]
        ),
        corrected_length=int(
            corrected_mean.shape[0]
        ),
        source_stride=source_stride,
        sigma_scale=float(
            sigma_scale
        ),
    )

