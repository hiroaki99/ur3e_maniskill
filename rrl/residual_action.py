#!/usr/bin/env python3
"""
Residual Reinforcement Learning用Action合成処理。

Reference action:
    a_ref

RLからのResidual action:
    a_res

最終Action:
    a = clip(
        a_ref + alpha * a_res,
        -1,
        +1
    )

Gripperはこのクラスでは扱わない。
このクラスが担当するのはUR3e Arm 6DoFのみ。
"""

from __future__ import annotations

import numpy as np


def _broadcast_bound(
    value,
    dof: int,
    name: str,
) -> np.ndarray:
    """
    controller.config.lower / upper が
    scalarでもarrayでも6DoF配列へ変換する。
    """

    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim == 0:

        array = np.full(
            dof,
            float(array),
            dtype=np.float64,
        )

    else:

        array = array.reshape(-1)

        if array.size == 1:

            array = np.full(
                dof,
                float(array[0]),
                dtype=np.float64,
            )

    if array.size != dof:

        raise ValueError(
            f"{name} size mismatch: "
            f"expected={dof}, "
            f"actual={array.size}"
        )

    return array


class ResidualActionComposer:
    """
    UR3e Arm用Residual Action合成器。
    """

    def __init__(
        self,
        physical_lower,
        physical_upper,
        alpha: float,
        dof: int = 6,
    ):

        self.dof = int(
            dof
        )

        if self.dof <= 0:

            raise ValueError(
                "dofは1以上である必要があります"
            )

        self.lower = _broadcast_bound(
            physical_lower,
            self.dof,
            "lower",
        )

        self.upper = _broadcast_bound(
            physical_upper,
            self.dof,
            "upper",
        )

        if np.any(
            self.upper
            <= self.lower
        ):

            raise ValueError(
                "upperはlowerより"
                "大きい必要があります"
            )

        self.alpha = float(
            alpha
        )

        if not (
            0.0
            <= self.alpha
            <= 1.0
        ):

            raise ValueError(
                "alphaは0～1で指定してください"
            )

        self.mid = (
            0.5
            * (
                self.upper
                + self.lower
            )
        )

        self.half_range = (
            0.5
            * (
                self.upper
                - self.lower
            )
        )

    # ------------------------------------------------------
    # Physical delta <-> normalized action
    # ------------------------------------------------------

    def physical_delta_to_normalized(
        self,
        physical_delta,
    ) -> np.ndarray:
        """
        controllerの物理deltaを
        normalized [-1,1] actionへ変換する。

        ManiSkill gym_utils.inv_scale_actionと同じ式。
        """

        delta = np.asarray(
            physical_delta,
            dtype=np.float64,
        ).reshape(-1)

        self._validate_shape(
            delta,
            "physical_delta",
        )

        normalized = (
            (
                delta
                - self.mid
            )
            / self.half_range
        )

        return np.clip(
            normalized,
            -1.0,
            1.0,
        )

    def normalized_to_physical_delta(
        self,
        action,
    ) -> np.ndarray:
        """
        normalized action [-1,1]を
        controllerの物理deltaへ戻す。
        """

        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        self._validate_shape(
            action,
            "action",
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        return (
            self.mid
            + self.half_range
            * action
        )

    # ------------------------------------------------------
    # Reference Action
    # ------------------------------------------------------

    def reference_action_from_qpos(
        self,
        current_qpos,
        target_qpos,
    ) -> np.ndarray:
        """
        pd_joint_delta_pos用Reference Action。

        physical_delta =
            q_ref - q_current
        """

        current = np.asarray(
            current_qpos,
            dtype=np.float64,
        ).reshape(-1)

        target = np.asarray(
            target_qpos,
            dtype=np.float64,
        ).reshape(-1)

        self._validate_shape(
            current,
            "current_qpos",
        )

        self._validate_shape(
            target,
            "target_qpos",
        )

        physical_delta = (
            target
            - current
        )

        return (
            self.physical_delta_to_normalized(
                physical_delta
            )
        )

    # ------------------------------------------------------
    # Residual composition
    # ------------------------------------------------------

    def compose(
        self,
        reference_action,
        residual_action,
    ) -> np.ndarray:
        """
        a =
            clip(
                a_ref
                + alpha * a_res,
                -1,
                +1
            )
        """

        reference = np.asarray(
            reference_action,
            dtype=np.float64,
        ).reshape(-1)

        residual = np.asarray(
            residual_action,
            dtype=np.float64,
        ).reshape(-1)

        self._validate_shape(
            reference,
            "reference_action",
        )

        self._validate_shape(
            residual,
            "residual_action",
        )

        reference = np.clip(
            reference,
            -1.0,
            1.0,
        )

        # TD3 Policy出力も[-1,1]へ制限
        residual = np.clip(
            residual,
            -1.0,
            1.0,
        )

        combined = (
            reference
            + self.alpha
            * residual
        )

        return np.clip(
            combined,
            -1.0,
            1.0,
        )

    def compose_from_qpos(
        self,
        current_qpos,
        target_qpos,
        residual_action,
    ):
        """
        q_current, q_ref, a_resから
        Reference/Residual/Combinedをまとめて返す。
        """

        reference = (
            self.reference_action_from_qpos(
                current_qpos,
                target_qpos,
            )
        )

        combined = self.compose(
            reference,
            residual_action,
        )

        return {
            "reference_action":
                reference,

            "residual_action":
                np.clip(
                    np.asarray(
                        residual_action,
                        dtype=np.float64,
                    ).reshape(-1),
                    -1.0,
                    1.0,
                ),

            "combined_action":
                combined,
        }

    # ------------------------------------------------------
    # Analysis
    # ------------------------------------------------------

    def physical_residual_effect(
        self,
        reference_action,
        residual_action,
    ) -> np.ndarray:
        """
        Residualを加える前後で、
        物理deltaが何rad変化したかを返す。

        clippingされた場合も反映される。
        """

        reference = np.asarray(
            reference_action,
            dtype=np.float64,
        ).reshape(-1)

        combined = self.compose(
            reference,
            residual_action,
        )

        reference_delta = (
            self.normalized_to_physical_delta(
                reference
            )
        )

        combined_delta = (
            self.normalized_to_physical_delta(
                combined
            )
        )

        return (
            combined_delta
            - reference_delta
        )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    def _validate_shape(
        self,
        value,
        name,
    ):

        if value.size != self.dof:

            raise ValueError(
                f"{name} shape mismatch: "
                f"expected={self.dof}, "
                f"actual={value.size}"
            )