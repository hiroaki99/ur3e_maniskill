from pathlib import Path

import numpy as np
import sapien

from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import (
    PDJointPosMimicControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent

from agents.ur3e import UR3e


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@register_agent()
class UR3eEZGripper(UR3e):
    """UR3e + SAKE Robotics EZGripper Gen2 Dual."""

    uid = "ur3e_ezgripper"

    urdf_path = str(
        PROJECT_ROOT
        / "assets"
        / "ur3e_ezgripper"
        / "ur3e_ezgripper.urdf"
    )

    gripper_joint_names = [
        "gripper_ezgripper_knuckle_palm_L1_1",
        "gripper_ezgripper_knuckle_palm_L1_2",
    ]

    # URDFで確認した関節制限
    gripper_lower = -1.57075
    gripper_upper = 0.27

    # 初期調整値。実機値ではなくシミュレーション用。
    gripper_stiffness = 100.0
    gripper_damping = 10.0
    gripper_force_limit = 2.0
    gripper_friction = 0.1

    gripper_open_action = -1.0
    gripper_close_action = 1.0

    gripper_open_qpos = -1.57075
    gripper_closed_qpos = 0.27

    # まずは安全な中立付近の姿勢にする。
    # 開状態・閉状態はGUI確認後に確定する。
    keyframes = {
    "rest": Keyframe(
        qpos=np.array(
            [
                # UR3e
                0.0,
                -np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                -np.pi / 2,
                0.0,

                # EZGripper：開状態
                -1.57075,
                -1.57075,
            ],
            dtype=np.float32,
        ),
        pose=sapien.Pose(
            p=[0.0, 0.0, 0.0],
        ),
    )
}

    # 指先に高摩擦materialを設定する。
    urdf_config = {
        "_materials": {
            "gripper": {
                "static_friction": 2.0,
                "dynamic_friction": 2.0,
                "restitution": 0.0,
            }
        },
        "link": {
            "gripper_ezgripper_finger_pad_1": {
                "material": "gripper",
                "patch_radius": 0.02,
                "min_patch_radius": 0.01,
            },
            "gripper_ezgripper_finger_pad_2": {
                "material": "gripper",
                "patch_radius": 0.02,
                "min_patch_radius": 0.01,
            },
        },
    }

    @property
    def tcp(self):
        """把持中心を表す仮TCP。"""
        return self.robot.links_map["grasp_tcp"]

    @property
    def finger1_link(self):
        return self.robot.links_map[
            "gripper_ezgripper_finger_pad_1"
        ]

    @property
    def finger2_link(self):
        return self.robot.links_map[
            "gripper_ezgripper_finger_pad_2"
        ]

    @property
    def _controller_configs(self):
        # 親クラスUR3eのarm controllerを取得する
        controller_configs = super()._controller_configs

        # 2つの指関節を1つのactionで同期制御する
        gripper_pd_joint_pos = (
            PDJointPosMimicControllerConfig(
                joint_names=self.gripper_joint_names,
                lower=self.gripper_lower,
                upper=self.gripper_upper,
                stiffness=self.gripper_stiffness,
                damping=self.gripper_damping,
                force_limit=self.gripper_force_limit,
                friction=self.gripper_friction,
                normalize_action=True,
                mimic={
                    "gripper_ezgripper_knuckle_palm_L1_2": {
                        "joint":
                            "gripper_ezgripper_"
                            "knuckle_palm_L1_1",
                        "multiplier": 1.0,
                        "offset": 0.0,
                    }
                },
            )
        )

        # UR3e側で定義されている各制御モードへ
        # 同じグリッパcontrollerを追加する
        for mode_config in controller_configs.values():
            mode_config["gripper"] = gripper_pd_joint_pos

        return deepcopy_dict(controller_configs)