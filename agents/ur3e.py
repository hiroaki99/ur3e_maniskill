from pathlib import Path

import copy
import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "ur3e"
    / "ur3e_maniskill.urdf"
)


@register_agent()
class UR3e(BaseAgent):
    uid = "ur3e_custom"

    urdf_path = str(URDF_PATH)

    # UR3eは台座固定の産業用マニピュレータとして扱う
    fix_root_link = True

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # 初期テスト用の仮のPDパラメータ
    # UR3eに対して最適化済みの値ではない
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    # 初期姿勢
    # 後でGUIで姿勢・自己衝突を確認して調整する
    keyframes = {
        "rest": Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -np.pi / 2,
                    np.pi / 2,
                    -np.pi / 2,
                    -np.pi / 2,
                    0.0,
                ],
                dtype=np.float32,
            ),
            pose=sapien.Pose(
                p=[0.0, 0.0, 0.0],
            ),
        )
    }

    @property
    def _controller_configs(self):
        # 絶対関節角指令
        pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        # 現在角度からの差分指令
        pd_joint_delta_pos = PDJointPosControllerConfig(
        self.arm_joint_names,
        lower=-0.03,
        upper=0.03,
        stiffness=self.arm_stiffness,
        damping=self.arm_damping,
        force_limit=self.arm_force_limit,
        use_delta=True,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = {
            "pd_joint_delta_pos": {
                "arm": pd_joint_delta_pos,
            },
            "pd_joint_pos": {
                "arm": pd_joint_pos,
            }, 
            "pd_joint_target_delta_pos": {
                "arm": pd_joint_target_delta_pos,
            },
        }

        return deepcopy_dict(controller_configs)
    
    @property
    def tcp(self):
        """現在の暫定TCP。EZGripper追加後は把持中心へ変更する。"""
        return self.robot.links_map["tool0"]