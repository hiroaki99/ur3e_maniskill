import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from agents.ur3e_ezgripper import UR3eEZGripper


@register_env(
    "EZGripperContactTest-v0",
    max_episode_steps=400,
)
class EZGripperContactTestEnv(BaseEnv):
    """固定キューブに対するEZGripper接触確認環境。"""

    SUPPORTED_ROBOTS = ["ur3e_ezgripper"]
    agent: UR3eEZGripper

    cube_half_size = 0.02

    # grasp_tcp調整後にcheckスクリプトで測定し、
    # 実際のTCPワールド座標へ更新する
    cube_position = [
        0.4568,
        0.3332,
        0.0665,
    ]

    def __init__(
        self,
        *args,
        robot_uids="ur3e_ezgripper",
        **kwargs,
    ):
        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0.0, 0.0, 0.0]),
        )

    def _load_scene(self, options: dict):
        builder = self.scene.create_actor_builder()

        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3,
        )

        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(
                base_color=[1.0, 0.2, 0.2, 1.0],
            ),
        )

        builder.initial_pose = sapien.Pose(
            p=self.cube_position,
        )

        # 最初は動かないキューブとして接触だけ確認
        self.cube = builder.build_kinematic(
            name="contact_test_cube",
        )

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(self.device):
            batch_size = len(env_idx)

            qpos = torch.tensor(
                [
                    # UR3e
                    0.0,
                    -np.pi / 2,
                    np.pi / 2,
                    -np.pi / 2,
                    -np.pi / 2,
                    0.0,

                    # EZGripper：開
                    -1.57075,
                    -1.57075,
                ],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)

            self.agent.reset(qpos)

            cube_position = torch.tensor(
                self.cube_position,
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)

            self.cube.set_pose(
                Pose.create_from_pq(
                    p=cube_position,
                    q=[1.0, 0.0, 0.0, 0.0],
                )
            )

    def evaluate(self):
        finger1_force = (
            self.scene.get_pairwise_contact_forces(
                self.agent.finger1_link,
                self.cube,
            )
        )

        finger2_force = (
            self.scene.get_pairwise_contact_forces(
                self.agent.finger2_link,
                self.cube,
            )
        )

        finger1_force_norm = torch.linalg.norm(
            finger1_force,
            dim=1,
        )

        finger2_force_norm = torch.linalg.norm(
            finger2_force,
            dim=1,
        )

        finger1_contact = finger1_force_norm > 0.1
        finger2_contact = finger2_force_norm > 0.1

        both_fingers_contact = (
            finger1_contact
            & finger2_contact
        )

        return {
            "success": both_fingers_contact,
            "finger1_force": finger1_force_norm,
            "finger2_force": finger2_force_norm,
            "finger1_contact": finger1_contact,
            "finger2_contact": finger2_contact,
        }