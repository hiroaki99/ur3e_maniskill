from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array

# import時にカスタムAgentを登録する
from agents.ur3e import UR3e


@register_env("UR3eReach-v0", max_episode_steps=100)
class UR3eReachEnv(BaseEnv):
    """UR3eのtool0を3次元目標位置へ移動する環境。"""

    SUPPORTED_ROBOTS = ["ur3e_custom"]
    agent: UR3e

    goal_radius = 0.03

    # 固定rest姿勢に対する目標位置。
    # diagnose_ur3e_reach.pyの結果から設定。
    fixed_goal_position = (
        0.33855,
        0.13105,
        0.34330,
    )

    def __init__(
        self,
        *args,
        robot_uids="ur3e_custom",
        robot_init_qpos_noise=0.0, # 初期位置固定
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        # UR3eの台座をワールド原点に配置
        super()._load_agent(
            options,
            sapien.Pose(p=[0.0, 0.0, 0.0]),
        )

    def _load_scene(self, options: dict):
        # 目標位置を示す赤い球。衝突形状は持たせない。
        builder = self.scene.create_actor_builder()

        builder.add_sphere_visual(
            radius=self.goal_radius,
            material=sapien.render.RenderMaterial(
                base_color=[1.0, 0.0, 0.0, 0.7],
            ),
        )

        builder.initial_pose = sapien.Pose(
            p=[0.25, 0.0, 0.25],
        )

        self.goal_site = builder.build_kinematic(
            name="goal_site",
        )

    # def _initialize_episode(
    #     self,
    #     env_idx: torch.Tensor,
    #     options: dict,
    # ):
    #     with torch.device(self.device):
    #         batch_size = len(env_idx)

    #         # agents/ur3e.pyで定義したrest姿勢
    #         qpos = torch.tensor(
    #             [
    #                 0.0,
    #                 -np.pi / 2,
    #                 np.pi / 2,
    #                 -np.pi / 2,
    #                 -np.pi / 2,
    #                 0.0,
    #             ],
    #             dtype=torch.float32,
    #         ).repeat(batch_size, 1)

    #         qpos += (
    #             torch.randn((batch_size, 6))
    #             * self.robot_init_qpos_noise
    #         )

    #         self.agent.reset(qpos)

    #         # 初期TCP位置の近傍に目標を設定する。
    #         # 最初から広いワークスペース全域を使わない。
    #         tcp_position = self.agent.tcp.pose.p[env_idx]

    #         # offset = torch.zeros((batch_size, 3))
    #         # offset[:, 0] = torch.rand(batch_size) * 0.12 - 0.06
    #         # offset[:, 1] = torch.rand(batch_size) * 0.12 - 0.06
    #         # offset[:, 2] = torch.rand(batch_size) * 0.10 + 0.03

    #         offset = torch.tensor(
    #             [0.04, 0.0, 0.04],
    #             dtype=torch.float32,
    #             device=self.device,
    #         ).repeat(batch_size, 1)


    #         goal_position = tcp_position + offset

    #         self.goal_site.set_pose(
    #             Pose.create_from_pq(
    #                 p=goal_position,
    #                 q=[1.0, 0.0, 0.0, 0.0],
    #             )
    #         )

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(self.device):
            batch_size = len(env_idx)

            qpos = torch.tensor(
                [
                    0.0,
                    -np.pi / 2,
                    np.pi / 2,
                    -np.pi / 2,
                    -np.pi / 2,
                    0.0,
                ],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)

            qpos += (
                torch.randn(
                    (batch_size, 6),
                    device=self.device,
                )
                * self.robot_init_qpos_noise
            )

            self.agent.reset(qpos)

            # reset直後のTCP poseには依存しない
            # goal_position = torch.tensor(
            #     self.fixed_goal_position,
            #     dtype=torch.float32,
            #     device=self.device,
            # ).repeat(batch_size, 1)

            # random化
            base_goal_position = torch.tensor(
                self.fixed_goal_position,
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)

            random_offset = torch.zeros(
                (batch_size, 3),
                dtype=torch.float32,
                device=self.device,
            )

            random_offset[:, 0] = (
                torch.rand(batch_size, device=self.device)
                * 0.02
                - 0.01
            )

            random_offset[:, 1] = (
                torch.rand(batch_size, device=self.device)
                * 0.02
                - 0.01
            )

            random_offset[:, 2] = (
                torch.rand(batch_size, device=self.device)
                * 0.02
                - 0.01
            )

            goal_position = (
                base_goal_position
                + random_offset
            )

            self.goal_site.set_pose(
                Pose.create_from_pq(
                    p=goal_position,
                    q=[1.0, 0.0, 0.0, 0.0],
                )
            )

    def evaluate(self):
        tcp_position = self.agent.tcp.pose.p
        goal_position = self.goal_site.pose.p

        distance = torch.linalg.norm(
            tcp_position - goal_position,
            dim=1,
        )

        return {
            "success": distance < self.goal_radius,
            "tcp_to_goal_dist": distance,
        }

    def _get_obs_extra(self, info: dict):
        tcp_position = self.agent.tcp.pose.p
        goal_position = self.goal_site.pose.p

        return {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "goal_pos": goal_position,
            "tcp_to_goal_pos": goal_position - tcp_position,
        }

    def compute_dense_reward(self, obs, action, info):
        distance = info["tcp_to_goal_dist"]

        reaching_reward = 1.0 - torch.tanh(10.0 * distance)

        # success_bonus = 2.0 * info["success"].float()

        # action_penalty = 0.005 * torch.sum(
        #     torch.square(action),
        #     dim=1,
        # )

        # reward = (
        #     reaching_reward
        #     + success_bonus
        #     - action_penalty
        # )

        reward = reaching_reward.clone()

        reward[info["success"]] = 3.0

        return reward


    def compute_normalized_dense_reward(self, obs, action, info):
        # 最大値は概ね3なので正規化する
        return self.compute_dense_reward(
            obs=obs,
            action=action,
            info=info,
        ) / 3.0

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.8, 0.8, 0.7],
            target=[0.0, 0.0, 0.3],
        )

        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1.0,
            near=0.01,
            far=10.0,
        )