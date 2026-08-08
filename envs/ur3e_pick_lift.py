"""固定位置キューブを用いたUR3e Pick-and-Lift環境。

Day 2では以下のみを実装する。

- UR3e＋EZGripperの読み込み
- テーブルの読み込み
- 固定位置の動的キューブ
- reset処理
- 状態観測
- 最低限の成功判定
- 最低限の密報酬

グリッパ開閉、接触判定、把持判定、Residual RLは後日追加する。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
import yaml

# import時に@register_agentが実行される
import agents.ur3e_ezgripper  # noqa: F401

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_lift.yaml"


def load_config() -> dict[str, Any]:
    """Day 1で確定した設定ファイルを読み込む。"""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("設定ファイルの最上位はmappingである必要があります")

    return config


CONFIG = load_config()

# 環境変数が設定されていれば、YAMLより優先する
DEFAULT_ROBOT_UID = os.environ.get(
    "UR3E_EZGRIPPER_UID",
    CONFIG["robot"].get("uid", ""),
)

if not DEFAULT_ROBOT_UID:
    raise ValueError(
        "robot.uidが設定されていません。"
        "configs/ur3e_pick_lift.yamlへEZGripper AgentのUIDを"
        "追加してください。"
    )

MAX_EPISODE_STEPS = int(
    CONFIG["task"]["max_episode_steps"]
)


@register_env(
    "UR3ePickLift-v0",
    max_episode_steps=MAX_EPISODE_STEPS,
)
class UR3ePickLiftEnv(BaseEnv):
    """固定位置キューブを用いたPick-and-Lift環境。"""

    def __init__(
        self,
        *args,
        robot_uids: str = DEFAULT_ROBOT_UID,
        robot_init_qpos_noise: float = 0.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        self.robot_base_position = np.asarray(
            CONFIG["robot"]["base_position"],
            dtype=np.float32,
        )

        self.cube_size = float(CONFIG["cube"]["size"])
        self.cube_half_size = self.cube_size / 2.0
        self.cube_mass = float(CONFIG["cube"]["mass"])

        self.cube_initial_position = np.asarray(CONFIG["cube"]["position"], dtype=np.float32,)
        self.cube_initial_orientation = np.asarray(CONFIG["cube"]["orientation_wxyz"], dtype=np.float32,  )

        self.lift_height = float(CONFIG["task"]["lift_height"])

        super().__init__(*args, robot_uids=robot_uids, **kwargs, )

    @property
    def _default_human_render_camera_configs(self):
        """GUIおよび録画用カメラを定義する。"""

        pose = sapien_utils.look_at(
            eye=[0.80, 0.75, 0.65],
            target=[0.25, 0.00, 0.15],
        )

        return CameraConfig(
            uid="render_camera",
            pose=pose,
            width=768,
            height=768,
            fov=1.0,
            near=0.01,
            far=100.0,
        )

    def _load_agent(self, options: dict):
        """UR3e＋EZGripperの台座位置を設定する。"""

        super()._load_agent(
            options,
            sapien.Pose( p=self.robot_base_position.tolist() ),
        )

    def _load_scene(self, options: dict):
        """テーブルと動的キューブを1回だけ生成する。"""

        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        # 質量 = 密度 × 体積
        #
        # 一辺0.05 m、質量0.10 kgの場合:
        # density = 0.10 / 0.05^3 = 800 kg/m^3
        cube_density = ( self.cube_mass / (self.cube_size ** 3))

        builder = self.scene.create_actor_builder()

        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3,
            density=cube_density,
        )

        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(
                base_color=[0.85, 0.10, 0.10, 1.0],
            ),
        )

        # GPU/CPUどちらでも、生成時に他物体と重ならない
        # 初期姿勢を設定しておく
        builder.set_initial_pose(
            sapien.Pose(
                p=self.cube_initial_position.tolist(),
                q=self.cube_initial_orientation.tolist(),
            )
        )

        # build()なのでdynamic actorとして生成される
        self.cube = builder.build(
            name="pick_cube"
        )

        # Day 4: EZGripper接触リンク

        gripper_config = CONFIG["gripper"]

        left_link_names = (
            gripper_config[
                "left_contact_links"
            ]
        )

        right_link_names = (
            gripper_config[
                "right_contact_links"
            ]
        )

        links_map = self.agent.robot.links_map

        missing_links = []

        for name in (
            left_link_names
            + right_link_names
        ):

            if name not in links_map:
                missing_links.append(name)

        if missing_links:

            raise KeyError(
                "接触判定用リンクがURDFに存在しません: "
                f"{missing_links}"
            )


        self.left_contact_links = [
            links_map[name]
            for name in left_link_names
        ]

        self.right_contact_links = [
            links_map[name]
            for name in right_link_names
        ]


        self.contact_force_threshold = float(
            gripper_config[
                "contact_force_threshold_n"
            ]
        )

        self.grasp_center_distance_threshold = float(
            gripper_config[
                "grasp_center_distance_threshold_m"
            ]
        )

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        """reset時にロボットとキューブを固定初期状態へ戻す。"""

        with torch.device(self.device):
            batch_size = len(env_idx)

            # テーブルとロボットを初期化する
            self.table_scene.initialize(env_idx)

            # -------------------------------------------------
            # UR3e + EZGripper の初期関節角を明示的に設定
            # -------------------------------------------------

            arm_qpos = torch.tensor(
                CONFIG["robot"]["initial_arm_qpos"],
                dtype=torch.float32,
                device=self.device,
            )

            gripper_qpos = torch.tensor(
                CONFIG["robot"]["initial_gripper_qpos"],
                dtype=torch.float32,
                device=self.device,
            )

            initial_qpos = torch.cat(
                [
                    arm_qpos,
                    gripper_qpos,
                ],
                dim=0,
            )

            initial_qpos = initial_qpos.unsqueeze(0).repeat(
                batch_size,
                1,
            )

            self.agent.robot.set_qpos(initial_qpos)

            initial_qvel = torch.zeros_like(
                initial_qpos
            )

            self.agent.robot.set_qvel(initial_qvel)

            cube_position = torch.tensor(
                self.cube_initial_position,
                dtype=torch.float32,
            ).repeat(batch_size, 1)

            cube_orientation = torch.tensor(
                self.cube_initial_orientation,
                dtype=torch.float32,
            ).repeat(batch_size, 1)

            cube_pose = Pose.create_from_pq(
                p=cube_position,
                q=cube_orientation,
            )

            self.cube.set_pose(cube_pose)

            # 前エピソードの速度を残さない
            zero_velocity = torch.zeros(
                (batch_size, 3),
                dtype=torch.float32,
            )
            self.cube.set_linear_velocity(zero_velocity)
            self.cube.set_angular_velocity(zero_velocity)

    def _get_tcp_pose(self):
        """Agent実装差を吸収してTCP Poseを取得する。"""

        if hasattr(self.agent, "tcp_pose"):
            return self.agent.tcp_pose

        if hasattr(self.agent, "tcp"):
            tcp = self.agent.tcp

            if hasattr(tcp, "pose"):
                return tcp.pose

        raise AttributeError(
            "UR3e＋EZGripper Agentにtcp_poseまたはtcp.poseが"
            "定義されていません。"
        )

    def evaluate(self):
        """Day 2用の最低限の成功・状態情報を返す。"""

        cube_height = (
            self.cube.pose.p[:, 2]
            - self.cube_initial_position[2]
        )

        is_lifted = cube_height >= self.lift_height

        # Contact force

        left_contact_force = (
            self._compute_side_contact_force(
                self.left_contact_links
            )
        )

        right_contact_force = (
            self._compute_side_contact_force(
                self.right_contact_links
            )
        )

        left_contact = (
            left_contact_force
            >= self.contact_force_threshold
        )

        right_contact = (
            right_contact_force
            >= self.contact_force_threshold
        )

        # Gripper center

        gripper_center = (
            self._compute_gripper_center()
        )

        cube_to_gripper_center_distance = (
            torch.linalg.norm(
                self.cube.pose.p
                - gripper_center,
                dim=1,
            )
        )

        # Grasp candidate

        is_grasp_candidate = (
            left_contact
            & right_contact
            & (
                cube_to_gripper_center_distance
                <= self.grasp_center_distance_threshold
            )
        )

        # Success
        #
        # Day 2では「持ち上がっただけ」で成功だったが、
        # Day 4以降は「把持しながら持ち上げる」ことを要求する。

        success = (
            is_lifted
            & is_grasp_candidate
        )

        return {
            "success": success,

            "is_lifted": is_lifted,

            "cube_height": cube_height,

            "left_contact_force": (
                left_contact_force
            ),

            "right_contact_force": (
                right_contact_force
            ),

            "left_contact": (
                left_contact
            ),

            "right_contact": (
                right_contact
            ),

            "is_grasp_candidate": (
                is_grasp_candidate
            ),

            "cube_to_gripper_center_distance": (
                cube_to_gripper_center_distance
            ),
        }

    def _get_obs_extra(self, info: dict):
        """状態観測へTCPとキューブの情報を追加する。"""

        tcp_pose = self._get_tcp_pose()

        obs = {
            "tcp_pose": tcp_pose.raw_pose,
            "cube_pose": self.cube.pose.raw_pose,
        }

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_cube_pos=(
                    self.cube.pose.p - tcp_pose.p
                ),
                cube_height=(
                    self.cube.pose.p[:, 2]
                    - self.cube_initial_position[2]
                ).unsqueeze(-1),
            )

        return obs

    def compute_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        """環境動作確認用の暫定報酬。

        Day 2では学習には使用しない。
        """

        tcp_pose = self._get_tcp_pose()

        tcp_to_cube_distance = torch.linalg.norm(
            self.cube.pose.p - tcp_pose.p,
            dim=1,
        )

        reach_reward = (
            1.0
            - torch.tanh(
                5.0 * tcp_to_cube_distance
            )
        )

        lift_progress = torch.clamp(
            info["cube_height"] / self.lift_height,
            min=0.0,
            max=1.0,
        )

        return reach_reward + lift_progress

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return (
            self.compute_dense_reward(
                obs=obs,
                action=action,
                info=info,
            )
            / 2.0
        )


    def _compute_side_contact_force(
        self,
        links,
    ):
        """
        指側に属する複数Linkとキューブとの
        pairwise contact forceの大きさを合計する。

        Returns
        -------
        torch.Tensor
            shape = (num_envs,)
            単位 = N
        """

        force_norms = []

        for link in links:

            force_vector = (
                self.scene.get_pairwise_contact_forces(
                    link,
                    self.cube,
                )
            )

            force_norm = torch.linalg.norm(
                force_vector,
                dim=1,
            )

            force_norms.append(
                force_norm
            )

        return torch.stack(
            force_norms,
            dim=0,
        ).sum(
            dim=0
        )


    def _compute_gripper_center(self):
        """
        左右の接触リンク位置から、
        現段階の簡易的なグリッパ中心を計算する。
        """

        left_positions = torch.stack(
            [
                link.pose.p
                for link in self.left_contact_links
            ],
            dim=0,
        ).mean(
            dim=0
        )

        right_positions = torch.stack(
            [
                link.pose.p
                for link in self.right_contact_links
            ],
            dim=0,
        ).mean(
            dim=0
        )

        gripper_center = (
            left_positions
            + right_positions
        ) / 2.0

        return gripper_center