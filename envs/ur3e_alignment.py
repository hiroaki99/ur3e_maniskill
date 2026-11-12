#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UR3e + EZGripper ManiSkill adaptation of Yaginuma's 3-object alignment task.

Scope of this environment
-------------------------
- Three dynamic 5 cm cubes A/B/C.
- Yaginuma-compatible logical random initialization.
- Configurable logical [x,z] <-> ManiSkill world [X,Y] mapping.
- Raw12 / pose9 state accessors.
- Yaginuma distance/alignment reward metrics.
- EZGripper contact instrumentation for object A.

Not implemented here yet
------------------------
- automatic approach/grasp of A
- GP Expert execution
- begin/end GP correction inside env.step()
- TD3 residual action

Those are intentionally kept in later adapter/wrapper layers so that
environment geometry/state conversion can be validated first.

Important compatibility note
----------------------------
The original Gazebo source stores an angle-like fourth value per object, but
sets Gazebo yaw and later reads Euler index 1.  For a planar ManiSkill table
task this environment uses the physically meaningful table-plane yaw and
exposes it in raw12/pose9.  Position semantics and tensor shapes are kept
compatible.  This choice must be logged when comparing numerically against
the original Gazebo experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
import yaml

import agents.ur3e_ezgripper  # noqa: F401

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_alignment.yaml"


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            "ur3e_alignment.yaml must contain a mapping."
        )

    return cfg


CONFIG = _load_config()

MAX_EPISODE_STEPS = int(
    CONFIG["project"]["max_episode_steps"]
)


@register_env(
    "UR3eAlignment-v0",
    max_episode_steps=MAX_EPISODE_STEPS,
)
class UR3eAlignmentEnv(BaseEnv):
    """Three-object alignment task for UR3e + EZGripper."""

    SUPPORTED_ROBOTS = ["ur3e_ezgripper"]

    def __init__(
        self,
        *args,
        robot_uids: str = "ur3e_ezgripper",
        robot_init_qpos_noise: float = 0.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(
            robot_init_qpos_noise
        )

        self.robot_base_position = np.asarray(
            CONFIG["robot"]["base_position"],
            dtype=np.float32,
        )

        object_cfg = CONFIG["objects"]
        self.object_size = float(
            object_cfg["size_m"]
        )
        self.object_half_size = (
            self.object_size / 2.0
        )
        self.object_mass = float(
            object_cfg["mass_kg"]
        )

        mapping = CONFIG["workspace_mapping"]
        self.logical_scale = float(
            mapping["scale"]
        )
        self.logical_reference_xz = np.asarray(
            mapping["logical_reference_xz_m"],
            dtype=np.float64,
        )
        self.world_reference_xy = np.asarray(
            mapping["world_reference_xy_m"],
            dtype=np.float64,
        )

        if (
            self.logical_reference_xz.shape
            != (2,)
        ):
            raise ValueError(
                "logical_reference_xz_m must have shape (2,)."
            )

        if (
            self.world_reference_xy.shape
            != (2,)
        ):
            raise ValueError(
                "world_reference_xy_m must have shape (2,)."
            )

        if self.logical_scale <= 0.0:
            raise ValueError(
                "workspace_mapping.scale must be > 0."
            )

        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs,
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.75, 0.65, 0.65],
            target=[0.34, 0.0, 0.05],
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

    def _load_agent(
        self,
        options: dict,
    ):
        super()._load_agent(
            options,
            sapien.Pose(
                p=self.robot_base_position.tolist()
            ),
        )

    def _build_alignment_cube(
        self,
        *,
        name: str,
        color,
    ):
        builder = (
            self.scene
            .create_actor_builder()
        )

        density = (
            self.object_mass
            / (self.object_size ** 3)
        )

        builder.add_box_collision(
            half_size=[
                self.object_half_size
            ] * 3,
            density=density,
        )

        builder.add_box_visual(
            half_size=[
                self.object_half_size
            ] * 3,
            material=sapien.render.RenderMaterial(
                base_color=list(color)
            ),
        )

        # Placeholder pose; randomized reset overwrites it.
        builder.set_initial_pose(
            sapien.Pose(
                p=[
                    float(
                        self.world_reference_xy[0]
                    ),
                    float(
                        self.world_reference_xy[1]
                    ),
                    self.object_half_size,
                ]
            )
        )

        return builder.build(
            name=name
        )

    def _load_scene(
        self,
        options: dict,
    ):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=(
                self.robot_init_qpos_noise
            ),
        )
        self.table_scene.build()

        colors = (
            CONFIG["objects"]
            ["colors_rgba"]
        )

        self.object_A = (
            self._build_alignment_cube(
                name="A",
                color=colors["A"],
            )
        )
        self.object_B = (
            self._build_alignment_cube(
                name="B",
                color=colors["B"],
            )
        )
        self.object_C = (
            self._build_alignment_cube(
                name="C",
                color=colors["C"],
            )
        )

        self.objects = (
            self.object_A,
            self.object_B,
            self.object_C,
        )

        # Gripper contact links are prepared now for the later grasp gate.
        links_map = (
            self.agent.robot.links_map
        )
        gripper_cfg = CONFIG["gripper"]

        left_names = list(
            gripper_cfg[
                "left_contact_links"
            ]
        )
        right_names = list(
            gripper_cfg[
                "right_contact_links"
            ]
        )

        missing = [
            name
            for name in (
                left_names
                + right_names
            )
            if name not in links_map
        ]

        if missing:
            raise KeyError(
                "Missing gripper contact links: "
                f"{missing}"
            )

        self.left_contact_links = [
            links_map[name]
            for name in left_names
        ]
        self.right_contact_links = [
            links_map[name]
            for name in right_names
        ]

        self.contact_force_threshold = float(
            gripper_cfg[
                "contact_force_threshold_n"
            ]
        )

    # ---------------------------------------------------------
    # Coordinate conversion
    # ---------------------------------------------------------
    def logical_xz_to_world_xy(
        self,
        logical_xz,
    ):
        """
        Yaginuma logical +x -> ManiSkill world +X
        Yaginuma logical +z -> ManiSkill world -Y
        """
        logical = torch.as_tensor(
            logical_xz,
            dtype=torch.float32,
            device=self.device,
        )

        ref_logical = torch.as_tensor(
            self.logical_reference_xz,
            dtype=logical.dtype,
            device=logical.device,
        )
        ref_world = torch.as_tensor(
            self.world_reference_xy,
            dtype=logical.dtype,
            device=logical.device,
        )

        world_x = (
            ref_world[0]
            + self.logical_scale
            * (
                logical[..., 0]
                - ref_logical[0]
            )
        )

        world_y = (
            ref_world[1]
            - self.logical_scale
            * (
                logical[..., 1]
                - ref_logical[1]
            )
        )

        return torch.stack(
            [world_x, world_y],
            dim=-1,
        )

    def world_xy_to_logical_xz(
        self,
        world_xy,
    ):
        world = torch.as_tensor(
            world_xy,
            dtype=torch.float32,
            device=self.device,
        )

        ref_logical = torch.as_tensor(
            self.logical_reference_xz,
            dtype=world.dtype,
            device=world.device,
        )
        ref_world = torch.as_tensor(
            self.world_reference_xy,
            dtype=world.dtype,
            device=world.device,
        )

        logical_x = (
            ref_logical[0]
            + (
                world[..., 0]
                - ref_world[0]
            )
            / self.logical_scale
        )

        logical_z = (
            ref_logical[1]
            - (
                world[..., 1]
                - ref_world[1]
            )
            / self.logical_scale
        )

        return torch.stack(
            [logical_x, logical_z],
            dim=-1,
        )

    # ---------------------------------------------------------
    # Reset sampling: residual_ddpg.py::gzb_init_pos()
    # ---------------------------------------------------------
    def _uniform(
        self,
        shape,
        low: float,
        high: float,
    ):
        # Deliberately retains low + (high-low)*u semantics,
        # including the original source's possible reversed interval.
        u = torch.rand(
            shape,
            dtype=torch.float32,
            device=self.device,
        )

        return (
            float(low)
            + (
                float(high)
                - float(low)
            )
            * u
        )

    def _sample_logical_reset(
        self,
        batch_size: int,
    ):
        task = CONFIG[
            "logical_task"
        ]

        box_length = float(
            task["box_length_m"]
        )
        box_width = float(
            task["box_width_m"]
        )
        table_length = float(
            task["table_length_m"]
        )
        table_width = float(
            task["table_width_m"]
        )
        a_x_min = float(
            task["a_x_min_m"]
        )

        angle_limit = np.deg2rad(
            float(
                task["angle_limit_deg"]
            )
        )

        a_x = self._uniform(
            (batch_size,),
            a_x_min,
            (
                table_length
                - box_length / 2.0
            ),
        )

        a_z = self._uniform(
            (batch_size,),
            box_width / 2.0,
            (
                table_width
                - box_width / 2.0
            ),
        )

        b_x = torch.full(
            (batch_size,),
            box_length / 2.0,
            dtype=torch.float32,
            device=self.device,
        )

        b_z = self._uniform(
            (batch_size,),
            box_width / 2.0,
            (
                table_width
                - box_width * 1.5
            ),
        )

        c_x = torch.full(
            (batch_size,),
            box_length / 2.0,
            dtype=torch.float32,
            device=self.device,
        )

        # Exact algebraic form of the source distribution.
        c_low = (
            b_z
            + box_width * 1.5
        )
        c_high = (
            table_width
            - box_width / 2.0
        )
        u = torch.rand(
            (batch_size,),
            dtype=torch.float32,
            device=self.device,
        )
        c_z = (
            c_low
            + (
                c_high
                - c_low
            )
            * u
        )

        angles = self._uniform(
            (batch_size, 3),
            -angle_limit,
            angle_limit,
        )

        logical = torch.stack(
            [
                torch.stack(
                    [a_x, a_z],
                    dim=-1,
                ),
                torch.stack(
                    [b_x, b_z],
                    dim=-1,
                ),
                torch.stack(
                    [c_x, c_z],
                    dim=-1,
                ),
            ],
            dim=1,
        )

        return (
            logical,
            angles,
        )

    @staticmethod
    def _yaw_to_wxyz(
        yaw,
    ):
        half = yaw / 2.0

        q = torch.zeros(
            (
                yaw.shape[0],
                4,
            ),
            dtype=yaw.dtype,
            device=yaw.device,
        )

        q[:, 0] = torch.cos(
            half
        )
        q[:, 3] = torch.sin(
            half
        )

        return q

    @staticmethod
    def _wxyz_to_yaw(
        q,
    ):
        # q=[w,x,y,z]
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        siny_cosp = (
            2.0
            * (
                w * z
                + x * y
            )
        )
        cosy_cosp = (
            1.0
            - 2.0
            * (
                y * y
                + z * z
            )
        )

        return torch.atan2(
            siny_cosp,
            cosy_cosp,
        )

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(
            self.device
        ):
            batch_size = len(
                env_idx
            )

            self.table_scene.initialize(
                env_idx
            )

            arm_qpos = torch.tensor(
                CONFIG["robot"]
                ["initial_arm_qpos"],
                dtype=torch.float32,
                device=self.device,
            )

            gripper_qpos = torch.tensor(
                CONFIG["robot"]
                ["initial_gripper_qpos"],
                dtype=torch.float32,
                device=self.device,
            )

            qpos = torch.cat(
                [
                    arm_qpos,
                    gripper_qpos,
                ],
                dim=0,
            )

            qpos = (
                qpos.unsqueeze(0)
                .repeat(
                    batch_size,
                    1,
                )
            )

            self.agent.robot.set_qpos(
                qpos
            )
            self.agent.robot.set_qvel(
                torch.zeros_like(
                    qpos
                )
            )

            (
                logical,
                angles,
            ) = self._sample_logical_reset(
                batch_size
            )

            table_z = float(
                CONFIG["table"]
                ["surface_z_m"]
            )

            zero_velocity = torch.zeros(
                (
                    batch_size,
                    3,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            for object_index, actor in enumerate(
                self.objects
            ):
                world_xy = (
                    self
                    .logical_xz_to_world_xy(
                        logical[
                            :,
                            object_index,
                            :,
                        ]
                    )
                )

                p = torch.zeros(
                    (
                        batch_size,
                        3,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )

                p[:, 0:2] = (
                    world_xy
                )
                p[:, 2] = (
                    table_z
                    + self.object_half_size
                )

                q = (
                    self._yaw_to_wxyz(
                        angles[
                            :,
                            object_index,
                        ]
                    )
                )

                actor.set_pose(
                    Pose.create_from_pq(
                        p=p,
                        q=q,
                    )
                )
                actor.set_linear_velocity(
                    zero_velocity
                )
                actor.set_angular_velocity(
                    zero_velocity
                )

    # ---------------------------------------------------------
    # Yaginuma-compatible state contract
    # ---------------------------------------------------------
    def get_alignment_raw12(
        self,
    ):
        fields = []

        for actor in self.objects:
            logical_xz = (
                self
                .world_xy_to_logical_xz(
                    actor.pose.p[
                        :,
                        :2,
                    ]
                )
            )

            yaw = (
                self._wxyz_to_yaw(
                    actor.pose.q
                )
            )

            zeros = torch.zeros_like(
                yaw
            )

            fields.extend(
                [
                    logical_xz[:, 0],
                    zeros,
                    logical_xz[:, 1],
                    yaw,
                ]
            )

        return torch.stack(
            fields,
            dim=1,
        )

    def get_alignment_pose9(
        self,
    ):
        raw12 = (
            self.get_alignment_raw12()
        )

        return raw12[
            :,
            [
                0, 2, 3,
                4, 6, 7,
                8, 10, 11,
            ],
        ]

    # ---------------------------------------------------------
    # Contact + source-compatible task metrics
    # ---------------------------------------------------------
    def _compute_side_contact_force(
        self,
        links,
        actor,
    ):
        force_norms = []

        for link in links:
            vector = (
                self.scene
                .get_pairwise_contact_forces(
                    link,
                    actor,
                )
            )
            force_norms.append(
                torch.linalg.norm(
                    vector,
                    dim=1,
                )
            )

        return torch.stack(
            force_norms,
            dim=0,
        ).sum(
            dim=0
        )

    @staticmethod
    def _safe_cos(
        v1,
        v2,
    ):
        denominator = (
            torch.linalg.norm(
                v1,
                dim=1,
            )
            * torch.linalg.norm(
                v2,
                dim=1,
            )
        )

        denominator = torch.clamp(
            denominator,
            min=1.0e-12,
        )

        return torch.sum(
            v1 * v2,
            dim=1,
        ) / denominator

    def _alignment_metrics(
        self,
    ):
        pose9 = (
            self.get_alignment_pose9()
        )

        A = pose9[:, [0, 1]]
        B = pose9[:, [3, 4]]
        C = pose9[:, [6, 7]]

        dis12 = torch.linalg.norm(
            A - B,
            dim=1,
        )
        dis23 = torch.linalg.norm(
            B - C,
            dim=1,
        )
        dis31 = torch.linalg.norm(
            C - A,
            dim=1,
        )

        distances = torch.stack(
            [
                dis12,
                dis23,
                dis31,
            ],
            dim=1,
        )

        r_dis = (
            10.0
            * torch.square(
                torch.sum(
                    distances,
                    dim=1,
                )
                - torch.max(
                    distances,
                    dim=1,
                ).values
            )
        )

        cos_A = self._safe_cos(
            B - A,
            C - A,
        )
        cos_B = self._safe_cos(
            A - B,
            C - B,
        )
        cos_C = self._safe_cos(
            A - C,
            B - C,
        )

        r_theta = torch.max(
            torch.stack(
                [
                    cos_A,
                    cos_B,
                    cos_C,
                ],
                dim=1,
            ),
            dim=1,
        ).values

        return {
            "pose9": pose9,
            "dis12": dis12,
            "dis23": dis23,
            "dis31": dis31,
            "r_dis": r_dis,
            "r_theta": r_theta,
        }

    def evaluate(
        self,
    ):
        metrics = (
            self._alignment_metrics()
        )

        alignment_cfg = CONFIG[
            "alignment"
        ]

        success = (
            (
                metrics["r_dis"]
                < float(
                    alignment_cfg[
                        "success_r_dis_threshold"
                    ]
                )
            )
            & (
                metrics["r_theta"]
                > float(
                    alignment_cfg[
                        "success_r_theta_threshold"
                    ]
                )
            )
        )

        left_force_A = (
            self._compute_side_contact_force(
                self.left_contact_links,
                self.object_A,
            )
        )
        right_force_A = (
            self._compute_side_contact_force(
                self.right_contact_links,
                self.object_A,
            )
        )

        threshold = (
            self.contact_force_threshold
        )

        return {
            "success": success,
            "r_dis": metrics["r_dis"],
            "r_theta": metrics["r_theta"],
            "dis12": metrics["dis12"],
            "dis23": metrics["dis23"],
            "dis31": metrics["dis31"],
            "left_contact_force_A":
                left_force_A,
            "right_contact_force_A":
                right_force_A,
            "left_contact_A":
                left_force_A
                >= threshold,
            "right_contact_A":
                right_force_A
                >= threshold,
        }

    def _get_obs_extra(
        self,
        info: dict,
    ):
        tcp_pose = (
            self.agent.tcp.pose
        )

        return {
            "tcp_pose":
                tcp_pose.raw_pose,
            "alignment_raw12":
                self.get_alignment_raw12(),
            "alignment_pose9":
                self.get_alignment_pose9(),
            "A_pose":
                self.object_A.pose.raw_pose,
            "B_pose":
                self.object_B.pose.raw_pose,
            "C_pose":
                self.object_C.pose.raw_pose,
        }

    def compute_dense_reward(
        self,
        obs,
        action,
        info,
    ):
        # Source terminal metric used as a diagnostic dense value here.
        # The RRL wrapper will later reproduce the source's step-dependent
        # reward/termination semantics exactly.
        return (
            -info["r_dis"]
            + info["r_theta"]
        )

    def compute_normalized_dense_reward(
        self,
        obs,
        action,
        info,
    ):
        # No source-defined normalization exists.
        return self.compute_dense_reward(
            obs,
            action,
            info,
        )
