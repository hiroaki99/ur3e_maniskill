"""UR3e + EZGripper fixed-position Pick-and-Place environment (Week 1).

Week 1 scope:
- fixed cube
- fixed goal
- scripted/reference control only
- no Residual RL and no randomization
"""

from __future__ import annotations

import os
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
CONFIG_PATH = REPO_ROOT / "configs" / "ur3e_pick_place.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("ur3e_pick_place.yaml must contain a mapping")
    return config


CONFIG = load_config()
DEFAULT_ROBOT_UID = os.environ.get(
    "UR3E_EZGRIPPER_UID",
    CONFIG["robot"]["uid"],
)
MAX_EPISODE_STEPS = int(CONFIG["pick_place"]["max_episode_steps"])


@register_env("UR3ePickPlace-v0", max_episode_steps=MAX_EPISODE_STEPS)
class UR3ePickPlaceEnv(BaseEnv):
    """Fixed Cube / fixed Goal Pick-and-Place environment."""

    def __init__(
        self,
        *args,
        robot_uids: str = DEFAULT_ROBOT_UID,
        robot_init_qpos_noise: float = 0.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)

        self.robot_base_position = np.asarray(
            CONFIG["robot"]["base_position"], dtype=np.float32
        )
        self.cube_size = float(CONFIG["cube"]["size"])
        self.cube_half_size = self.cube_size / 2.0
        self.cube_mass = float(CONFIG["cube"]["mass"])
        self.cube_initial_position = np.asarray(
            CONFIG["cube"]["position"], dtype=np.float32
        )
        self.cube_initial_orientation = np.asarray(
            CONFIG["cube"]["orientation_wxyz"], dtype=np.float32
        )
        self.goal_position = np.asarray(
            CONFIG["pick_place"]["goal_position"], dtype=np.float32
        )

        pp = CONFIG["pick_place"]
        self.goal_xy_tolerance = float(pp["goal_xy_tolerance_m"])
        self.goal_z_tolerance = float(pp["goal_z_tolerance_m"])
        self.stable_linear_velocity = float(pp["stable_linear_velocity_mps"])
        self.stable_angular_velocity = float(pp["stable_angular_velocity_radps"])
        self.required_stable_steps = int(pp["stable_steps"])
        self.release_open_qpos_threshold = float(
            CONFIG["gripper"]["release_open_qpos_threshold"]
        )

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.82, 0.72, 0.62],
            target=[0.40, 0.06, 0.10],
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
        super()._load_agent(
            options,
            sapien.Pose(p=self.robot_base_position.tolist()),
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        cube_density = self.cube_mass / (self.cube_size ** 3)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3,
            density=cube_density,
        )
        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(
                base_color=[0.85, 0.10, 0.10, 1.0]
            ),
        )
        builder.set_initial_pose(
            sapien.Pose(
                p=self.cube_initial_position.tolist(),
                q=self.cube_initial_orientation.tolist(),
            )
        )
        self.cube = builder.build(name="pick_place_cube")

        # Goal marker is visual-only: it must not affect dynamics.
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_box_visual(
            half_size=[0.035, 0.035, 0.001],
            material=sapien.render.RenderMaterial(
                base_color=[0.10, 0.80, 0.20, 0.45]
            ),
        )
        goal_builder.set_initial_pose(
            sapien.Pose(
                p=[
                    float(self.goal_position[0]),
                    float(self.goal_position[1]),
                    float(CONFIG["table"]["surface_z"]) + 0.001,
                ]
            )
        )
        self.goal_marker = goal_builder.build_static(name="place_goal_marker")

        gripper_cfg = CONFIG["gripper"]
        links_map = self.agent.robot.links_map
        left_names = list(gripper_cfg["left_contact_links"])
        right_names = list(gripper_cfg["right_contact_links"])

        missing = [
            name
            for name in (left_names + right_names)
            if name not in links_map
        ]
        if missing:
            raise KeyError(f"Missing gripper contact links: {missing}")

        self.left_contact_links = [links_map[name] for name in left_names]
        self.right_contact_links = [links_map[name] for name in right_names]
        self.contact_force_threshold = float(
            gripper_cfg["contact_force_threshold_n"]
        )

        active_names = [joint.name for joint in self.agent.robot.active_joints]
        name_to_index = {name: i for i, name in enumerate(active_names)}
        gripper_joint_names = list(self.agent.gripper_joint_names)
        self.gripper_joint_indices = [
            name_to_index[name] for name in gripper_joint_names
        ]

        self._stable_place_steps = torch.zeros(
            (self.num_envs,),
            dtype=torch.int32,
            device=self.device,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)

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
            initial_qpos = torch.cat([arm_qpos, gripper_qpos], dim=0)
            initial_qpos = initial_qpos.unsqueeze(0).repeat(batch_size, 1)
            self.agent.robot.set_qpos(initial_qpos)
            self.agent.robot.set_qvel(torch.zeros_like(initial_qpos))

            cube_p = torch.tensor(
                self.cube_initial_position,
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            cube_q = torch.tensor(
                self.cube_initial_orientation,
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            self.cube.set_pose(Pose.create_from_pq(p=cube_p, q=cube_q))
            zero = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)
            self.cube.set_linear_velocity(zero)
            self.cube.set_angular_velocity(zero)

            self._stable_place_steps[env_idx] = 0

    def _get_tcp_pose(self):
        if hasattr(self.agent, "tcp_pose"):
            return self.agent.tcp_pose
        if hasattr(self.agent, "tcp") and hasattr(self.agent.tcp, "pose"):
            return self.agent.tcp.pose
        raise AttributeError("Agent has neither tcp_pose nor tcp.pose")

    def _compute_side_contact_force(self, links):
        force_norms = []
        for link in links:
            vector = self.scene.get_pairwise_contact_forces(link, self.cube)
            force_norms.append(torch.linalg.norm(vector, dim=1))
        return torch.stack(force_norms, dim=0).sum(dim=0)

    def evaluate(self):
        cube_p = self.cube.pose.p
        goal = torch.as_tensor(
            self.goal_position,
            dtype=cube_p.dtype,
            device=cube_p.device,
        ).unsqueeze(0)

        delta = cube_p - goal
        goal_xy_error = torch.linalg.norm(delta[:, :2], dim=1)
        goal_z_error = torch.abs(delta[:, 2])
        within_xy = goal_xy_error <= self.goal_xy_tolerance
        within_z = goal_z_error <= self.goal_z_tolerance

        left_force = self._compute_side_contact_force(self.left_contact_links)
        right_force = self._compute_side_contact_force(self.right_contact_links)
        left_contact = left_force >= self.contact_force_threshold
        right_contact = right_force >= self.contact_force_threshold

        qpos = self.agent.robot.get_qpos()[:, self.gripper_joint_indices]
        is_gripper_open = torch.all(
            qpos <= self.release_open_qpos_threshold,
            dim=1,
        )
        no_finger_contact = ~(left_contact | right_contact)
        released = is_gripper_open & no_finger_contact

        linear_speed = torch.linalg.norm(self.cube.linear_velocity, dim=1)
        angular_speed = torch.linalg.norm(self.cube.angular_velocity, dim=1)
        is_stable = (
            (linear_speed <= self.stable_linear_velocity)
            & (angular_speed <= self.stable_angular_velocity)
        )

        stable_candidate = within_xy & within_z & released & is_stable
        self._stable_place_steps = torch.where(
            stable_candidate,
            self._stable_place_steps + 1,
            torch.zeros_like(self._stable_place_steps),
        )
        success = self._stable_place_steps >= self.required_stable_steps

        return {
            "success": success,
            "goal_xy_error_m": goal_xy_error,
            "goal_z_error_m": goal_z_error,
            "within_goal_xy": within_xy,
            "within_goal_z": within_z,
            "is_gripper_open": is_gripper_open,
            "released": released,
            "is_stable": is_stable,
            "stable_place_steps": self._stable_place_steps.clone(),
            "cube_linear_speed_mps": linear_speed,
            "cube_angular_speed_radps": angular_speed,
            "left_contact_force": left_force,
            "right_contact_force": right_force,
            "left_contact": left_contact,
            "right_contact": right_contact,
        }

    def _get_obs_extra(self, info: dict):
        tcp_pose = self._get_tcp_pose()
        cube_p = self.cube.pose.p
        goal = torch.as_tensor(
            self.goal_position,
            dtype=cube_p.dtype,
            device=cube_p.device,
        ).unsqueeze(0).repeat(cube_p.shape[0], 1)

        obs = {
            "tcp_pose": tcp_pose.raw_pose,
            "cube_pose": self.cube.pose.raw_pose,
            "goal_position": goal,
        }
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_cube_pos=cube_p - tcp_pose.p,
                cube_to_goal_pos=goal - cube_p,
                tcp_to_goal_pos=goal - tcp_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Week1 diagnostic reward only. Do not train with this reward yet.
        tcp_pose = self._get_tcp_pose()
        reach_dist = torch.linalg.norm(self.cube.pose.p - tcp_pose.p, dim=1)
        reach_reward = 1.0 - torch.tanh(5.0 * reach_dist)
        goal_reward = 1.0 - torch.tanh(10.0 * info["goal_xy_error_m"])
        return reach_reward + goal_reward + info["success"].float() * 2.0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 4.0
