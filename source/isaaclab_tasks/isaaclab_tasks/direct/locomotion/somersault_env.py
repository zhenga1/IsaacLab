# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
import math
from isaaclab.sensors import ContactSensor
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
import carb
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class SomersaultEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1)) # inverse of the starting quaternion rotation
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        #self.contact_sensor_left = ContactSensor(cfg=self.cfg.contact_sensor_cfg_left)
        #self.contact_sensor_right = ContactSensor(cfg=self.cfg.contact_sensor_cfg_right)
        #self.contact_sensor_left.initialize()
        #self.contact_sensor_right.initialize()
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # # initialize sensor callbacks
        # self.contact_sensor_left._initialize_callback(event=None)
        # self.contact_sensor_right._initialize_callback(event=None)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        #self.scene.sensors["contact_forces_LF"] = self.contact_sensor_left
        #self.scene.sensors["contact_forces_RF"] = self.contact_sensor_right
        # add lights
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)
        

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        pitch_velocity = self.angvel_loc[:,1]
        # contact_forces_sum = (
        #     self.scene["contact_forces_LF"].data.contact_forces +
        #     self.robot.sensors["contact_forces_RF"].data.contact_forces
        # )  
        #contact_forces_left = torch.linalg.norm(self.scene["contact_forces_LF"].data.net_forces_w, dim=-1)
        #contact_forces_right = torch.linalg.norm(self.scene["contact_forces_RF"].data.net_forces_w, dim=-1)

        #contact_forces_sum = contact_forces_left + contact_forces_right

        if "has_flipped" not in self.extras:
            self.extras["has_flipped"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        reward, update_has_flipped = compute_rewards(
            torso_height=self.torso_position[:, 2],
            heading_proj=self.heading_proj,
            angular_velocity=self.ang_velocity,
            up_proj=self.up_proj,
            heading_weight=self.cfg.heading_weight,
            up_weight=self.cfg.up_weight,
            pitch=self.pitch,
            pitch_velocity=pitch_velocity,
            #contact_forces_sum=contact_forces_sum,
            actions=self.actions,
            flipped_success=self.extras["has_flipped"],
        )
        self.extras["has_flipped"] = update_has_flipped
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_position[:, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_dof_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids=env_ids)

        # reset joint positions and velocities
        self.robot.write_joint_state_to_sim(default_dof_pos, torch.zeros_like(default_dof_pos), env_ids=env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        # reset custom environment state
        self.extras["has_flipped"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    torso_height: torch.Tensor, # Height of the torso
    heading_proj: torch.Tensor, # Heading projection of the torso
    angular_velocity: torch.Tensor, # Angular velocity of the torso
    up_proj: torch.Tensor, # Up projection of the torso
    heading_weight: float, # Weight for heading reward
    up_weight: float, # Weight for up reward
    pitch: torch.Tensor, # Pitch of the torso
    pitch_velocity: torch.Tensor, # Pitch velocity of the torso
    #contact_forces_sum: torch.Tensor, # Contact forces on the feet
    actions: torch.Tensor, # Actions taken by agent
    flipped_success: torch.Tensor, ## Has flipped or not?
    flip_angle_threshold: float = 6.0, # Threshold for flip angle (in radians, 6.0 rad implying almost a full flip)
    upright_angle_threshold: float = 0.3, # Threshold for upright angle (in radians)
    contact_force_threshold: float = 50.0, # Threshold for contact forces
    flip_reward_scale: float = 5.0, # Scale for flip reward
    landing_reward_scale: float = 3.0, # Scale for landing reward
    action_penalty_scale: float = 0.01, # Scale for action penalty
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the reward for the somersault task. 
    """
    #reward = torch.zeros(root_states.shape[0], device=root_states.device)
   # Optional: Clamp to avoid extreme spin rewards
        # Reward positive pitch velocity (forward flip)

    # 1. Pitch velocity reward
    pos_reward = torch.clamp(pitch_velocity, min=0.0, max=10.0)

    # Penalize negative or zero pitch velocity to break symmetry
    neg_penalty = (pitch_velocity <= 0).float() * 0.3  # tweak 0.3 as needed

    reward = 0.5 * pos_reward - neg_penalty

    # 2. Flip completion bonus (when pitch exceeds pi and hasn’t already flipped)
    flip_completed = (pitch > math.pi) & (~flipped_success)
    reward += flip_completed.float() * 5.0
    flipped_success = flipped_success | flip_completed

    # 3. Post-flip viability bonus: upright + low angular velocity
    upright = torso_height > 0.8
    stable_spin = torch.norm(angular_velocity, dim=1) < 5.0
    reward +=  (upright & stable_spin & flipped_success).float() * 1.0

    # 4. Penalty for flopping (too low)
    flop_penalty = (torso_height < 0.5).float() 
    reward -= flop_penalty * 0.5
    return reward, flipped_success



@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
