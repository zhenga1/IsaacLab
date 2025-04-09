from __future__ import annotations

import torch
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from isaacsim.core.utils.torch import quat_rotate_inverse, get_euler_xyz

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to be within [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class SomersaultEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg
    
    """Somersault environment for humanoid locomotion."""

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        self.action_space = (int(self.cfg.action_space), 1) # action_space is the tuple shape 
        self.forces = torch.zeros(self.action_space)
        ## Starting rotation 
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.actions * self.joint_gears * self.action_scale
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)
        self.forces = forces

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                # Height of the torso
                self.torso_position[:, 2].view(-1,1),

                # Orientation (euler angles) of the torso
                normalize_angle(self.roll).unsqueeze(1),
                normalize_angle(self.pitch).unsqueeze(1),
                normalize_angle(self.yaw).unsqueeze(1),

                # Angular velocity of the Torsos 
                self.base_ang_vel * self.cfg.angular_velocity_scale,

                # Linear velocity of the torso
                self.base_lin_vel * self.cfg.linear_velocity_scale,

                # Joint Positions and velocities
                self.dof_pos * self.dof_pos_scaled,

                # Joint velocities
                self.dof_vel * self.cfg.dof_vel_scale,

                # # Joint efforts
                # self.forces * self.motor_effort_ratio * self.cfg.torque_scale,

                # Actions taken by the agent
                self.actions * self.action_scale

            ),
            dim=-1
        )
        print("The shape of the observation is simply: ", obs.shape)
        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        pitch = self.robot.data.root_euler[:, 1]
        pitch_velocity = self.robot.data.root_ang_vel[:,1]
        contact_forces_sum = self.robot.data.feet_contact_forces.norm(dim=-1).sum(dim=-1)

        if "has_flipped" not in self.extras:
            self.extras["has_flipped"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        reward, update_has_flipped = compute_rewards(
            pitch=pitch,
            pitch_velocity=pitch_velocity,
            contact_forces_sum=contact_forces_sum,
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

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
            self.torso_rotation, self.inv_start_rot, to_target, basis_vec0, basis_vec1, 2
        )
        
        vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
            torso_quat, velocity, ang_velocity, targets, torso_position
        )
        # self.robot.data.soft_joint_pos_limits is usually an array of shape : ([envs, num_joints, 2]), where the
        # last dimension is of shape [lower, upper]
        dof_lower_limits, dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 0], self.robot.data.soft_joint_pos_limits[0, :, 1]

        self.dof_pos_scaled = torch_utils.maths.unscale(self.dof_pos, dof_lower_limits, dof_upper_limits)

        root_states = self.robot.data.root_state_w.view(self.num_envs, -1)
        root_pos = root_states[:, :3] # So the first couple elements in the root states 
        root_quat = root_states[:, 3:7]
        root_lin_vel = root_states[:, 7:10]
        root_ang_vel = root_states[:, 10:13]

        # Save torso/root position
        self.torso_position = root_pos

        # Orientation of the Euler angles (radians)
        self.roll, self.pitch, self.yaw = get_euler_xyz(root_quat) # shape: (num_ens, 3)

        # Normalize orientation angles to [-pi, pi]
        self.roll = normalize_angle(self.roll)
        self.pitch = normalize_angle(self.pitch)
        self.yaw = normalize_angle(self.yaw)

        # Velocities of the local (body) frame
        self.base_lin_vel = quat_rotate_inverse(root_quat, root_lin_vel) # shape: (num_envs, 3)
        self.base_ang_vel = quat_rotate_inverse(root_quat, root_ang_vel) # shape: (num_envs, 3)






    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            # all of them
            env_ids = torch.arange(self.num_envs, device=self.sim.device)
            # reset the robot
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # default root state from the defaults (includes the default upright position and rotation)
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_dof_pos = self.robot.data.default_joint_pos[env_ids]

        # reset root torso position and its velocity
        self.robot.write_root_state_to_sim(default_root_state, env_ids=env_ids)

        # reset joint positions and velocities
        self.robot.write_joint_state_to_sim(default_dof_pos, torch.zeros_like(default_dof_pos), env_ids=env_ids)

        # reset custom environment state
        self.extras["has_flipped"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    pitch: torch.Tensor, # Pitch of the torso
    pitch_velocity: torch.Tensor, # Pitch velocity of the torso
    contact_forces_sum: torch.Tensor, # Contact forces on the feet
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
    flipped_now = torch.abs(pitch) >= flip_angle_threshold # Check if torso is flopped 
    update_has_flipped = flipped_now | flipped_success # Update the flipped success status
    flip_reward = flip_reward_scale * flipped_now.float() * (~flipped_success).float() # reward for flipping (first time)

    is_upright = torch.abs(normalize_angle(pitch)) < upright_angle_threshold # pass into normalize angle to ensure the angle is not out of bounds
    is_in_contact = contact_forces_sum > contact_force_threshold 
    landed_successfully = update_has_flipped & is_upright & is_in_contact # Check if the agent has landed successfully

    landing_reward = landing_reward_scale * landed_successfully.float() # reward for landing successfully
    action_penalty = action_penalty_scale * torch.sum(actions ** 2, dim = -1) # L2 action penalty for the actions taken by agent (magnitude)

    shaping_reward = 0.02 * pitch_velocity * (~update_has_flipped).float() # reward for pitch velocity when not flipped

    total_reward = flip_reward + landing_reward - action_penalty + shaping_reward # the total reward
    return total_reward, update_has_flipped