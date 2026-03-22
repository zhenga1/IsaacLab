import math
import torch
from typing import Dict, Any
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import UNITREE_A1_CFG


@configclass
class BallFollowingEnvCfg(DirectRLEnvCfg):
    """Configuration for the ball following environment."""

    # env
    episode_length_s = 20.0
    decimation = 4
    num_actions = 12  # 12 joints for Unitree A1
    num_observations = 49  # Robot state + ball relative position
    num_states = 0

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1 / 240, render_interval=decimation)
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True)

    # spaces
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=float)
    observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(49,), dtype=float)

    # robot
    robot_cfg: RigidObjectCfg = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ball
    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.5)),
    )


class BallFollowingEnv(DirectRLEnv):
    """Environment for training a robot to follow a ball."""

    cfg: BallFollowingEnvCfg

    def __init__(self, cfg: BallFollowingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # tracking variables
        self.action_scale = 0.25
        self.robot_dof_lower_limits = None
        self.robot_dof_upper_limits = None
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)

    def _setup_scene(self):
        """Setup the scene with ground plane and lighting."""
        # ground plane
        spawn_cfg = sim_utils.GroundPlaneCfg()
        spawn_cfg.func("/World/defaultGroundPlane", spawn_cfg)
        
        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        # create scene entities
        self.robot = Articulation(self.cfg.robot_cfg)
        self.ball = RigidObject(self.cfg.ball_cfg)
        
        # add to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (will be set to policy)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        # set up spaces
        self.single_observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.num_observations,), dtype=float
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=float
        )

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _compute_intermediate_values(self):
        """Compute intermediate values used by the environment."""
        # compute joint limits after everything is set up
        if hasattr(self, 'robot') and self.robot is not None:
            self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
            self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-processing before stepping the physics simulation."""
        self.actions = actions.clone()
        
        # scale actions to joint limits
        targets = self.robot.data.default_joint_pos + self.actions * self.action_scale
        
        # apply actions
        self.robot.set_joint_position_target(targets)

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        pass  # Actions are applied in _pre_physics_step

    def _get_observations(self) -> dict:
        """Compute observations for the environment."""
        # make sure joint limits are initialized
        if self.robot_dof_lower_limits is None:
            self._compute_intermediate_values()
            
        # robot joint positions and velocities
        dof_pos_scaled = (
            2.0
            * (self.robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        dof_vel_scaled = self.robot.data.joint_vel * 0.05

        # robot base pose and velocity
        root_states = self.robot.data.root_state_w
        robot_pos = root_states[:, :3]
        robot_quat = root_states[:, 3:7]
        robot_lin_vel = root_states[:, 7:10]
        robot_ang_vel = root_states[:, 10:13]

        # ball position
        ball_pos = self.ball.data.root_state_w[:, :3]
        
        # relative position from robot to ball
        ball_relative_pos = ball_pos - robot_pos
        
        # distance to ball
        ball_distance = torch.linalg.norm(ball_relative_pos, dim=1, keepdim=True)

        # projected gravity (useful for orientation)
        projected_gravity = torch.zeros((self.num_envs, 3), device=self.device)
        projected_gravity[:, 2] = -1.0  # simplified gravity vector

        # concatenate all observations
        obs = torch.cat([
            dof_pos_scaled,           # 12
            dof_vel_scaled,           # 12
            robot_lin_vel,            # 3
            robot_ang_vel,            # 3
            projected_gravity,        # 3
            ball_relative_pos,        # 3
            ball_distance,            # 1
            self.actions,             # 12 (previous actions)
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for the environment."""
        # robot and ball positions
        robot_pos = self.robot.data.root_state_w[:, :3]
        ball_pos = self.ball.data.root_state_w[:, :3]
        
        # distance to ball
        ball_distance = torch.linalg.norm(ball_pos - robot_pos, dim=1)
        
        # reward for getting closer to ball
        distance_reward = 1.0 / (1.0 + ball_distance)
        
        # reward for forward velocity towards ball
        robot_lin_vel = self.robot.data.root_state_w[:, 7:10]
        ball_direction = (ball_pos - robot_pos)
        ball_direction = ball_direction / (torch.linalg.norm(ball_direction, dim=1, keepdim=True) + 1e-6)
        forward_reward = torch.sum(robot_lin_vel * ball_direction, dim=1)
        forward_reward = torch.clamp(forward_reward, 0.0, 1.0)
        
        # penalty for high joint velocities (energy efficiency)
        joint_vel_penalty = -0.01 * torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        
        # penalty for large actions
        action_penalty = -0.01 * torch.sum(torch.square(self.actions), dim=1)
        
        # bonus for being very close to ball
        close_bonus = torch.where(ball_distance < 0.5, 2.0, 0.0)
        
        # total reward
        total_reward = distance_reward + 0.5 * forward_reward + joint_vel_penalty + action_penalty + close_bonus
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation conditions."""
        # robot falls over
        robot_pos = self.robot.data.root_state_w[:, :3]
        robot_height = robot_pos[:, 2]
        fallen_over = robot_height < 0.2
        
        # robot goes too far from ball
        ball_pos = self.ball.data.root_state_w[:, :3]
        ball_distance = torch.linalg.norm(ball_pos - robot_pos, dim=1)
        too_far = ball_distance > 10.0
        
        # episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # terminations
        terminated = fallen_over | too_far
        
        # truncations
        truncated = time_out
        
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at given indices."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # reset robot
        joint_pos = self.robot.data.default_joint_pos[env_ids] + torch.randn_like(
            self.robot.data.default_joint_pos[env_ids]
        ) * 0.1
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)

        # reset robot base
        robot_pos = torch.zeros(len(env_ids), 3, device=self.device)
        robot_pos[:, :2] = torch.rand(len(env_ids), 2, device=self.device) * 2.0 - 1.0  # random x,y in [-1,1]
        robot_pos[:, 2] = 0.5  # fixed height
        robot_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        robot_vel = torch.zeros(len(env_ids), 6, device=self.device)
        self.robot.write_root_state_to_sim(torch.cat([robot_pos, robot_quat, robot_vel], dim=1), env_ids)

        # reset ball to random position
        ball_pos = torch.zeros(len(env_ids), 3, device=self.device)
        ball_pos[:, :2] = torch.rand(len(env_ids), 2, device=self.device) * 4.0 - 2.0  # random x,y in [-2,2]
        ball_pos[:, 2] = 0.5  # fixed height
        ball_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        ball_vel = torch.zeros(len(env_ids), 6, device=self.device)
        self.ball.write_root_state_to_sim(torch.cat([ball_pos, ball_quat, ball_vel], dim=1), env_ids)

        # reset actions
        self.actions[env_ids] = 0.0

        # reset episode length buffer
        self.episode_length_buf[env_ids] = 0