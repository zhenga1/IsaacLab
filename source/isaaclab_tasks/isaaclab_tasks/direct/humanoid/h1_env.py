# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv
from isaaclab_tasks.direct.locomotion.somersault_env import SomersaultEnv

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import H1_CFG  # isort: skip


@configclass
class ContactSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Contact sensor for left foot
    contact_forces_LF: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/LF_FOOT",       # Relative path within the foot link
        update_period=0.0,
        history_length=6,
        debug_vis=True
    )

    # Contact sensor for right foot
    contact_forces_RF : ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/RF_FOOT",  # Relative path within the foot link
        update_period=0.0,
        history_length=6,
        debug_vis=True
    )


@configclass
class H1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 19
    observation_space = 69
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    #scene = ContactSensorSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    scene : InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    # Final robot config with both sensors
    robot: ArticulationCfg = H1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    joint_gears: list = [
        50.0,  # left_hip_yaw
        50.0,  # right_hip_yaw
        50.0,  # torso
        50.0,  # left_hip_roll
        50.0,  # right_hip_roll
        50.0,  # left_shoulder_pitch
        50.0,  # right_shoulder_pitch
        50.0,  # left_hip_pitch
        50.0,  # right_hip_pitch
        50.0,  # left_shoulder_roll
        50.0,  # right_shoulder_roll
        50.0,  # left_knee
        50.0,  # right_knee
        50.0,  # left_shoulder_yaw
        50.0,  # right_shoulder_yaw
        50.0,  # left_ankle
        50.0,  # right_ankle
        50.0,  # left_elbow
        50.0,  # right_elbow
    ]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

    ### specific to the Somersault environment
    linear_velocity_scale:float = 1.0
    torque_scale: float = 1.0


class H1Env(SomersaultEnv):
    cfg: H1EnvCfg
    """Humanoid environment for RL training."""

    def __init__(self, cfg: H1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
