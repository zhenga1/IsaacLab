import math
import torch
from typing import Dict, Any

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets import UNITREE_A1_CFG

import isaaclab.envs.mdp as mdp

@configclass
class BallFollowingEnvCfg(InteractiveSceneCfg):
    """Configuration for the ball following environment."""
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
                          spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    
    # robots
    unitree_robot = UNITREE_A1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    # ball 
    ball = RigidObjectCfg(
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
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0)
    )

@configclass
class ActionCfg:
    """These here are actions specifically for the MDP."""
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)

@configclass
class ObservationCfg:
    """Observations specific to the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Observations for the policy group
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)

@configclass
class BallFollowingEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the ball following environment."""
    # scene configuration
