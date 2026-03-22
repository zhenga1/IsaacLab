"""
Animate the franca robot to follow some red cube

-------------------------------------------------

- Demonstrate usage of DOF properties and states
"""
import math 
import numpy as np
import argparse
from isaaclab.app import AppLauncher

# argparse the arguments
parser = argparse.ArgumentParser(description="Franca Robot cube follower")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### ---------- ###
### More Imports
import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets import FRANKA_PANDA_CFG
# from isaaclab.utils.math import quat_identity_tensor
# from isaaclab.utils.geometry import Pose
from isaaclab.utils.math import make_pose, unmake_pose, matrix_from_quat, subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController


def design_scene() -> tuple[dict, list[list[float]]]:
    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate Groups 1 - 2
    origins = [[0.0, 0.0, 0.0]] #, [-1.0, 0.0, 0.0]]
    # Origins 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # # Origins 2
    # prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    
    ### Articulations
    franca_cfg = FRANKA_PANDA_CFG.copy()
    franca_cfg.prim_path = "/World/Origin.*/Franca"
    franca = Articulation(cfg=franca_cfg)

    # RigidObjectCfg
    red_cube_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Obj_red",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Users/aaron/IsaacLab/source/isaaclab_assets/data/Props/CubeMultiColor/cube_multicolor.usd",
            scale=(0.2,0.2,0.2),
            # RigidBodyMaterialCfg - controls hyperparameters like static friction and the like
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
                disable_gravity=True,
                kinematic_enabled=True
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                roughness=0.1,
                metallic=0.2,
                opacity=1.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=1.0
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    red_cube = RigidObject(cfg=red_cube_cfg)

    # Returns the scene entities
    scene_entities = {"franca": franca, "red_cube": red_cube}
    return scene_entities, origins

def reset_robot(robot, origins, robot_name):
    # root state has 2 different environments
    root_state = robot.data.default_root_state.clone()
    # all environments, set root state (the first 3 numbers) to be the origins 
    root_state[:, :3] = origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    if robot_name != "red_cube":
        # Set joint positions with some noise
        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        joint_pos += torch.rand_like(joint_pos) * 0.1 # noise
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # clear the internal buffer
    robot.reset()
    print("[INFO]: Resetting the robot state ")

# def run_simulator(sim, entities, origins):
#     sim_dt = sim.get_physics_dt()
#     t_values = np.linspace(0, 1, 100)
    
#     # Reset all robots once
#     gap = torch.tensor([1.5, 0.0, 0.0], device=origins.device)
#     robot_keys = list(entities.keys())
#     for i, key in enumerate(robot_keys):
#         shift = origins + gap * i
#         reset_robot(entities[key], shift)
    
#     # Animation state
#     step_idx = 0
#     dof_idx_global = 0 # the shared DOF index across ALL the robots

#     # initialize the robots
#     franca = entities["franca"]
#     red_cube = entities["red_cube"]
    
#     def get_cube_position(sim):
#         cube = sim.scene.get_object("red_cube")
#         cube_pos = cube.data.root_pos_w
#         return cube_pos
#     def quat_identity():
#         return quat_identity_tensor()
    
#     while simulation_app.is_running():
#         # 1 Get position of red cube
#         cube_position = get_cube_position(sim)
#         # 2 Define target pose + target orientation for end-effector (EEF):
#         target_position = cube_position + np.array([0.0, 0.0, 0.1]) # small offset above cube
#         target_orientation = quat_identity()

#         # Solve the IK equation? to get the Franka robot to send it to Franka
#         ee_pose_desired = Pose()

def run_simulator(sim, entities, origins, ik_controller):
    sim_dt = sim.get_physics_dt()
    device = sim.device
    franca = entities["franca"]
    red_cube = entities["red_cube"]
    
    # Reset robots
    reset_robot(franca, origins, "franca")
    reset_robot(red_cube, origins, "red_cube")

    # Resolve indices once so we don’t recompute every step
    EE_LINK_NAME = "panda_hand"
    # pick the first attribute that exists
    if hasattr(franca, "link_name_to_index"):
        ee_body_id = franca.link_name_to_index["panda_hand"]
    elif hasattr(franca, "find_body_index"):
        ee_body_id = franca.find_body_index("panda_hand")
    elif hasattr(franca, "get_link_index"):
        ee_body_id = franca.get_link_index("panda_hand")
    else:                                   # fall-back: scan the name list
        ee_body_id = franca.body_names.index("panda_hand")
    #ee_body_id   = franca.find_link(EE_LINK_NAME)            # → int
    num_dof   = franca.data.joint_pos.shape[1]          # (N,  DoF)
    joint_ids = torch.arange(num_dof, device=device)    # tensor([0, 1, …, DoF-1])
    num_envs     = franca.data.root_pos_w.shape[0]


    # Pre-alloc IK command tensor: (N, 7) → xyz + wxyz quat
    ik_cmd = torch.zeros(num_envs, 7, device=device)
    ik_cmd[:, 3] = 1.0  # identity quat (w=1,x=y=z=0)

     # Convenience lambda for current cube position
    get_cube_pos = lambda: red_cube.data.root_pos_w              # (N,3)

    while simulation_app.is_running():
        sim.step(render=True)

        #import pdb
        #pdb.set_trace()
        cube_pos = get_cube_pos()  # (3,)
        target_pos = cube_pos + torch.tensor([0.0, 0.0, 0.1], device=cube_pos.device)

        ik_cmd[:, :3] = target_pos                           # update xyz

        # ----- tell the controller what we want
        ik_controller.set_command(ik_cmd)

        # ----- gather current state needed for IK solve
        jacobian     = franca.root_physx_view.get_jacobians()[:, ee_body_id, :, joint_ids]
        ee_pose_w    = franca.data.body_pos_w[:, ee_body_id]        # (N,3)
        ee_quat_w    = franca.data.body_quat_w[:, ee_body_id]       # (N,4)
        root_pose_w  = franca.data.root_pos_w                       # (N,3)
        root_quat_w = franca.data.root_quat_w                       # (N, 4)
        joint_pos    = franca.data.joint_pos[:, joint_ids]           # (N,DoF)

        # import pdb
        # pdb.set_trace()
        # Convert EE pose into the robot base frame
        # ee_pos_b, ee_quat_b = subtract_frame_transforms(
        #     root_pose_w,  root_quat_w, # parent = base , not world frame
        #     ee_pose_w,   ee_quat_w,    # also not world frame
        # )

        # ----- run the IK solver
        joint_targets = ik_controller.compute(
            ee_pose_w,            # (N,3)
            ee_quat_w,           # (N,4)
            jacobian,            # (N,6,DoF)
            joint_pos            # (N,DoF)
        )


        # # Create an identity quaternion
        # identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=target_pos.device)
        # rot_mat = matrix_from_quat(identity_quat).unsqueeze(0) # (1, 3, 3)

        # # Create a full pose using position + quaternion
        # target_pose = make_pose(
        #     pos=target_pos.unsqueeze(0),      # (1, 3)
        #     rot=rot_mat   # (3,3)
        # )[0]  # extract single instance

        # joint_targets = franca.compute_ik(target_pose)
        franca.set_joint_position_target(joint_targets)
        franca.write_data_to_sim()

def main():
    """Main function"""
    # Load kit function
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design the scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    #Play simulator
    sim.reset()
    # Now we are finished
    print("[INFO]: Setup Completed! ")
    franca = scene_entities["franca"]
    num_envs = franca.data.root_pos_w.shape[0]
    from isaaclab.controllers import DifferentialIKControllerCfg
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",        # track full 6-DoF pose
        use_relative_mode=False,    # absolute target, not delta
        ik_method="dls",            # damped-least-squares pseudo-inverse
        ik_params={"lambda_val": 1e-4}   # ← what you called “damping”
    )
    ik_ctrl = DifferentialIKController(
        cfg=ik_cfg,
        num_envs=num_envs,
        device=sim.device,
    )
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, ik_ctrl)

if __name__ == "__main__":
    # Run the main function
    main()
    # close sim
    simulation_app.close()
    



