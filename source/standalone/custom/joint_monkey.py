"""
What is Joint Monkey?

-------------------------------------------------------

- Animate degree of freedom ranges for a given asset
- Demonstrate usage of DOF properties and states
- Line drawing utilities to visualize the DOF frames
"""

import math
import numpy as np

import argparse 

from isaaclab.app import AppLauncher

# argparse the arguments
parser = argparse.ArgumentParser(description="Degree of freedom ranges for assets")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### ------------------------- ###
### More imports
import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import HUMANOID_28_CFG, CARTPOLE_CFG, FRANKA_PANDA_CFG, KINOVA_GEN3_N7_CFG, ANYMAL_B_CFG, ANT_CFG

def design_scene() -> tuple[dict, list[list[float]]]:
    # Ground Plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    ## Create separate groups called "Origin 1", "Origin 2"
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origins 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origins 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    ## Articulations 
    franca_cfg = FRANKA_PANDA_CFG.copy()
    franca_cfg.prim_path = "/World/Origin.*/Franca"
    franca = Articulation(cfg=franca_cfg)

    anymal_cfg = ANYMAL_B_CFG.copy()
    anymal_cfg.prim_path = "/World/Origin.*/Anymal"
    anymal = Articulation(cfg=anymal_cfg)

    humanoid_cfg = HUMANOID_28_CFG.copy()
    humanoid_cfg.prim_path = "/World/Origin.*/ShadowHand"
    shadowhand = Articulation(cfg=humanoid_cfg)

    kinova_cfg = KINOVA_GEN3_N7_CFG.copy()
    kinova_cfg.prim_path = "/World/Origin.*/Kinova"
    kinova = Articulation(cfg=kinova_cfg)

    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Cartpole"
    cartpole = Articulation(cfg=cartpole_cfg)

    ant_cfg = ANT_CFG.copy()
    ant_cfg.prim_path = "/World/Origin.*/Ant"
    ant = Articulation(cfg=ant_cfg)

    # Returns the scene entities
    scene_entities = {"franca": franca, "anymal": anymal, "shadowhand": shadowhand, "kinova": kinova, "cartpole": cartpole, "ant": ant}
    return scene_entities, origins

def reset_robot(robot, origins):
    # root state has 2 different environments
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origins
    robot.write_root_pose_to_sim(root_state[:,:7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    # Set joint positions with some noise
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    joint_pos += torch.rand_like(joint_pos) * 0.1 # noise
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # clear internal buffer
    robot.reset()
    print("[INFO]: Resetting robot state... ")

# def apply_random_robot_joint_effects(robot, sim_dt, sim):
#     # Apply random actions
#     # generate random joint efforts
#     efforts = torch.randn_like(robot.data.joint_pos) * 5.0
#     robot.set_joint_effort_target(efforts)
#     # write data to sim
#     robot.write_data_to_sim()
#     # Perform the step
#     sim.step()
#     # Update buffers
#     robot.update(sim_dt)
import time
def animate_dofs_range_joints(robot, sim, sim_dt):
    num_dofs = robot.num_joints
    joint_names = robot.joint_names #robot.dof_names
    joint_limits = robot.root_physx_view.get_dof_limits()
    lower_limits = joint_limits[:, :, 0]
    upper_limits = joint_limits[:, :, 1]

    num_envs, num_dofs = lower_limits.shape
    for dof_idx in range(num_dofs):
        joint_names = joint_names[dof_idx] if joint_names else f"DOF {dof_idx}"
        print(f"Animating the DOFs {dof_idx}: {joint_names}")

        for t in np.linspace(0, 1, 100):
            # Sinusoidal interpolation between limits
            #pos = lower_limits[dof_idx] + (upper_limits[dof_idx] - lower_limits[dof_idx]) * 0.5 * (1 + np.sin(2 * np.pi * t))
            dof_lowers = lower_limits[:, dof_idx]
            dof_uppers = upper_limits[:, dof_idx]
            pos = dof_lowers + (dof_uppers - dof_lowers) * 0.5 * (1 + np.sin(2 * np.pi * t))

            assert torch.all(pos >= dof_lowers) and torch.all(pos <= dof_uppers)
            target_pos = robot._data.joint_pos.clone()
            target_pos[:, dof_idx] = pos
            robot.set_joint_position_target(target_pos)

            robot.write_data_to_sim()
            sim.step(render=True)
            # Update buffers
            robot.update(sim_dt)

animation_state = {
    "dof_idx": 0,
    "step_idx": 0,
    "t_values": np.linspace(0, 1, 100),
    "robot_keys": ["franca", "shadowhand"],
    "robot_idx": 0
}
# def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins:torch.Tensor):
#     # robot_franca = entities["franca"]
#     # robot_anymal = entities["anymal"]
#     # robot_shadowhand = entities["shadowhand"]
#     # robot_kinova = entities["kinova"]
#     # robot_cartpole = entities["cartpole"]
#     # robot_ant = entities["ant"]
#     sim_dt = sim.get_physics_dt()
#     t_values = animation_state['t_values']

#     count = 0
#     robot_keys = list(entities.keys())
#     while simulation_app.is_running():
#         for robot_key in robot_keys:
#             #robot_key = animation_state["robot_keys"][animation_state["robot_idx"]]
#             robot = entities[robot_key]
#             if animation_state["step_idx"] == 0 and animation_state["dof_idx"] == 0:
#                 reset_robot(robot, origins)
            
#             num_envs, num_dofs = robot.num_instances, robot.num_joints
#             joint_limits = robot.root_physx_view.get_dof_limits()
#             lower_limits = joint_limits[:, :, 0]
#             upper_limits = joint_limits[:, :, 1]

#             # Animate only one step
#             dof_idx = animation_state["dof_idx"]
#             t = t_values[animation_state["step_idx"]]
#             dof_lower = lower_limits[:, dof_idx]
#             dof_upper = upper_limits[:, dof_idx]
#             pos = dof_lower + (dof_upper - dof_lower) * (1 + np.sin(2 * np.pi * t))

#             target_pos = robot._data.joint_pos.clone()
#             target_pos[:, dof_idx] = pos
#             robot.set_joint_position_target(target_pos)

#             robot.write_data_to_sim()
#             sim.step(render=True)
#             robot.update(sim_dt)

#             # Increment state, change step count first then change robot count
#             animation_state["step_idx"] += 1
#             if animation_state["step_idx"] >= len(t_values):
#                 animation_state["step_idx"] = 0
#                 animation_state["dof_idx"] += 1
#                 if animation_state["dof_idx"] >= num_dofs:
#                     animation_state['dof_idx'] = 0
#                     animation_state["robot_idx"] += 1
#                     if animation_state["robot_idx"] >= len(animation_state["robot_keys"]):
#                         animation_state["robot_idx"] = 0

#         # # Reset
#         # if count % 500:
#         #     # Reset the scene
#         #     count = 0
#         #     gap = torch.tensor([2, 0, 0]).to(origins.device)
#         #     reset_robot(robot=robot_franca, origins=origins)
#         #     shift_origin1 = origins + gap
#         #     reset_robot(robot=robot_anymal, origins=shift_origin1)
#         #     shift_origin2 = shift_origin1 + gap
#         #     reset_robot(robot=robot_shadowhand, origins=shift_origin2)
#         #     shift_origin3 = shift_origin2 + gap
#         #     reset_robot(robot=robot_kinova, origins=shift_origin3)
#         #     shift_origin4 = shift_origin3 + gap
#         #     reset_robot(robot=robot_cartpole, origins=shift_origin4)
#         #     shift_origin5 = shift_origin4 + gap
#         #     reset_robot(robot=robot_ant, origins=shift_origin5)
        
#         # animate_dofs_range_joints(robot=robot_franca, sim_dt=sim_dt, sim=sim)
#         # animate_dofs_range_joints(robot=robot_anymal, sim_dt=sim_dt, sim=sim)
#         # animate_dofs_range_joints(robot=robot_shadowhand, sim_dt=sim_dt, sim=sim)
#         # animate_dofs_range_joints(robot=robot_kinova, sim_dt=sim_dt, sim=sim)
#         # animate_dofs_range_joints(robot=robot_cartpole, sim_dt=sim_dt, sim=sim)
#         # animate_dofs_range_joints(robot=robot_ant, sim_dt=sim_dt, sim=sim)
#         # # Increment counter
#         # count += 1

def run_simulator(sim, entities, origins):
    sim_dt = sim.get_physics_dt()
    t_values = np.linspace(0, 1, 100)
    robot_keys = list(entities.keys())

    # Reset all robots once
    gap = torch.tensor([1.5, 0.0, 0.0], device=origins.device)
    for i, key in enumerate(robot_keys):
        shift = origins + gap * i
        reset_robot(entities[key], shift)

    # Animation state
    step_idx = 0
    dof_idx_global = 0  # shared DOF index across all robots

    while simulation_app.is_running():
        for key in robot_keys:
            robot = entities[key]
            joint_limits = robot.root_physx_view.get_dof_limits()
            lower_limits = joint_limits[:, :, 0]  # [num_envs, num_dofs]
            upper_limits = joint_limits[:, :, 1]

            num_dofs = robot.num_joints
            dof_idx = dof_idx_global % num_dofs
            t = t_values[step_idx]

            # Calculate position for current DOF
            dof_lower = lower_limits[:, dof_idx]
            dof_upper = upper_limits[:, dof_idx]
            pos = dof_lower + (dof_upper - dof_lower) * 0.5 * (1 + np.sin(2 * np.pi * t))

            # Apply target positions
            target_pos = robot.data.joint_pos.clone()
            target_pos[:, dof_idx] = pos
            robot.set_joint_position_target(target_pos)
            robot.write_data_to_sim()

        # Step simulation once
        sim.step(render=True)

        # Update all robots
        for robot in entities.values():
            robot.update(sim_dt)

        # Increment animation step
        step_idx += 1
        if step_idx >= len(t_values):
            step_idx = 0
            dof_idx_global += 1  # move to next DOF


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
    # Play the simulator
    sim.reset()
    # Now we are finished
    print("[INFO]: Setup Completed! ")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    # Run the main function
    main()
    # close sim app
    simulation_app.close()