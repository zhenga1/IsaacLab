import math
import numpy as np

# isaacsim imports
import argparse

from isaaclab.app import AppLauncher

#argparse args
parser = argparse.ArgumentParser(description="Scaling files and documents")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### ---------------------------------###
## More imports
import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import FRANKA_PANDA_CFG, ANYMAL_B_CFG, SHADOW_HAND_CFG, KINOVA_GEN3_N7_CFG

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

    shadowhand_cfg = SHADOW_HAND_CFG.copy()
    shadowhand_cfg.prim_path = "/World/Origin.*/ShadowHand"
    shadowhand = Articulation(cfg=shadowhand_cfg)

    kinova_cfg = KINOVA_GEN3_N7_CFG.copy()
    kinova_cfg.prim_path = "/World/Origin.*/Kinova"
    kinova = Articulation(cfg=kinova_cfg)

    # Returns the scene entities
    scene_entities = {"franca": franca, "anymal": anymal, "shadowhand": shadowhand, "kinova": kinova}
    return scene_entities, origins


def reset_robot(robot, origins):
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    # set joint positions with some noise
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    joint_pos += torch.rand_like(joint_pos) * 0.1
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # clear internal buffers
    robot.reset()
    print("[INFO]: Resetting robot state...")

def apply_random_robot_joint_effects(robot, sim_dt, sim):
    # Apply random action
    # -- generate random joint efforts
    efforts = torch.randn_like(robot.data.joint_pos) * 5.0
    # -- apply action to the robot
    robot.set_joint_effort_target(efforts)
    # -- write data to sim
    robot.write_data_to_sim()
    # Perform step
    sim.step()
    # Update buffers
    robot.update(sim_dt)

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    robot_franca = entities["franca"]
    robot_anymal = entities["anymal"]
    robot_shadow_hand = entities["shadowhand"]
    robot_kinova = entities["kinova"]
    sim_dt = sim.get_physics_dt()

    count = 0
    while simulation_app.is_running():
        # Reset
        if count % 500:
            # Reset the scene
            count = 0
            gap = torch.tensor([2, 0, 0]).to(origins.device)
            reset_robot(robot=robot_franca, origins=origins)
            shift_origin1 = origins + gap
            reset_robot(robot=robot_anymal, origins=shift_origin1)
            shift_origin2 = shift_origin1 + gap
            reset_robot(robot=robot_shadow_hand, origins=shift_origin2)
            shift_origin3 = shift_origin2 + gap
            reset_robot(robot=robot_kinova, origins=shift_origin3)
        apply_random_robot_joint_effects(robot=robot_franca, sim_dt=sim_dt, sim=sim)
        apply_random_robot_joint_effects(robot=robot_anymal, sim_dt=sim_dt, sim=sim)
        apply_random_robot_joint_effects(robot=robot_shadow_hand, sim_dt=sim_dt, sim=sim)
        apply_random_robot_joint_effects(robot=robot_kinova, sim_dt=sim_dt, sim=sim)
        # Increment counter
        count += 1

def main():
    """Main Function"""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we have finished!!!
    print("[INFO]: Setup complete.")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    # Run the main function
    main()
    # close sim app
    simulation_app.close()


