import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# Append Applauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch the omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController
from isaaclab.utils.math import subtract_frame_transforms
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab_assets import UNITREE_A1_CFG
import torch

def design_scene():
    """Designs the scene. """
    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    
    # Create separate group called "Origin1", "Origin2"
    # Each group will have robot in it
    # aim = first one for the robot, second one for the ball
    origins = [[0.5, 0, 0], [-0.5, 0, 0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
    
    # Rigid Object
    robot_cfg = UNITREE_A1_CFG.copy()
    robot_cfg.prim_path = "/World/Origin.*/Robot"
    robot = Articulation(cfg=robot_cfg)
    
    ball_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    ball = RigidObject(cfg=ball_cfg)

    # Scene entities return
    scene_entities = {"unitreea1": robot, "ball": ball}
    return scene_entities, torch.tensor(origins, dtype=torch.float32)

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins:torch.Tensor):
    """Runs the simulation loop"""
    # Extract simulation 
    robot = entities["unitreea1"]
    ball = entities["ball"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500:
            # reset counter
            count = 0
            # reset the scene entities (Process the robot)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint position with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.randn_like(joint_pos) * 0.1 # randomize the joint positions
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")

            # reset the ball
            root_state = ball.data.default_root_state.clone()
            root_state[:, :3] += (origins + torch.tensor([0.0, 0.0, 0.3], device=origins.device))
            ball.write_root_pose_to_sim(root_state[:, :7])
            ball.write_root_velocity_to_sim(root_state[:, 7:])
            ball.reset()
            print("[INFO]: Resetting ball state...")
            #
        
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        robot.update(sim_dt)

def main():
    """Main function to run the simulation."""
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design the scene
    entities, origins = design_scene()
    # Convert origins to tensor in the correct device
    origins = torch.tensor(origins, device=sim.device)
    # Run the simulator
    sim.reset() # Reset the simulation
    
    run_simulator(sim, entities, origins)

            

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation app
    simulation_app.close()