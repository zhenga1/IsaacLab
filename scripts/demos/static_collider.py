import argparse
import torch
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg
)
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# cone_spawn_cfg = sim_utils.ConeCfg(
#     radius=0.15,
#     height=0.5,
#     collision_props=sim_utils.CollisionPropertiesCfg(),
#     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
# )
# cone_spawn_cfg.func(
#     "/World/Cone", cone_spawn_cfg, translation=(0.0, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
# )
from isaaclab_assets.robots.cassie import CASSIE_CFG # isort:skip
from isaaclab_assets import H1_CFG
from isaaclab_assets import G1_CFG

def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define origins
    origins = torch.tensor([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]).to(device=sim.device)

    cone_spawn_cfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cone_spawn_cfg.func(
        "/World/Cone", cone_spawn_cfg, translation=(0.0, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )
    anymal_spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        collision_props=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    )
    anymal_spawn_cfg.func(
        "/World/ANYmal", anymal_spawn_cfg, translation=(0.0, 0.0, 0.8), orientation=(1.0, 0.0, 0.0, 0.0)
    )
    # Robots
    cassie = Articulation(CASSIE_CFG.replace(prim_path="/World/Cassie"))
    h1 = Articulation(H1_CFG.replace(prim_path="/World/H1"))
    g1 = Articulation(G1_CFG.replace(prim_path="/World/G1"))
    robots = [cassie, h1, g1]

    return robots, origins

def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            for index, robot in enumerate(robots):
                # reset dof state
                joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        for robot in robots:
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in robots:
            robot.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # design a scene
    robots, origins = design_scene(sim)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, robots, origins)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()