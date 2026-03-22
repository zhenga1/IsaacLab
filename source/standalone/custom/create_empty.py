import argparse


from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Create an empty scene in Isaac Sim.")
# Append AppLauncher cli args to the parser
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()
# Launch the omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext, SimulationCfg

def main():
    """Main function to create an empty scene."""
    # Create a simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Create the camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0,0,0])

    # Reset the simulation
    sim.reset()

    print("[INFO]: Empty scene created successfully.")

    # Simulation Physics
    while simulation_app.is_running():  
        sim.step()

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation app
    simulation_app.close()
