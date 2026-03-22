#!/usr/bin/env python3

"""Script to train a policy for ball following with Unitree A1."""

import argparse
import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an agent to follow a ball.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BallFollowing-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after the app is launched."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

# import the environment configuration
from ball_following_env import BallFollowingEnv, BallFollowingEnvCfg

# register the environment
gym.register(
    id="BallFollowing-v0",
    entry_point="ball_following_env:BallFollowingEnv",
    kwargs={"cfg": BallFollowingEnvCfg()},
)


def main():
    """Train with a simple policy loop."""
    
    # create environment configuration
    env_cfg = BallFollowingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # print environment information
    print(f"Environment: {args_cli.task}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # set seed
    if args_cli.seed is not None:
        env.seed(args_cli.seed)
    
    # Simple training loop for demonstration
    # In practice, you'd use a proper RL library like stable-baselines3 or RSL-RL
    
    print("Starting training...")
    
    # reset environment
    obs, _ = env.reset()
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    
    for step in range(1000):  # Run for 1000 steps
        # random actions for now (replace with actual policy)
        actions = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device) * 0.1
        
        # step environment
        obs, rewards, dones, truncated, infos = env.step(actions)
        
        # accumulate rewards
        episode_rewards += rewards
        
        # log progress
        if step % 100 == 0:
            avg_reward = episode_rewards.mean().item()
            print(f"Step {step}: Average reward = {avg_reward:.2f}")
            
            # reset episode rewards for environments that are done
            reset_mask = dones | truncated
            episode_rewards[reset_mask] = 0.0
    
    print("Training complete!")
    
    # close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()