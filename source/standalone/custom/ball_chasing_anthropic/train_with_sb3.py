#!/usr/bin/env python3

"""Script to train a policy for ball following using stable-baselines3."""

import argparse
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an agent to follow a ball using SB3.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="BallFollowing-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of training iterations.")
parser.add_argument("--save_interval", type=int, default=100, help="Interval for saving checkpoints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after the app is launched."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepperCfg, VecEnvStepper

# import the environment configuration
from ball_following_env import BallFollowingEnv, BallFollowingEnvCfg

# register the environment
gym.register(
    id="BallFollowing-v0",
    entry_point="ball_following_env:BallFollowingEnv",
    kwargs={"cfg": BallFollowingEnvCfg()},
)


class IsaacLabVecEnvWrapper(VecEnv):
    """Wrapper to make Isaac Lab environment compatible with stable-baselines3."""
    
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self):
        obs, _ = self.env.reset()
        return obs["policy"]
    
    def step_async(self, actions):
        self.actions = actions
        
    def step_wait(self):
        obs, rewards, dones, truncated, infos = self.env.step(self.actions)
        
        # Convert to numpy
        obs_np = obs["policy"].cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        dones_np = (dones | truncated).cpu().numpy()
        
        return obs_np, rewards_np, dones_np, [{}] * self.num_envs
    
    def close(self):
        self.env.close()
        
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)]
    
    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]


def main():
    """Train with stable-baselines3."""
    
    # create environment configuration
    env_cfg = BallFollowingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # wrap environment for stable-baselines3
    vec_env = IsaacLabVecEnvWrapper(env)
    
    print(f"Environment: {args_cli.task}")
    print(f"Number of environments: {vec_env.num_envs}")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args_cli.save_interval * vec_env.num_envs,
        save_path="./checkpoints/",
        name_prefix="ball_following_ppo",
    )
    
    # create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=3e-4,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/",
        device=args_cli.device,
    )
    
    print("Starting training...")
    
    # train the model
    model.learn(
        total_timesteps=args_cli.max_iterations * vec_env.num_envs,
        callback=[checkpoint_callback],
        tb_log_name="ball_following_ppo",
    )
    
    print("Training complete!")
    
    # save final model
    model.save("ball_following_final")
    
    # close environment
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()