#!/usr/bin/env python3

"""Script to train a policy for ball following using RSL-RL."""

import sys
import typing_extensions
sys.modules['pip._vendor.typing_extensions'] = typing_extensions

import argparse
import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an agent to follow a ball using RSL-RL.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-BallFollowing-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum number of training iterations.")
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run name to load checkpoint from.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after the app is launched."""

import os
from datetime import datetime

# RSL-RL imports
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

# Isaac Lab imports
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# import the environment configuration
from ball_following_env import BallFollowingEnv, BallFollowingEnvCfg

# register the environment
gym.register(
    id="Isaac-BallFollowing-v0",
    entry_point="ball_following_env:BallFollowingEnv",
    kwargs={"cfg": BallFollowingEnvCfg()},
)


class RslRlVecEnvWrapper(VecEnv):
    """Wraps Isaac Lab environment for RSL-RL."""

    def __init__(self, env):
        """Initialize the wrapper."""
        self.env = env
        unwrapped_env = env
        # unwrap the environment
        while hasattr(unwrapped_env, "env"):
            unwrapped_env = unwrapped_env.env

        # store environment information
        self.num_envs = unwrapped_env.num_envs
        self.device = unwrapped_env.device
        self.max_episode_length = unwrapped_env.max_episode_length
        
        # RGL-RL specific attributes
        self.num_obs = unwrapped_env.cfg.num_observations
        self.num_privileged_obs = None  # No privileged observations
        self.num_actions = unwrapped_env.cfg.num_actions
        
        # episode statistics
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
    def reset(self):
        """Reset all environments."""
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"]
    
    def step(self, actions):
        """Step the environment."""
        # step the environment
        obs_dict, rewards, dones, truncated, extras = self.env.step(actions)
        
        # update episode length buffer
        self.episode_length_buf += 1
        
        # update reset buffer
        self.reset_buf = dones | truncated
        
        # reset episode length for terminated episodes
        self.episode_length_buf = self.episode_length_buf * (~self.reset_buf).long()
        
        return obs_dict["policy"], rewards, self.reset_buf, extras

    def get_observations(self):
        """Get observations from environment."""
        return self.env.observation_manager.compute()["policy"]

    def get_privileged_observations(self):
        """Get privileged observations (not used)."""
        return None


def train_rsl_rl(env, agent_cfg, log_dir):
    """Train using RSL-RL."""
    
    # create agent
    if agent_cfg.get("rnn", False):
        agent = ActorCriticRecurrent(
            **agent_cfg
        ).to(env.device)
    else:
        agent = ActorCritic(
            **agent_cfg
        ).to(env.device)
    
    # create algorithm
    ppo = PPO(agent, device=env.device)
    
    # configure training
    ppo.init_storage(
        num_envs=env.num_envs,
        num_transitions_per_env=args_cli.max_iterations,
        actor_obs_shape=[env.num_obs],
        critic_obs_shape=[env.num_obs],
        action_shape=[env.num_actions],
    )
    
    # training loop
    start_time = datetime.now()
    
    for iteration in range(args_cli.max_iterations):
        # rollout
        for step in range(ppo.storage.num_transitions_per_env):
            # get observations
            obs = env.get_observations()
            
            # get actions from policy
            actions = ppo.act(obs)
            
            # step environment
            next_obs, rewards, dones, infos = env.step(actions)
            
            # store transition
            ppo.storage.add_transitions(
                obs, actions, rewards, dones, infos.get("values", torch.zeros_like(rewards))
            )
        
        # compute returns
        last_obs = env.get_observations()
        ppo.storage.compute_returns(last_obs)
        
        # update policy
        mean_value_loss, mean_surrogate_loss = ppo.update()
        ppo.storage.clear()
        
        # logging
        if iteration % 10 == 0:
            current_time = datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()
            
            print(f"Iteration {iteration:4d}")
            print(f"  Mean reward: {rewards.mean():.2f}")
            print(f"  Mean episode length: {env.episode_length_buf.float().mean():.2f}")
            print(f"  Value loss: {mean_value_loss:.4f}")
            print(f"  Surrogate loss: {mean_surrogate_loss:.4f}")
            print(f"  Elapsed time: {elapsed_time:.1f}s")
            print("-" * 50)
        
        # save model
        if iteration % 100 == 0:
            save_path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save(ppo.actor_critic.state_dict(), save_path)
            print(f"Saved model to {save_path}")
    
    # save final model
    final_save_path = os.path.join(log_dir, "model_final.pt")
    torch.save(ppo.actor_critic.state_dict(), final_save_path)
    print(f"Saved final model to {final_save_path}")


def main():
    """Train with RSL-RL."""
    
    # create environment configuration
    env_cfg = BallFollowingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env)
    
    print(f"Environment: {args_cli.task}")
    print(f"Number of environments: {wrapped_env.num_envs}")
    print(f"Number of observations: {wrapped_env.num_obs}")
    print(f"Number of actions: {wrapped_env.num_actions}")
    
    # agent configuration
    agent_cfg = {
        "num_actor_obs": wrapped_env.num_obs,
        "num_privileged_obs": wrapped_env.num_privileged_obs,
        "num_critic_obs": wrapped_env.num_obs,
        "num_actions": wrapped_env.num_actions,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
    }
    
    # create log directory
    log_dir = os.path.join("logs", "ball_following", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # save configuration
    dump_yaml(os.path.join(log_dir, "env_cfg.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "agent_cfg.yaml"), agent_cfg)
    
    print("Starting training...")
    
    # train the agent
    train_rsl_rl(wrapped_env, agent_cfg, log_dir)
    
    print("Training complete!")
    
    # close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()