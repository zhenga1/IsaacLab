import gymnasium as gym
from ball_following_env import BallFollowingEnv, BallFollowingEnvCfg
from . import agents
# register the environment
gym.register(
    id="Isaac-BallFollowing-v0",
    entry_point="ball_following_env:BallFollowingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_following_env:BallFollowingEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BallChasingPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)