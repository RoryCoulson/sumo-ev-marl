import os
import sys

from sumo_ev_rl.environment.env import env
from sumo_ev_rl.environment.charging_station import CLOSEST_STATIONS_NUM

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import ray
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOTF1Policy
from ray import air
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

ray.init()

net_dir_path = "../../../"
register_env(
    "2_station_strip",
    lambda _: PettingZooEnv(
        env(
            net_file=net_dir_path + "nets/2_station_strip/2_station_strip.net.xml",
            sim_file=net_dir_path + "nets/2_station_strip/2_station_strip.sumocfg",
            output_file="../../outputs/2_station_strip/ppo/ppo",
            enable_gui=True,
            seconds=5000,
        )
    ),
)

config = PPOConfig()
config.environment("2_station_strip")
config.multi_agent(
    policies={"0": (PPOTF1Policy, spaces.Box(low=np.zeros(
        2 + CLOSEST_STATIONS_NUM), high=np.ones(2 + CLOSEST_STATIONS_NUM)), spaces.Discrete(2), {})},
    policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "0",
)
config = config.rollouts(num_rollout_workers=1)
config.training(
    gamma=0.99,
    lambda_=0.95,
    lr=0.001,
    sgd_minibatch_size=512,
    num_sgd_iter=10,
    clip_param=0.2,
)

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"timesteps_total": 500000},
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=5,
            checkpoint_at_end=True,
            checkpoint_score_attribute="episode_reward_mean"
        ),
        name="ppo",
        local_dir="../../results/2_station_strip/ppo",
    ),
    param_space=config.to_dict(),
)
results = tuner.fit()

# Get the best result based on a particular metric.
best_result = results.get_best_result(
    metric="episode_reward_mean", mode="max")
print('best_result:', best_result)

# Get the best checkpoint corresponding to the best result.
best_checkpoint = best_result.checkpoint
print('\n\n--------best_checkpoint:\m', best_checkpoint, '----------')
