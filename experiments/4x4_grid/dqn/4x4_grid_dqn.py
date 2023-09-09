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
from ray.rllib.algorithms.dqn import DQNTFPolicy
from ray import air
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

ray.init()

net_dir_path = "../../../"
register_env(
    "4x4_grid",
    lambda _: PettingZooEnv(
        env(
            net_file=net_dir_path + "nets/4x4_grid/4x4_grid.net.xml",
            sim_file=net_dir_path + "nets/4x4_grid/4x4_grid.sumocfg",
            output_file="../../outputs/4x4_grid/dqn/dqn",
            enable_gui=True,
            seconds=5000,


        )
    ),
)

config = DQNConfig()

config.exploration_config = {
    "type": "EpsilonGreedy",
    "initial_epsilon": 1.0,
    "final_epsilon": 0.01,
    "epsilon_timesteps": 5000,
}

config.environment("4x4_grid")
config.multi_agent(
    policies={"0": (DQNTFPolicy, spaces.Box(low=np.zeros(
        2 + CLOSEST_STATIONS_NUM), high=np.ones(2 + CLOSEST_STATIONS_NUM)), spaces.Discrete(2), {})},
    policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "0",
)

config.rollouts(num_rollout_workers=1, rollout_fragment_length=50)
config.training(lr=0.001)

tuner = tune.Tuner(
    "DQN",
    run_config=air.RunConfig(
        stop={"timesteps_total": 500000},
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=5,
            checkpoint_at_end=True,
            checkpoint_score_attribute="episode_reward_mean"
        ),
        name="dqn",
        local_dir="../../results",
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
