import os
import sys

from sumo_ev_rl.environment.env import SumoEVEnvironment
from sumo_ev_rl.environment.env import env
from sumo_ev_rl.environment.charging_station import CLOSEST_STATIONS_NUM


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
# from gym import spaces
from gymnasium import spaces
# from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.ppo import PPOTF1Policy


from ray import air
from ray import tune

# from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

# import sumo_rl

if __name__ == "__main__":
    # ray.init(ignore_reinit_error=True, num_cpus=4)  # num_gpus=1
    ray.init()
    RESOLUTION = (3200, 1800)
    net_dir_path = "../../../"
    register_env(
        "4_station_strip",
        lambda _: PettingZooEnv(
            env(
                net_file=net_dir_path + "nets/ev_stations-Rory/4_station_strip/4_station_strip.net.xml",
                sim_file=net_dir_path + "nets/ev_stations-Rory/4_station_strip/4_station_strip.sumocfg",
                out_csv_name="../../outputs/4_station_strip/ppo/ppo",
                use_gui=True,
                num_seconds=5000,  # ?episode length..
                render_mode="human",
                virtual_display=RESOLUTION
            )
        ),
    )

    config = PPOConfig()
    config.environment("4_station_strip")
    config.multi_agent(
        policies={"0": (PPOTF1Policy, spaces.Box(low=np.zeros(
            2 + CLOSEST_STATIONS_NUM), high=np.ones(2 + CLOSEST_STATIONS_NUM)), spaces.Discrete(2), {})},
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "0",
    )
    # ?
    # config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)
    # config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=1)

    config.training(
        gamma=0.99,
        lambda_=0.95,
        lr=0.001,  # 0.001,
        sgd_minibatch_size=512,
        # train_batch_size=10000,  # ?
        num_sgd_iter=10,
        clip_param=0.2,
    )

    # config.no_done_at_end = True #? deprecated..

    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"timesteps_total": 2500000},
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
                checkpoint_score_attribute="episode_reward_mean"
            ),
            name="ppo",
            local_dir="../../results/4_station_strip/ppo",
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
