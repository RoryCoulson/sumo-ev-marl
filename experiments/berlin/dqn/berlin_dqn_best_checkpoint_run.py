import os
import sys
from datetime import datetime
from sumo_ev_rl.environment.env import SumoEVEnvironment
from ray.rllib.algorithms.algorithm import Algorithm
import ray

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from sumo_ev_rl.environment.env import env

# Update to your best checkpoint, an example path may look like:
BEST_CHECKPOINT_PATH = "../../results/dqn/DQN_berlin_aa226_00000_0_2023-04-29_03-00-51/checkpoint_000500"

ray.init()

net_dir_path = "../../../"
register_env(
    "berlin",
    lambda _: PettingZooEnv(
        env(
            net_file=net_dir_path + "nets/berlin/berlin.net.xml",
            sim_file=net_dir_path + "nets/berlin/berlin.sumocfg",
            output_file="../../outputs/berlin/dqn/best/best_run",
            enable_gui=True,
            seconds=5000,
        )
    ),
)

experiment_time = str(datetime.now()).split(".")[0]
out_csv = f"../../outputs/berlin/dqn/dqn_best_checkpoint/{experiment_time}"

env = SumoEVEnvironment(
    net_file=net_dir_path + "nets/berlin/berlin.net.xml",
    sim_file=net_dir_path + "nets/berlin/berlin.sumocfg",
    output_file=out_csv,
    enable_gui=True,
    seconds=5000,
)

algo = Algorithm.from_checkpoint(BEST_CHECKPOINT_PATH)

for run in range(1, 2):
    initial_states = env.reset()
    done = {"__all__": False}
    infos = []
    obs = env.reset()

    while not done["__all__"]:
        actions = {cs: algo.compute_single_action(obs[cs], policy_id='0')
                    for cs in env.cs_ids}
        obs, r, done, _ = env.step(actions=actions)

    env.save_csv(out_csv, run)
    env.close()
