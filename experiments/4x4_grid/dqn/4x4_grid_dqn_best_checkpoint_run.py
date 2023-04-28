import argparse
import os
import sys
from datetime import datetime

from sumo_ev_rl.environment.env import SumoEVEnvironment

# from sumo_ev_rl.environment import SumoEVEnvironment
from ray import tune
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

# Update with new run
BEST_CHECKPOINT_PATH = "../../results/dqn/DQN_4x4_grid_aa607_00000_0_2023-04-18_18-08-41/checkpoint_000500"

if __name__ == "__main__":
    ray.init()
    RESOLUTION = (3200, 1800)
    net_dir_path = "../../../"
    # ??? needed here again?
    register_env(
        "4x4_grid",
        lambda _: PettingZooEnv(
            env(
                net_file=net_dir_path + "nets/4x4_grid/4x4_grid.net.xml",
                sim_file=net_dir_path + "nets/4x4_grid/4x4_grid.sumocfg",
                out_csv_name="../../outputs/4x4_grid/dqn/best/best_run",
                use_gui=True,
                num_seconds=1000,  # ?episode length..
                render_mode="human",
                virtual_display=RESOLUTION
            )
        ),
    )

    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default=net_dir_path + "nets/4x4_grid/4x4_grid.rou.xml",
        help="Route definition xml file.\n",
    )
    # Add parameters for the model
    # Alpha learning rate
    prs.add_argument("-a", dest="alpha", type=float, default=0.1,
                     required=False, help="Alpha learning rate.\n")
    # Gamma discount rate
    prs.add_argument("-g", dest="gamma", type=float, default=0.99,
                     required=False, help="Gamma discount rate.\n")
    # Epsilon value
    prs.add_argument("-e", dest="epsilon", type=float,
                     default=1, required=False, help="Epsilon.\n")
    # Minimum epsilon value
    prs.add_argument("-me", dest="min_epsilon", type=float,
                     default=0.005, required=False, help="Minimum epsilon.\n")
    # Epsilon decay
    prs.add_argument("-d", dest="decay", type=float, default=0.99,
                     required=False, help="Epsilon decay.\n")
    # Use GUI boolean for visualization
    prs.add_argument("-gui", action="store_true", default=False,
                     help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False,  # ?
                     help="Run with fixed timing traffic signals.\n")

    # ? (update as needed) Number of simulation seconds (length of simulation in simulation seconds)
    prs.add_argument("-s", dest="seconds", type=int, default=5000,  # 100000 (change in rou.xml also)
                     required=False, help="Number of simulation seconds.\n")
    # Print experience tuple
    prs.add_argument("-v", action="store_true", default=False,
                     help="Print experience tuple.\n")
    # Number of runs
    prs.add_argument("-runs", dest="runs", type=int,
                     default=1, help="Number of runs.\n")

    args = prs.parse_args()
    experiment_time = str(datetime.now()).split(".")[0]
    out_csv = f"../../outputs/4x4_grid/dqn/dqn_best_checkpoint/{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}"

    RESOLUTION = (3200, 1800)

    env = SumoEVEnvironment(
        net_file=net_dir_path + "nets/4x4_grid/4x4_grid.net.xml",
        sim_file=net_dir_path + "nets/4x4_grid/4x4_grid.sumocfg",
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        render_mode="human",
        virtual_display=RESOLUTION
    )

    algo = Algorithm.from_checkpoint(BEST_CHECKPOINT_PATH)

    for run in range(1, args.runs + 1):  # runs = epochs??
        initial_states = env.reset()
        # agents = {
        #     cs: my_restored_policy
        #     for cs in env.cs_ids
        # }

        done = {"__all__": False}
        infos = []

        obs = env.reset()
        print('obs:', obs)
        print('cs_ids:', env.cs_ids)

        while not done["__all__"]:
            # agents[cs].compute_actions
            # TODO what is the right function here to get the policy to run it's action? not act(), not compute_actions(), ...?

            actions = {cs: algo.compute_single_action(obs[cs], policy_id='0')
                       for cs in env.cs_ids}
            obs, r, done, _ = env.step(actions=actions)

        env.save_csv(out_csv, run)
        env.close()
