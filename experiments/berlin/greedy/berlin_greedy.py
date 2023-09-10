import argparse
import os
import sys
from datetime import datetime
from sumo_ev_rl.environment.env import SumoEVEnvironment

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

prs = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""Q-Learning Single-Intersection""")
net_dir_path = "../../../"
experiment_time = str(datetime.now()).split(".")[0]
out_csv = f"../../outputs/berlin/greedy/greedy"

env = SumoEVEnvironment(
    net_file=net_dir_path + "nets/berlin/berlin.net.xml",
    sim_file=net_dir_path + "nets/berlin/berlin.sumocfg",
    output_file=out_csv,
    enable_gui=True,
    seconds=1000,
)

for run in range(1, 51):
    initial_states = env.reset()
    done = {"__all__": False}
    infos = []

    while not done["__all__"]:
        actions = {}
        for id, cs in env.charging_stations.items():
            if not cs.consider_vehicle:
                actions[id] = 0
            else:
                closest_battery = cs.get_battery(
                    cs.consider_vehicle)
                actions[id] = int(closest_battery <= 0.2)

        s, r, done, _ = env.step(actions=actions)

    env.save_csv(out_csv, run)
    env.close()
