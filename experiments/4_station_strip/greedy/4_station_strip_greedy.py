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


if __name__ == "__main__":

    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    net_dir_path = "../../../"
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default=net_dir_path + "nets/ev_stations-Rory/4_station_strip/4_station_strip.rou.xml",
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
                     default=200, help="Number of runs.\n")  # number of epsiodes?

    args = prs.parse_args()
    experiment_time = str(datetime.now()).split(".")[0]
    # out_csv = f"outputs/2_station_strip/greedy/{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}"
    out_csv = f"../../outputs/4_station_strip/greedy/greedy"

    RESOLUTION = (3200, 1800)

    env = SumoEVEnvironment(
        net_file=net_dir_path + "nets/ev_stations-Rory/4_station_strip/4_station_strip.net.xml",
        # route_file=args.route,
        sim_file=net_dir_path + "nets/ev_stations-Rory/4_station_strip/4_station_strip.sumocfg",
        out_csv_name=out_csv,
        use_gui=True,
        num_seconds=args.seconds,
        render_mode="human",
        virtual_display=RESOLUTION
    )

    for run in range(1, args.runs + 1):  # runs = epochs??
        initial_states = env.reset()

        done = {"__all__": False}
        infos = []

        while not done["__all__"]:
            actions = {}
            for id, cs in env.charging_stations.items():
                if not cs.consider_vehicle:
                    actions[id] = 0
                else:
                    closest_battery = cs.get_battery_decimal(
                        cs.consider_vehicle)
                    actions[id] = int(closest_battery <= 0.2)

            print(f'actions: {actions}')
            s, r, done, _ = env.step(actions=actions)

        env.save_csv(out_csv, run)
        env.close()
