import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import sumo_ev_rl


def test_api():
    env = gym.make(
        "sumo-ev-rl-v0",
        num_seconds=100,
        use_gui=False,
        net_file="../nets/ev_stations-Rory/2_station_strip/2_station_strip.net.xml",
        sim_file="../nets/ev_stations-Rory/2_station_strip/2_station_strip.sumocfg",
    )
    env.reset()
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


if __name__ == "__main__":
    test_api()
