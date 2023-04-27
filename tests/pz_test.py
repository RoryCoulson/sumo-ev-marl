from pettingzoo.test import api_test, parallel_api_test

import sumo_ev_rl


def test_api():
    env = sumo_ev_rl.env(
        net_file="../nets/ev_stations-Rory/2_station_strip/2_station_strip.net.xml",
        sim_file="../nets/ev_stations-Rory/2_station_strip/2_station_strip.sumocfg",
        out_csv_name="outputs/2_station_strip/test",
        use_gui=False,
        num_seconds=100,
    )
    api_test(env)
    env.close()


def test_parallel_api():
    env = sumo_ev_rl.parallel_env(
        net_file="../nets/ev_stations-Rory/2_station_strip/2_station_strip.net.xml",
        sim_file="../nets/ev_stations-Rory/2_station_strip/2_station_strip.sumocfg",
        out_csv_name="outputs/2_station_strip/test",
        use_gui=False,
        num_seconds=100,
    )
    parallel_api_test(env, num_cycles=10)
    env.close()


if __name__ == "__main__":
    test_api()
    test_parallel_api()
