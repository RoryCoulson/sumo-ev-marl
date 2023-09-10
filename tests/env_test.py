from pettingzoo.test import api_test
import sumo_ev_rl


def test_api():
    env = sumo_ev_rl.env(
        net_file="../nets/2_station_strip/2_station_strip.net.xml",
        sim_file="../nets/2_station_strip/2_station_strip.sumocfg",
        output_file="outputs/2_station_strip/test",
        enable_gui=False,
        seconds=500,
    )
    api_test(env)
    env.close()


test_api()
