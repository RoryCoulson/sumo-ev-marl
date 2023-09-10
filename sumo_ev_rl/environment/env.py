import os
import random
from typing import Optional, Tuple, Union
import sys
import gymnasium as gym
import traci
import numpy as np
import pandas as pd
import sumolib
from pathlib import Path
from .charging_station import ChargingStation
from pettingzoo import AECEnv
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.utils import seeding
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils import agent_selector, wrappers
import logging

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


def env(**kwargs):
    # PettingZoo environment instance
    env = SumoEVEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEVEnvironment(gym.Env):
    logging.basicConfig(level=logging.DEBUG)

    metadata = {
        "render_modes": ["human"],
    }

    # To support multi-client TraCI
    CONNECTION_LABEL = 0

    def __init__(
        self,
        net_file: str,
        sim_file: str,
        begin_time: int = 0,
        seconds: int = 10000,
        enable_gui: bool = False,
        sumo_seed: Union[str, int] = "random",
        sumo_warnings: bool = True,
        add_per_agent_info: bool = True,
        virtual_display: Tuple[int, int] = (3200, 1800),
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        add_system_info: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        output_file: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:

        self._net = net_file
        self._sim = sim_file
        self.enable_gui = enable_gui
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.display = None

        if self.enable_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        self.begin_time = begin_time
        self.sim_max_time = seconds
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.sumo_seed = sumo_seed
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEVEnvironment.CONNECTION_LABEL)
        SumoEVEnvironment.CONNECTION_LABEL += 1
        # TraCI connection with SUMO
        self.conn = None
        self.cumulative_mean_wait_time = 0
        self.cumulative_total_wait_time = 0
        self.mean_rewards = 0

        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])
            traci_connection = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net],
                        label="init_connection" + self.label)
            traci_connection = traci.getConnection(
                "init_connection" + self.label)

        self.cs_ids = list(traci_connection.chargingstation.getIDList())
        self.cs_edges = {}
        for cs_id in self.cs_ids:
            lane = traci_connection.chargingstation.getLaneID(cs_id)
            edge = traci_connection.lane.getEdgeID(lane)
            self.cs_edges[cs_id] = edge

        # Initialise charging station agents
        self.charging_stations = {
            cs: ChargingStation(self, cs, traci_connection)
            for cs in self.cs_ids
        }
        # Initialise the stations' closest stations
        for cs in self.charging_stations.values():
            cs.get_close_stations()

        logging.debug(f"Charging stations: {self.charging_stations}")

        traci_connection.close()

        self.charging_stations_busy_vals = {cs: 0 for cs in self.cs_ids}
        self.cumulative_rewards = [0 for _ in self.cs_ids]
        self.cumulative_diff_rewards = [0 for _ in self.cs_ids]
        self.prev_rewards = {cs: 0 for cs in self.cs_ids}

        self.cumulative_penalties = 0
        self.curr_sim_step = 0
        self.num_steps = 0
        self.total_metrics = []

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.output_file = output_file
        self.observations = {cs: None for cs in self.cs_ids}
        self.rewards = {cs: None for cs in self.cs_ids}

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-c",
            self._sim,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.enable_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.conn = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.conn = traci.getConnection(self.label)
        if self.enable_gui or self.render_mode is not None:
            self.conn.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        logging.debug("Reset")
        if self.episode != 0:
            self.num_steps = self.curr_sim_step
            self.close()
            self.save_csv(self.output_file, self.episode)
        self.episode += 1
        self.metrics = []
        if seed is not None:
            self.sumo_seed = seed

        self._start_simulation()

        self.charging_stations = {
            cs: ChargingStation(self, cs, self.conn)
            for cs in self.cs_ids
        }
        # initialise the stations' close stations
        for cs in self.charging_stations.values():
            cs.get_close_stations()

        self.vehicles = dict()

        return self._compute_observations()

    # Return current SUMO simulation second
    @property
    def sim_step(self) -> float:
        return self.conn.simulation.getTime()

    # Return only the EVs currently in the simulation
    def get_evs(self):
        vehicles = self.conn.vehicle.getIDList()
        evs = [v for v in vehicles if self.conn.vehicle.getParameter(
            v, "has.battery.device") == "true"]
        return evs

    # Dynamically update the colour of the vehicles according to their battery levels
    def update_color(self):
        evs = self.get_evs()
        for vehicle in evs:
            max_battery = float(self.conn.vehicle.getParameter(
                vehicle, 'device.battery.maximumBatteryCapacity'))
            battery = float(self.conn.vehicle.getParameter(
                vehicle, 'device.battery.actualBatteryCapacity'))
            # Set random battery to vehicle when entering the simulation
            if battery == -1:
                battery = random.randint(1, max_battery)
                self.conn.vehicle.setParameter(
                    vehicle, 'device.battery.actualBatteryCapacity', str(battery))

            battery_percent = (battery/max_battery) * 100
            # Green for full and red for low/no charge
            green = battery_percent * 2.55
            red = 255 - green

            # Update colour of the vehicles:
            self.conn.vehicle.setColor(vehicle, [red, green, 0, 255])

    # Run simulation step by applying actions, calculating rewards and computing next steps observations
    def step(self, actions: Union[dict, int]):
        logging.debug("Step")
        logging.debug("Actions: {actions}")

        if actions is None or actions == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(actions)
            self._run_steps()

        self.update_color()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        # Episode ends when reached max seconds
        info = self._compute_info()

        return observations, rewards, dones, info

    def _run_steps(self):
        self._sumo_step()

    # Chose to reroute the vehicles for each charging station
    def _apply_actions(self, actions):
        for cs, action in actions.items():
            close_vehicle = self.charging_stations[cs].consider_vehicle
            self.handle_actions(action, close_vehicle, cs)

    # Logic for applying actions
    def handle_actions(self, action, close_vehicle, cs):
        # clear any vehicles that are now in the charging lane which were rerouted
        self.charging_stations[cs].remove_rerouted_vehicles()

        if action == 0 and close_vehicle:
            battery = float(self.conn.vehicle.getParameter(
                close_vehicle, 'device.battery.actualBatteryCapacity'))
            max_battery = float(self.conn.vehicle.getParameter(
                close_vehicle, 'device.battery.maximumBatteryCapacity'))

            battery = (battery/max_battery)

            # e.g. 20% -> 0, 100% -> -70ish
            self.charging_stations[cs].not_charged_reward += - \
                self.charging_stations[cs].get_combined_charge_reward(battery)

            logging.debug(
                f"Not charged reward: {self.charging_stations[cs].not_charged_reward}")

        # Map 1 to the close_vehicle from observation (if exists)
        elif close_vehicle:
            self.charging_stations[cs].reroute_vehicle(close_vehicle)

    def _compute_dones(self):
        dones = {cs_id: False for cs_id in self.cs_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        self.curr_sim_step = self.sim_step
        info = {"step": self.curr_sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info)

        # (computing total info metrics also)
        info2 = info.copy()
        info2["step"] = len(self.total_metrics)+1

        self.total_metrics.append(info2)

        return info

    def _compute_observations(self):
        self.observations.update(
            {cs: self.charging_stations[cs].compute_observation(
            ) for cs in self.cs_ids}
        )

        return self.observations

    def _compute_rewards(self):
        self.rewards.update(
            {cs: self.charging_stations[cs].compute_reward(
            ) for cs in self.cs_ids}
        )

        return {cs: self.rewards[cs] for cs in self.rewards.keys()}

    def observation_spaces(self, cs_id: str):
        return self.charging_stations[cs_id].observation_space

    def action_spaces(self, cs_id: str) -> gym.spaces.Discrete:
        return self.charging_stations[cs_id].action_space

    def _sumo_step(self):
        self.conn.simulationStep()

    def _get_system_info(self):
        vehicles = self.get_evs()
        waiting_times = [self.conn.vehicle.getWaitingTime(
            vehicle) for vehicle in vehicles]

        self.cumulative_total_wait_time += sum(waiting_times)
        self.cumulative_mean_wait_time += np.mean(waiting_times)

        return {
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "cumulative_total_wait_time": self.cumulative_total_wait_time,
            "cumulative_mean_wait_time": self.cumulative_mean_wait_time,
        }

    def _get_per_agent_info(self):
        accumulated_waiting_time = [
            self.charging_stations[cs].get_lane_wait_time() for cs in self.cs_ids
        ]

        info = {}
        logging.debug('Rewards:', self.rewards)

        # Get info metrics
        for i, cs in enumerate(self.cs_ids):
            info[f"{cs}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{cs}_reward"] = self.rewards[cs]

            self.cumulative_rewards[i] += self.rewards[cs]
            if self.rewards[cs] < 0:
                self.cumulative_penalties -= self.rewards[cs]

            self.cumulative_diff_rewards[i] += (
                self.rewards[cs] - self.prev_rewards[cs])

            self.prev_rewards[cs] = self.rewards[cs]

            info[f"{cs}_cumulative_rewards"] = self.cumulative_rewards[i]
            info[f"{cs}_cumulative_diff_rewards"] = self.cumulative_diff_rewards[i]

            # Reset here so that you can still plot
            self.charging_stations[cs].reset_rewards()

        info["agents_total_accumulated_waiting_time"] = sum(
            accumulated_waiting_time)
        info["cumulative_penalties"] = self.cumulative_penalties
        info["cumulative_rewards"] = sum(self.cumulative_rewards)
        mean_rewards = np.mean(list(self.rewards.values()))
        info["mean_reward"] = mean_rewards

        self.mean_rewards += mean_rewards
        info["mean_cumulative_reward"] = self.mean_rewards

        return info

    def close(self):
        if not self.conn:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()

        if self.display:
            self.display.stop()
            self.display = None

        self.conn = None

    def __del__(self):
        self.close()

    def render(self):
        # sumo-gui will already render
        if self.render_mode == "human":
            return

    # Save metrics to output csv files for plotting performance
    def save_csv(self, output_file, episode):
        if output_file is not None:
            Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.metrics).to_csv(output_file +
                                              f"_episode{episode}" + ".csv", index=False)
            pd.DataFrame(
                self.total_metrics).to_csv(
                output_file + f"_total_metrics_{self.label}" + ".csv", index=False)


# PettingZooo AECEnv interface iplementation wrapper for SUMO
class SumoEVEnvironmentPZ(AECEnv, EzPickle):
    metadata = {"render.modes": ["human"],
                "name": "sumo_ev_marl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs
        self.seed()
        self.env = SumoEVEnvironment(**self._kwargs)

        self.agents = self.env.cs_ids
        self.possible_agents = self.env.cs_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.action_spaces = {
            cs: self.env.action_spaces(cs) for cs in self.agents}
        self.observation_spaces = {
            cs: self.env.observation_spaces(cs) for cs in self.agents}

        self.rewards = {cs: 0 for cs in self.agents}
        self.terminations = {cs: False for cs in self.agents}
        self.truncations = {cs: False for cs in self.agents}
        self.infos = {cs: {} for cs in self.agents}

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.cumulative_penalties = 0
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def save_csv(self, output_file, episode):
        self.env.save_csv(output_file, episode)

    def step(self, action):
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(f"Action must be Discrete")

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env.update_color()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.env._compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
