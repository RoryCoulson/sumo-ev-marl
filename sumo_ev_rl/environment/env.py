"""SUMO Environment for Traffic Signal Control."""

import os
import random
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from .charging_station import ChargingStation
from pyvirtualdisplay.smartdisplay import SmartDisplay

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEVEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEVEnvironment(gym.Env):
    """SUMO Environment for EV Charging Stations.

    TODO: reword this for ev when done:

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The time in seconds the simulation must end. Default: 3600
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict): String with the name of the reward function used by the agents, a reward function, or dictionary with reward functions assigned to individual traffic lights by their keys.
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        sim_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 10000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict] = "battery",
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        # fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata[
            "render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._sim = sim_file
        self.use_gui = use_gui

        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        # self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEVEnvironment.CONNECTION_LABEL)
        SumoEVEnvironment.CONNECTION_LABEL += 1
        self.sumo = None
        self.cumulative_mean_wait_time = 0
        self.cumulative_total_wait_time = 0
        self.mean_rewards = 0

        if LIBSUMO:
            # Start only to retrieve traffic light information
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])
            conn = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net],
                        label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label)

        self.cs_ids = list(conn.chargingstation.getIDList())
        self.cs_edges = {}
        for cs_id in self.cs_ids:
            lane = conn.chargingstation.getLaneID(cs_id)
            edge = conn.lane.getEdgeID(lane)
            self.cs_edges[cs_id] = edge

        if isinstance(self.reward_fn, dict):
            self.charging_stations = {
                cs: ChargingStation(
                    self,
                    cs,
                    self.reward_fn[cs],
                    conn,
                )
                for cs in self.reward_fn.keys()
            }
        else:
            self.charging_stations = {
                cs: ChargingStation(
                    self,
                    cs,
                    self.reward_fn,
                    conn,
                )
                for cs in self.cs_ids
            }
        print('charging_stations:', self.charging_stations)

        self.charging_stations_busy_vals = {cs: 0 for cs in self.cs_ids}

        conn.close()

        self.cumulative_rewards = [0 for _ in self.cs_ids]
        self.cumulative_diff_rewards = [0 for _ in self.cs_ids]
        self.prev_rewards = {cs: 0 for cs in self.cs_ids}

        self.cumulative_penalties = 0
        self.curr_sim_step = 0
        self.num_steps = 0
        self.total_metrics = []

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))  # ?
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
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
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(
                    ["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        if self.use_gui or self.render_mode is not None:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)
        print('reset')
        if self.episode != 0:
            self.num_steps = self.curr_sim_step

            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []
        if seed is not None:
            self.sumo_seed = seed

        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.charging_stations = {
                cs: ChargingStation(
                    self,
                    cs,
                    self.reward_fn[cs],
                    self.sumo,
                )
                for cs in self.reward_fn.keys()
            }
        else:
            self.charging_stations = {
                cs: ChargingStation(
                    self,
                    cs,
                    self.reward_fn,
                    self.sumo,
                )
                for cs in self.cs_ids
            }
        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.cs_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()

    def get_evs(self):
        vehicles = self.sumo.vehicle.getIDList()
        evs = [v for v in vehicles if self.sumo.vehicle.getParameter(
            v, "has.battery.device") == "true"]

        return evs

    def remove_stuck(self):
        pass
        # #todo remove vehicles stuck at junction
        # vehicles = self.sumo.vehicle.getIDList()
        # # for v in vehicles:
        # #     if self.sumo.vehicle.atJunction(v) and self.sumo.vehicle.getWaitingTime(v) > 20:
        # #         self.sumo.vehicle.remove(v)
        # #todo remove vehicles repeatedly trying to change lanes
        # self.sumo.vehicle.getLaneChangeState
        # # colliding_vehicles = self.sumo.simulation.getCollidingVehiclesIDList()
        # # print('colliding_vehicles:', colliding_vehicles)
        # # for v in colliding_vehicles:
        # #     self.sumo.vehicle.remove(v)

    def update_color(self):
        # Dynamically change colour of the vehicles depending on their batteries
        evs = self.get_evs()

        for vehicle in self.sumo.vehicle.getIDList():
            self.sumo.vehicle.setLaneChangeMode(vehicle, 1)

        for vehicle in evs:
            max_battery = float(self.sumo.vehicle.getParameter(
                vehicle, 'device.battery.maximumBatteryCapacity'))

            battery = float(self.sumo.vehicle.getParameter(
                vehicle, 'device.battery.actualBatteryCapacity'))

            if battery == -1:  # ?
                # ? change '1' to something like within range of the closest charging station?
                battery = random.randint(1, max_battery)
                self.sumo.vehicle.setParameter(
                    vehicle, 'device.battery.actualBatteryCapacity', str(battery))

            battery_percent = (battery/max_battery) * 100

            # # (for smooth battery colouring)
            green = battery_percent * 2.55
            red = 255 - green

            # # binary colouring to tell if low or not
            # curr_color = self.sumo.vehicle.getColor(vehicle)
            # # (only setting colour at start)
            # if curr_color == (128, 128, 0, 255):
            #     # matching with classes
            #     if battery_percent <= 10:
            #         red = 255
            #         green = 0
            #     elif battery_percent <= 20:
            #         red = 175
            #         green = 80
            #     elif battery_percent <= 30:
            #         red = 125
            #         green = 130
            #     else:
            #         green = 255
            #         red = 0

            # update color of the vehicles:
            self.sumo.vehicle.setColor(vehicle, [red, green, 0, 255])

    def step(self, actions: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        print('STEP')
        print(f'ACTIONS: {actions}')

        # ? No action what to do? Nothing?
        # No action, follow fixed TL defined in self.phases <- update
        if actions is None or actions == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(actions)
            self._run_steps()

        self.update_color()
        self.remove_stuck()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        # ? update this? maybe not
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()

        if self.single_agent:
            return observations[self.cs_ids[0]], rewards[self.cs_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _run_steps(self):
        # ? no special update needed for each station no within step?
        self._sumo_step()
        # for cs in self.cs_ids:
        #     self.charging_stations[cs].update()

    def _apply_actions(self, actions):
        """Chose to reroute the next vehicle for the charging station

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_vehicles
                     If multiagent, actions is a dict {cs_id : ?} #?
        """

        # self.single_agent = False  # todo: remove the single_agent stuff
        if self.single_agent:
            # close_vehicles = self.observations[cs]
            cs = list(self.charging_stations.keys())[0]
            close_vehicle = self.charging_stations[cs].consider_vehicle
            self.handle_actions(actions, close_vehicle, cs)
        else:
            for cs, action in actions.items():
                close_vehicle = self.charging_stations[cs].consider_vehicle
                self.handle_actions(action, close_vehicle, cs)

    # Logic for applying actions
    def handle_actions(self, action, close_vehicle, cs):
        # clear any vehicles that are now in the charging lane which were rerouted
        self.charging_stations[cs].remove_rerouted_vehicles()

        if action == 0 and close_vehicle:
            battery = float(self.sumo.vehicle.getParameter(
                close_vehicle, 'device.battery.actualBatteryCapacity'))
            max_battery = float(self.sumo.vehicle.getParameter(
                close_vehicle, 'device.battery.maximumBatteryCapacity'))

            battery = (battery/max_battery)

            # e.g. 20% -> 0, 100% -> -70ish
            self.charging_stations[cs].not_charged_reward += - self.charging_stations[cs].get_combined_charge_reward(
                battery)

            print('----------NOT CHARGED REWARD:',
                  self.charging_stations[cs].not_charged_reward)

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

    @property
    def observation_space(self):
        """Return the observation space of a charging station.

        Only used in case of single-agent environment.
        """
        return self.charging_stations[self.cs_ids[0]].observation_space

    @property
    def action_space(self):
        """Return the action space of a charging station.

        Only used in case of single-agent environment.
        """
        return self.charging_stations[self.cs_ids[0]].action_space

    def observation_spaces(self, cs_id: str):
        """Return the observation space of a charging station."""
        return self.charging_stations[cs_id].observation_space

    def action_spaces(self, cs_id: str) -> gym.spaces.Discrete:
        """Return the action space of a charging station."""
        return self.charging_stations[cs_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()

    def _get_system_info(self):
        vehicles = self.get_evs()
        # speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]

        waiting_times = [self.sumo.vehicle.getWaitingTime(
            vehicle) for vehicle in vehicles]

        self.cumulative_total_wait_time += sum(waiting_times)
        self.cumulative_mean_wait_time += np.mean(waiting_times)  # ???

        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            # "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            # "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            "cumulative_total_wait_time": self.cumulative_total_wait_time,
            "cumulative_mean_wait_time": self.cumulative_mean_wait_time,


        }

    def _get_per_agent_info(self):
        stopped = [self.charging_stations[cs].get_total_queued()
                   for cs in self.cs_ids]
        accumulated_waiting_time = [
            self.charging_stations[cs].get_lane_wait_time() for cs in self.cs_ids
        ]
        average_speed = [self.charging_stations[cs].get_average_speed()
                         for cs in self.cs_ids]
        info = {}
        print('REWARDS:', self.rewards)

        for i, cs in enumerate(self.cs_ids):
            info[f"{cs}_stopped"] = stopped[i]
            info[f"{cs}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{cs}_average_speed"] = average_speed[i]
            info[f"{cs}_reward"] = self.rewards[cs]

            # if self.charging_stations[cs].consider_vehicle:
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

        info["agents_total_stopped"] = sum(stopped)
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
        """Close the environment and stop the SUMO simulation."""
        if self.sumo is None:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self.close()

    def render(self):
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame
        elif self.render_mode == "rgb_array":
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name +
                      f"_conn{self.label}_ep{episode}" + ".csv", index=False)

            df = pd.DataFrame(self.total_metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name +
                      f"_total_metrics_{self.label}" + ".csv", index=False)

    def encode(self, state, cs_id):
        """Encode the state of the charging station into a hashable object."""

        battery = state
        # ? already a hashable object
        return tuple(battery)

    def _discretize_density(self, density):
        return min(int(density * 10), 9)


class SumoEVEnvironmentPZ(AECEnv, EzPickle):
    """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo.

    For more information, see https://pettingzoo.farama.org/api/aec/.

    The arguments are the same as for :py:class:`sumo_ev_rl.environment.env.SumoEVEnvironment`.
    """

    metadata = {"render.modes": ["human", "rgb_array"],
                "name": "sumo_ev_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        """Initialize the environment."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEVEnvironment(**self._kwargs)

        self.agents = self.env.cs_ids
        self.possible_agents = self.env.cs_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {
            a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {
            a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        """Set the seed for the environment."""
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
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
        """Return the observation space for the agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for the agent."""
        return self.action_spaces[agent]

    def observe(self, agent):
        """Return the observation for the agent."""
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.env.close()

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render(mode)

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file."""
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        """Step the environment."""

        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({})."
                "It is currently {}".format(
                    agent, self.action_spaces[agent].n, action)
            )

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env.update_color()
            self.env.remove_stuck()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.env._compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        # ? why set to zero and then accumulate?
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
