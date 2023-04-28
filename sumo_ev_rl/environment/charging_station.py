"""This module contains the ChargingStation class, which represents a charging station in the simulation."""
import os
import sys
from typing import Callable, Union
import sumolib
import traci
import numpy as np
from gymnasium import spaces
import logging

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("'SUMO_HOME' environment variable")


MAX_CLOSEST_DISTANCE = 500
MAX_RANGE = 150000
CHARGING_DURATION = 80

# Wait time given as a fraction of the estimated max wait time, currently set to 1hr
ESTIMATED_MAX_WAIT_TIME = 1*60*60
CLOSEST_STATIONS_NUM = 2


class ChargingStation:
    logging.basicConfig(level=logging.DEBUG)

    def __init__(self, env, cs_id, sumo):
        self.id = cs_id
        self.env = env
        self.sumo = sumo

        # penalties/reward counts
        self.charge_reward = 0
        self.not_charged_reward = 0

        # Get station properties
        self.consider_vehicle = None
        self.decided_vehicles = set()
        self.rerouted_vehicles = set()
        self.waiting_count = 0
        self.busy_val = 0
        self.closest_stations = None
        self.lane_id = self.sumo.chargingstation.getLaneID(self.id)
        self.lane_length = self.sumo.lane.getLength(self.lane_id)
        self.edge_id = self.sumo.lane.getEdgeID(self.lane_id)
        self.cs_end_pos = self.sumo.chargingstation.getEndPos(self.id)
        net = sumolib.net.readNet(self.env._net)
        self.incoming_lanes = net.getLane(self.lane_id).getIncoming()
        self.incoming_lanes_ids = set(
            [lane.getID() for lane in self.incoming_lanes])

        # Subscription for detecting nearby vehicles
        traci.chargingstation.subscribeContext(
            self.id, traci.constants.CMD_GET_VEHICLE_VARIABLE, MAX_CLOSEST_DISTANCE)

        # Observations: [close_vehicle_battery, busy_val, closest_busy_val, 2nd_closest_busy_val] all range from 0 to 1
        self.observation_space = spaces.Box(
            low=np.zeros(2 + CLOSEST_STATIONS_NUM, dtype=np.float32),
            high=np.ones(2 + CLOSEST_STATIONS_NUM, dtype=np.float32),
        )

        # Boolean action: either charge detected vehicle (1) or don't (0)
        self.action_space = spaces.Discrete(2)

    def reset_rewards(self):
        self.charge_reward = 0
        self.not_charged_reward = 0

    # Update vehicle's route to charge at the station
    def reroute_vehicle(self, vehicle_id: int):
        battery = self.get_battery(vehicle_id)
        # Receive the charging reward
        self.charge_reward += self.get_combined_charge_reward(
            battery)
        logging.debug(f"Charge reward: {self.charge_reward}")

        # Check if already rerouted
        last_curr_edge_id = self.sumo.vehicle.getRoadID(vehicle_id)
        vehicle_route = self.sumo.vehicle.getRoute(vehicle_id)
        charging_station_edges = self.env.cs_edges.values()

        # Don't reroute if another agent has rerouted
        if set(vehicle_route).intersection(set(charging_station_edges)):
            logging.debug(f"Vehicle already rerouted in this step")
            return

        # Reroute to charging station
        destination_edge_id = vehicle_route[-1]
        route_to_station = self.sumo.simulation.findRoute(
            last_curr_edge_id, self.edge_id).edges
        route_to_destination = self.sumo.simulation.findRoute(
            self.edge_id, destination_edge_id).edges
        self.rerouted_vehicles.add(vehicle_id)
        new_route = route_to_station[:-1] + route_to_destination
        logging.debug(f'New route: (vehicle: {vehicle_id}): {new_route}')

        self.sumo.vehicle.setRoute(vehicle_id, new_route)
        self.sumo.vehicle.setStop(
            vehicle_id, self.edge_id, pos=self.cs_end_pos,  duration=CHARGING_DURATION)

    def compute_observation(self):
        charging_station_edges = self.env.cs_edges.values()
        # Get driving distance of EVs in close proximity not already rerouted or decided upon
        dists_to_station = {v: self.get_dist_to_station(v) for v in self.get_close_evs() if not set(
            self.sumo.vehicle.getRoute(v)).intersection(set(charging_station_edges)) and v not in self.decided_vehicles}
        logging.debug(
            f"Distances to station of close evs, (station: {self.id}): {dists_to_station}")

        closest_vehicle_battery = self.get_closest_battery(dists_to_station)
        # If no close vehicle detected return observation of zeros
        if closest_vehicle_battery == None:
            return np.zeros(2 + CLOSEST_STATIONS_NUM, dtype=np.float32)

        # Get busy values of this stations and close stations
        busy_val = self.get_busy_val()
        close_busy_vals = self.get_close_busy_vals()

        observation = np.array(
            [closest_vehicle_battery, busy_val] + close_busy_vals, dtype=np.float32)
        logging.debug(f"Observation (cs: {self.id}): {observation}")
        logging.debug(f"Consider vehicle: {self.consider_vehicle}")
        logging.debug(
            f"Busy value: {busy_val}, Close busy values: {close_busy_vals}")
        return observation

    def compute_reward(self):
        return self.battery_wait_time_reward()

    # Reward combining of battery and wait time weightings
    def battery_wait_time_reward(self):
        return self.not_charged_reward + self.charge_reward

    def get_closest_battery(self, dists_to_station) -> np.ndarray:

        # Only consider to reroute vehicles not already rerouted to a station and not already been decided by the agent whether to charge or not
        print('dists_to_station:', dists_to_station)
        consider_vehicles_dists_batteries = [[v,  dists_to_station[v], self.get_battery(
            v)] for v, d in dists_to_station.items() if d > 0 and d <= MAX_CLOSEST_DISTANCE]

        # Sort by closest vehicles
        consider_vehicles_dists_batteries.sort(key=lambda x: x[1])

        print('consider_vehicles_dists_batteries:',
              consider_vehicles_dists_batteries)

        if len(consider_vehicles_dists_batteries) == 0:
            print(f"No vehicles to consider, (station: {self.id})")
            self.consider_vehicle = None
            return None

        # get first (closest)
        consider_vehicle_dist_battery = consider_vehicles_dists_batteries[0]
        # Use for getting the corresponding ids of the observation vehicles
        self.consider_vehicle = consider_vehicle_dist_battery[0]
        print('consider_vehicle:', self.consider_vehicle)
        # don't get removed so looping edge case not covered
        self.decided_vehicles.add(self.consider_vehicle)
        print('decided_vehicles_now: ', self.decided_vehicles)

        battery = self.get_battery(self.consider_vehicle)
        return battery

    def remove_rerouted_vehicles(self):
        # get all vehicles currently on lane and if in rerouted_vehicles then remove it
        # remove rerouted vehicles which are now on the lane of the charger
        vehicles_on_lane = self.sumo.lane.getLastStepVehicleIDs(self.lane_id)
        self.rerouted_vehicles = set([
            v for v in self.rerouted_vehicles if v not in vehicles_on_lane])

    def get_lane_wait_time(self):
        # Simplifying assumption: Each station is at end of a lane and that current wait_time will impact future??
        lanes = self.incoming_lanes_ids.copy()
        lanes.add(self.lane_id)
        wait_time = sum([self.sumo.lane.getWaitingTime(lane)
                        for lane in lanes])

        # (in seconds)
        # normalize the value to between 0-estimated_max
        wait_time = wait_time / ESTIMATED_MAX_WAIT_TIME

        return wait_time

    def get_lane_density(self):
        # 1 -> most dense (lane full of vehicles), 0 -> not dense (lane empty)
        # ? add the number vehicles routed to the station - not perfect but seems useful?
        lanes = self.incoming_lanes_ids.copy()
        lanes.add(self.lane_id)
        lane_density_sum = 0
        rerouted_av = (len(self.rerouted_vehicles)/len(lanes))
        MIN_GAP = 2.5
        for lane in lanes:
            lane_length = self.sumo.lane.getLength(lane)
            lane_density = (self.sumo.lane.getLastStepVehicleNumber(lane) + rerouted_av) / (
                lane_length / (MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            lane_density_sum += lane_density

        # the average density of the station lane and any lanes directly to that lane
        total_lane_density = lane_density_sum / len(lanes)

        if total_lane_density > 1:  # clipping
            total_lane_density = 1

        # TODO test
        return total_lane_density

    # TODO test
    def get_close_busy_vals(self):
        print('self.closest_stations:', self.closest_stations)
        if not self.closest_stations:  # ! only should occur at start... check! or none after
            busy_vals = list(np.ones(CLOSEST_STATIONS_NUM))
            print('error if repeating!')
        else:
            busy_vals = [cs.busy_val if cs.busy_val <
                         1 else 1 for cs in self.closest_stations.values()]  # plus clipping!
        # ? if not enough close stations then add dummy very busy stations?
        while len(busy_vals) < CLOSEST_STATIONS_NUM:
            busy_vals.append(1)

        return busy_vals

    def get_busy_val(self):
        wait_time = self.get_lane_wait_time()
        density = self.get_lane_density()

        # TODO adjust this weighting?
        self.busy_val = max(wait_time, density)
        print('busy_val:', self.busy_val)
        if self.busy_val > 1:  # clipping
            self.busy_val = 1
        return self.busy_val

    # reward for charging
    def get_combined_charge_reward(self, battery):
        # todo: tune
        max_bat_rew = 2
        # weighted, range[-max_rew, max_rew]
        battery_reward = - \
            max_bat_rew if battery > 0.4 else np.cos(battery*7.85)*max_bat_rew

        energy_consumed = float(self.sumo.vehicle.getParameter(
            self.consider_vehicle, "device.battery.totalEnergyConsumed"))
        distance_travelled = self.sumo.vehicle.getDistance(
            self.consider_vehicle)

        # get remaining range of vehicle
        if distance_travelled == 0 or energy_consumed == 0:
            remaining_range = MAX_RANGE
        else:
            mWh = distance_travelled / energy_consumed  # ???
            remaining_range = float(self.sumo.vehicle.getParameter(
                self.consider_vehicle, "device.battery.actualBatteryCapacity")) * mWh

        # TODO test
        close_busy_vals_diff = self.get_close_busy_vals_diff(remaining_range)

        # todo tune
        # ranges [-max_collab_rew,max_collab_rew] (should be bigger than the other combined reward...?)
        max_collab_rew = 1  # ?should be more than battery...? to make sure if charge low but could've charge low somewhere else then give pen? <- check
        # ? update? reward is min busyness diff, so if a close station is less busy and in range then penalize, if none less busy then reward - by the min-diff
        collaborative_reward = (min(close_busy_vals_diff)
                                if close_busy_vals_diff else 0) * max_collab_rew

        # If not needing a refill don't include how busy stations are into reward
        # ? 0.2? push this up?
        if battery > 0.2 or collaborative_reward == 0:  # or battery < 0.05:#?
            # collab reward and busy reward not relevant if a) doesn't need charging,( b) needs charging urgently -> if it can make it to next it should..?)
            return battery_reward

        print(
            f"battery_reward: {battery_reward}, collab_reward: {collaborative_reward}")

        # ?????? trying to make sure it's not small, collab reward range: [1->2, -1->-2]
        if collaborative_reward > 0:
            collaborative_reward += 1
        else:
            collaborative_reward -= 1

        # ? collab_rew includes battery by range...
        return collaborative_reward

    def get_close_busy_vals_diff(self, remaining_range):
        close_busy_vals_diff = []

        veh_to_station_distances = {}
        last_curr_edge_id = self.sumo.vehicle.getRoadID(self.consider_vehicle)
        last_curr_lane_id = self.sumo.vehicle.getLaneID(self.consider_vehicle)
        lane_length = self.sumo.lane.getLength(last_curr_lane_id)

        for cs_id, cs in self.env.charging_stations.items():
            # TODO check if in same units! km?
            if cs_id != self.id:
                cs2_edge = self.env.cs_edges[cs_id]
                veh_to_station_distance = self.sumo.simulation.getDistanceRoad(
                    last_curr_edge_id, 0, cs2_edge, 0, isDriving=True)

                veh_to_station_distances[cs_id] = veh_to_station_distance

        sorted_station_distances = {id: dist for id, dist in sorted(
            veh_to_station_distances.items(), key=lambda item: item[1]) if dist > 0}

        print('sorted_station_distances:', sorted_station_distances)
        closest_ids = list(sorted_station_distances.keys())[
            :CLOSEST_STATIONS_NUM]
        print('closest_ids:', closest_ids)

        # # ? get charging_stations another way?
        self.closest_stations = {
            cs_id: self.env.charging_stations[cs_id] for cs_id in closest_ids}

        print(
            f'self.closest_stations to (cd_{self.id}):', self.closest_stations.keys())

        for cs in self.closest_stations.values():
            if remaining_range > veh_to_station_distance:
                print(f'veh can get to {cs}')
                close_busy_vals_diff.append(cs.busy_val - self.busy_val)

        #! get only the top 3 closest!)

        print(f"TEST!!: close_busy_vals_diff: {close_busy_vals_diff}")
        return close_busy_vals_diff

    # Get distance from vehicle to charging station

    def get_close_evs(self):
        results = self.sumo.chargingstation.getContextSubscriptionResults(
            self.id)
        # print('results:', results)
        vehicles_ids = results.keys()
        close_evs = [v for v in vehicles_ids if v in self.sumo.vehicle.getIDList() and self.sumo.vehicle.getParameter(
            v, "has.battery.device") == "true"]

        # print(f'(cs: {self.id}) subscription results:', results)
        print(f'station {self.id} -> close evs: {close_evs}')
        return close_evs

    def get_dist_to_station(self, vehicle):
        vehicle_edge = self.sumo.vehicle.getRoadID(
            vehicle)
        # vehicle_pos = self.sumo.vehicle.getLanePosition(
        #     vehicle)    # ? test - is this the correct for pos1

        dist_to_station = self.sumo.simulation.getDistanceRoad(
            vehicle_edge, 0, self.edge_id, self.cs_end_pos, isDriving=True)

        return float(dist_to_station)

    def get_battery(self, vehicle):
        battery = float(self.sumo.vehicle.getParameter(
            vehicle, 'device.battery.actualBatteryCapacity'))
        return battery

    def get_battery(self, vehicle):
        battery = self.get_battery(vehicle)
        max_battery = float(self.sumo.vehicle.getParameter(
            vehicle, 'device.battery.maximumBatteryCapacity'))
        return (battery/max_battery)

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting."""
        return self.sumo.lane.getLastStepHaltingNumber(self.lane_id)

    def _get_veh_list(self):
        veh_list = self.sumo.lane.getLastStepVehicleIDs(self.lane_id)
        return veh_list
