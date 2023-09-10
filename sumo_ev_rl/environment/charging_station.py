import os
import sys
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


MAX_CLOSEST_DISTANCE = 300
MAX_RANGE = 150000
CHARGING_DURATION = 80

# Wait time given as a fraction of the estimated max wait time, currently set to 1hr
ESTIMATED_MAX_WAIT_TIME = 1*60*60
CLOSEST_STATIONS_NUM = 2


class ChargingStation:
    logging.basicConfig(level=logging.DEBUG)

    def __init__(self, env, cs_id, traci_connection):
        self.id = cs_id
        self.env = env
        self.traci_connection = traci_connection

        # penalties/reward counts
        self.charge_reward = 0
        self.not_charged_reward = 0

        # Get station properties
        self.consider_vehicle = None
        self.decided_vehicles = set()
        self.rerouted_vehicles = set()
        self.waiting_count = 0
        self.busy_val = 0
        self.closest_station_ids_in_range = []
        self.consider_vehicle_remaining_range = MAX_RANGE
        self.closest_stations_to_considered_vehicle = None
        self.lane_id = self.traci_connection.chargingstation.getLaneID(self.id)
        self.lane_length = self.traci_connection.lane.getLength(self.lane_id)
        self.edge_id = self.traci_connection.lane.getEdgeID(self.lane_id)
        self.cs_end_pos = self.traci_connection.chargingstation.getEndPos(self.id)
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

    # Get the closest stations in driving distance from current
    def get_close_stations(self):
        dists_to_stations = {}
        for cs_id in self.env.charging_stations.keys():
            if cs_id != self.id:
                cs2_edge = self.env.cs_edges[cs_id]
                dist_to_station = self.traci_connection.simulation.getDistanceRoad(
                    self.edge_id, 0, cs2_edge, 0)

                if dist_to_station > 0:
                    dists_to_stations[cs_id] = dist_to_station

        self.closest_station_ids = sorted(dists_to_stations, key=dists_to_stations.get)[
            :CLOSEST_STATIONS_NUM]

    def compute_observation(self):
        charging_station_edges = self.env.cs_edges.values()
        # Get driving distance of EVs in close proximity not already rerouted or decided upon
        dists_to_station = {
            v: self.get_dist_to_station(v) for v in self.get_close_evs()
            if
            not
            set(self.traci_connection.vehicle.getRoute(v)).intersection(
                set(charging_station_edges)) and v not in self.decided_vehicles}
        logging.debug(
            f"Distances to station of close evs, (station: {self.id}): {dists_to_station}")

        closest_vehicle_battery = self.get_closest_battery(dists_to_station)

        # If no close vehicle detected return observation of zeros
        if closest_vehicle_battery == None:
            return np.zeros(2 + CLOSEST_STATIONS_NUM, dtype=np.float32)

        # Get busy values of this stations and close stations
        busy_val = self.get_busy_val()
        close_busy_vals = self.get_close_busy_vals()

        logging.debug(f"close_busy_vals: {close_busy_vals}")
        logging.debug(f"busy_val: {busy_val}")
        logging.debug(f"close_busy_vals: {close_busy_vals}")

        observation = np.array(
            [closest_vehicle_battery, busy_val] + close_busy_vals, dtype=np.float32)
        logging.debug(f"Observation (cs: {self.id}): {observation}")
        logging.debug(
            f"Busy value: {busy_val}, Close busy values: {close_busy_vals}")
        return observation

    def reset_rewards(self):
        self.charge_reward = 0
        self.not_charged_reward = 0

    # Update vehicle's route to charge at the station
    def reroute_vehicle(self, vehicle_id: int):
        battery = self.get_battery(vehicle_id)
        # Receive the charging reward
        self.charge_reward += self.get_combined_charge_reward(
            battery)
        logging.debug(
            f"Charge reward (station: {self.id}): {self.charge_reward}")

        # Check if already rerouted
        last_curr_edge_id = self.traci_connection.vehicle.getRoadID(vehicle_id)
        vehicle_route = self.traci_connection.vehicle.getRoute(vehicle_id)
        charging_station_edges = self.env.cs_edges.values()

        # Don't reroute if another agent has rerouted
        if set(vehicle_route).intersection(set(charging_station_edges)):
            logging.debug(f"Vehicle already rerouted in this step")
            return

        # Reroute to charging station
        destination_edge_id = vehicle_route[-1]
        route_to_station = self.traci_connection.simulation.findRoute(
            last_curr_edge_id, self.edge_id).edges
        route_to_destination = self.traci_connection.simulation.findRoute(
            self.edge_id, destination_edge_id).edges
        self.rerouted_vehicles.add(vehicle_id)
        new_route = route_to_station[:-1] + route_to_destination
        logging.debug(f'New route: (vehicle: {vehicle_id}): {new_route}')

        self.traci_connection.vehicle.setRoute(vehicle_id, new_route)
        self.traci_connection.vehicle.setStop(
            vehicle_id, self.edge_id, pos=self.cs_end_pos,  duration=CHARGING_DURATION)

    def compute_reward(self):
        return self.battery_wait_time_reward()

    # Reward combining of battery and wait time weightings
    def battery_wait_time_reward(self):
        logging.debug(
            f"Not charged reward: {self.not_charged_reward}, Charged reward: {self.charge_reward}")
        reward = self.not_charged_reward + self.charge_reward
        logging.debug(
            f"Combined reward (cs: {self.id}): {reward}")
        return reward

    # Returns the battery of the closest vehicle to the station in driving distance
    def get_closest_battery(self, dists_to_station) -> np.ndarray:
        # Only considers vehicles not already rerouted to a station and not already been decided by the agent
        logging.debug(f"Distances to station: {dists_to_station}")
        consider_vehicles_dists_batteries = [[v,  dists_to_station[v], self.get_battery(
            v)] for v, d in dists_to_station.items() if d > 0 and d <= MAX_CLOSEST_DISTANCE]

        # Sort by closest vehicles
        consider_vehicles_dists_batteries.sort(key=lambda x: x[1])
        logging.debug(
            f"consider_vehicles_dists_batteries: {consider_vehicles_dists_batteries}")

        if len(consider_vehicles_dists_batteries) == 0:
            logging.debug(f"No vehicles to consider (station: {self.id})")
            self.consider_vehicle = None
            return None

        # Consider vehicle is the closest vehicle that the station will make an action on
        consider_vehicle_dist_battery = consider_vehicles_dists_batteries[0]
        self.consider_vehicle = consider_vehicle_dist_battery[0]
        logging.debug(
            f"Consider vehicle (station: {self.id}): {self.consider_vehicle}")

        self.decided_vehicles.add(self.consider_vehicle)
        logging.debug(
            f"Decided vehicles updated (station: {self.id}): {self.decided_vehicles}")

        battery = consider_vehicle_dist_battery[2]
        return battery

    # Remove the rerouted vehicles from the set once it's about to charge
    def remove_rerouted_vehicles(self):
        vehicles_on_lane = self.traci_connection.lane.getLastStepVehicleIDs(self.lane_id)
        self.rerouted_vehicles = set([
            v for v in self.rerouted_vehicles if v not in vehicles_on_lane])

    def get_lane_wait_time(self):
        lanes = self.incoming_lanes_ids.copy()
        lanes.add(self.lane_id)
        wait_time = sum([self.traci_connection.lane.getWaitingTime(lane)
                        for lane in lanes])

        # Normalize the value to between 0-estimated_max
        wait_time = wait_time / ESTIMATED_MAX_WAIT_TIME
        return wait_time

    # Calculates lane density using current and incoming lanes to station
    def get_lane_density(self):
        lanes = self.incoming_lanes_ids.copy()
        lanes.add(self.lane_id)
        lane_density_sum = 0
        # Combines the rerouted vehicles arriving to improve density estimations
        rerouted_av = (len(self.rerouted_vehicles)/len(lanes))
        MIN_GAP = 2.5
        for lane in lanes:
            lane_length = self.traci_connection.lane.getLength(lane)
            lane_density = (self.traci_connection.lane.getLastStepVehicleNumber(lane) + rerouted_av) / (
                lane_length / (MIN_GAP + self.traci_connection.lane.getLastStepLength(lane)))
            lane_density_sum += lane_density

        # Average density of the station lane and any lanes directly to that lane
        total_lane_density = lane_density_sum / len(lanes)

        # Clipping to maximum if other the threshold
        total_lane_density = 1 if total_lane_density > 1 else total_lane_density
        return total_lane_density

    # Get the busy values of close stations
    def get_close_busy_vals(self):
        closest_station_ids_in_range = []
        veh_to_station_distances = {}
        last_curr_edge_id = self.traci_connection.vehicle.getRoadID(self.consider_vehicle)

        logging.debug(
            f"Closest station ids to cs:{self.id} (not specifically in range): {self.closest_station_ids}")
        # If no close stations display equivalent as all full in observation
        if not self.closest_station_ids:
            return list(np.ones(CLOSEST_STATIONS_NUM))

        self.consider_vehicle_remaining_range = self.get_remaining_range(
            self.consider_vehicle)

        # Get only the close stations that are in range and accessible by the considered vehicle
        for cs_id in self.closest_station_ids:
            cs2_edge = self.env.cs_edges[cs_id]
            veh_to_station_distance = self.traci_connection.simulation.getDistanceRoad(
                last_curr_edge_id, 0, cs2_edge, 0, isDriving=True)

            if veh_to_station_distance > 0 and self.consider_vehicle_remaining_range > veh_to_station_distance:
                closest_station_ids_in_range.append(cs_id)

            veh_to_station_distances[cs_id] = veh_to_station_distance

        self.closest_station_ids_in_range = closest_station_ids_in_range
        logging.debug(
            f"Closest station ids to cs:{self.id} (in range of vehicle: {self.consider_vehicle}): {self.closest_station_ids_in_range}")

        logging.debug(
            f"Closest stations: {self.closest_stations_to_considered_vehicle}")

        # Get the busy values of the close stations that are in range and accessible to the vehicle
        busy_vals = []
        if self.closest_station_ids_in_range:
            busy_vals = [
                self.env.charging_stations[cs_id].busy_val
                for cs_id in self.closest_station_ids_in_range]
        else:
            busy_vals = list(np.ones(CLOSEST_STATIONS_NUM))
        # Fill remaining values if not filled by stations
        while len(busy_vals) < CLOSEST_STATIONS_NUM:
            busy_vals.append(1)

        return busy_vals

    # Gets the max of the wait time and density of station
    def get_busy_val(self):
        wait_time = self.get_lane_wait_time()
        density = self.get_lane_density()
        self.busy_val = max(wait_time, density)

        logging.debug(f"Busy value: {self.busy_val}")
        # Clip if over threshold
        self.busy_val = 1 if self.busy_val > 1 else self.busy_val
        return self.busy_val

    # Calculate the estimated remaining range using the previous vehicle stats
    def get_remaining_range(self, vehicle):
        # Get remaining range of considered vehicle
        energy_consumed = float(self.traci_connection.vehicle.getParameter(
            vehicle, "device.battery.totalEnergyConsumed"))
        distance_travelled = self.traci_connection.vehicle.getDistance(vehicle)
        if distance_travelled == 0 or energy_consumed == 0:
            remaining_range = MAX_RANGE
        else:
            mWh = distance_travelled / energy_consumed
            remaining_range = float(self.traci_connection.vehicle.getParameter(
                vehicle, "device.battery.actualBatteryCapacity")) * mWh

        return remaining_range

    # Returns reward for charging
    def get_combined_charge_reward(self, battery):
        # Weighted battery reward: range[-max_rew, max_rew]
        max_bat_rew = 2
        battery_reward = - \
            max_bat_rew if battery > 0.4 else np.cos(battery*7.85)*max_bat_rew

        # Get busyness differences of closest stations to current that are within the remaining range
        close_busy_vals_diff = self.get_close_busy_vals_diff()

        # Weighted minimum busyness difference, if a close station is less busy min(diff) will be negative and vice versa
        max_collab_rew = 1
        collaborative_reward = (min(close_busy_vals_diff)
                                if close_busy_vals_diff else 0) * max_collab_rew

        # If vehicle doesn't need chargining then don't combine how busy stations are into the reward
        if battery > 0.2 or collaborative_reward == 0:
            return battery_reward

        # If collaborative reward exists then increase the significance
        if collaborative_reward > 0:
            collaborative_reward += 1
        else:
            collaborative_reward -= 1

        return collaborative_reward

    # Get busy values of close stations within remaining range of considered vehicle
    def get_close_busy_vals_diff(self):
        close_busy_vals_diff = []

        logging.debug(
            f'Closest station ids in range of considered vehicle: {self.closest_station_ids_in_range}')

        for cs_id in self.closest_station_ids_in_range:
            logging.debug(f'Vehicle is in range of: {cs_id}')
            close_busy_vals_diff.append(
                self.env.charging_stations[cs_id].busy_val - self.busy_val)

        logging.debug(f'Close busy values difference: {close_busy_vals_diff}')
        return close_busy_vals_diff

    # Get subscription distance from vehicle to charging station
    def get_close_evs(self):
        results = self.traci_connection.chargingstation.getContextSubscriptionResults(
            self.id)
        vehicles_ids = results.keys()
        close_evs = [v for v in vehicles_ids if v in self.traci_connection.vehicle.getIDList(
        ) and self.traci_connection.vehicle.getParameter(v, "has.battery.device") == "true"]

        logging.debug(f"Station ({self.id}) subscription results: {results}")
        logging.debug(f"Close EVs: {close_evs}")
        return close_evs

    # Returns the driving distance of a vehicle to this station
    def get_dist_to_station(self, vehicle):
        vehicle_edge = self.traci_connection.vehicle.getRoadID(
            vehicle)

        dist_to_station = self.traci_connection.simulation.getDistanceRoad(
            vehicle_edge, 0, self.edge_id, self.cs_end_pos, isDriving=True)

        return float(dist_to_station)

    def get_battery_capacity(self, vehicle):
        battery = float(self.traci_connection.vehicle.getParameter(
            vehicle, 'device.battery.actualBatteryCapacity'))
        return battery

    def get_battery(self, vehicle):
        battery = self.get_battery_capacity(vehicle)
        max_battery = float(self.traci_connection.vehicle.getParameter(
            vehicle, 'device.battery.maximumBatteryCapacity'))
        return (battery/max_battery)
