"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces
import traci


from .charging_station import CLOSEST_STATIONS_NUM, MAX_CLOSEST_DISTANCE, ChargingStation


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, cs: ChargingStation):
        """Initialize observation function."""
        self.cs = cs

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, cs: ChargingStation):
        """Initialize default observation function."""
        super().__init__(cs)

    def __call__(self) -> np.ndarray:
        # Get all vehicles currently in the simulator
        # self.cs.evs = self.cs.env.get_evs()
        charging_station_edges = self.cs.env.cs_edges.values()

        dists_to_station = {v: self.cs.get_dist_to_station(v) for v in self.cs.get_close_evs() if not set(
            self.cs.sumo.vehicle.getRoute(v)).intersection(set(charging_station_edges)) and v not in self.cs.decided_vehicles}

        print(
            f"dists_to_station of close evs, (cs: {self.cs.id}): {dists_to_station}")

        close_vehicle_battery = self.cs.get_closest_battery(dists_to_station)

        # close_vehicle_battery = -1

        if close_vehicle_battery == -1:
            # ?(wait_time or density not necessary if no vehicles.. set to zero ok?)
            return np.zeros(2 + CLOSEST_STATIONS_NUM, dtype=np.float32)

        busy_val = self.cs.get_busy_val()
        # the busy values of nearby stations
        close_busy_vals = self.cs.get_close_busy_vals()
        # TODO test
        print('rerouted_vehicles:', self.cs.rerouted_vehicles)

        observation = np.array(
            [close_vehicle_battery, busy_val] + close_busy_vals, dtype=np.float32)
        print(
            f"OBSERVATION (cs: {self.cs.id}): {observation}, consider_vehicle: {self.cs.consider_vehicle}, busy_val: {busy_val}, close_busy_vals: {close_busy_vals}")
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""

        # one-hot encoded.
        return spaces.Box(
            low=np.zeros(2 + CLOSEST_STATIONS_NUM, dtype=np.float32),
            high=np.ones(2 + CLOSEST_STATIONS_NUM, dtype=np.float32),
        )
