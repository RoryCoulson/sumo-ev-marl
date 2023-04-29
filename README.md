# sumo-ev-marl

This repository includes a cooperative custom environment using both Gym and PettingZoo libraries. RLlib is also used in developing the PPO, DQN, QL algorithms for running in the SUMO simulator. This extends upon the work from the sumo-ev-marl-competitive repository with an improved environment configuration and more simulation networks.

Run pip install . in the root to build the package allowing the experiments to access the sumo_ev_rl_competitive module.

A correct installation of SUMO is required to run these simulations. Run: export LIBSUMO_AS_TRACI=1 for performance increase with SUMO.

Acknowledgements:

The file: sumo_ev_rl_competitive/environment/env.py in this project is heavily reliant on Gym (https://www.gymlibrary.dev/content/environment_creation/) and PettingZoo (https://pettingzoo.farama.org/content/environment_creation/) documentation and an implementation with traffic lights, sumo-rl (https://github.com/LucasAlegre/sumo-rl) for setting up a custom multi-agent environment with the correct structure and boilerplate needed.
