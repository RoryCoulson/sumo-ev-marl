# sumo-ev-marl

Electric vehicle charging point demand handling using multi-agent reinforcement learning. The project aims to improve the problem of queueing at EV charging points by providing smarter distribution. The work includes a cooperative custom environment and the agents are deployed using PPO, DQN and greedy algorithms within a SUMO simulator. Although the simulation won't fully match reality, the findings provide an insight into the potential capability of such a system.



Run `pip install .` in the root to build the package allowing the experiments to access the sumo_ev_rl_competitive module.
Set `enable_gui` to True when running an experiment to visualize the training.

A correct installation of SUMO is required to run these simulations.

Acknowledgements:

The file: sumo_ev_rl_competitive/environment/env.py in this project is heavily reliant on Gym (https://www.gymlibrary.dev/content/environment_creation/) and PettingZoo (https://pettingzoo.farama.org/content/environment_creation/) documentation and an implementation with traffic lights, sumo-rl (https://github.com/LucasAlegre/sumo-rl) for setting up a custom multi-agent environment with the correct structure and boilerplate needed.
