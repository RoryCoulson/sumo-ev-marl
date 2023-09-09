# sumo-ev-marl

Electric vehicle charging point demand handling using multi-agent reinforcement learning. The project aims to improve the problem of queueing at EV charging points by providing smarter distribution. The work includes a cooperative custom environment and the agents are deployed using PPO, DQN and greedy algorithms within a SUMO simulator. Although the simulation won't fully match reality, the findings provide an insight into the potential capability of such a system.


## Setup
Run `pip install .` in the root to build the package allowing the experiments to access the sumo_ev_rl_competitive module.
Set `enable_gui` to True when running an experiment to visualize the training.

A correct installation of SUMO is required to run these simulations.

Acknowledgements:

The file: sumo_ev_rl_competitive/environment/env.py in this project is heavily reliant on Gym (https://www.gymlibrary.dev/content/environment_creation/) and PettingZoo (https://pettingzoo.farama.org/content/environment_creation/) documentation and an implementation with traffic lights, sumo-rl (https://github.com/LucasAlegre/sumo-rl) for setting up a custom multi-agent environment with the correct structure and boilerplate needed.



<img width="407" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/9ffccb4e-0f33-436b-ab18-e696522508cd">
<img width="326" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/38261253-aa8c-4018-8910-8455f3511446">
<img width="269" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/f8de548d-748b-4d9f-839d-ce4306cffe74"><br/><br/>


<img width="328" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/c4e108a3-392f-4e6c-b5df-40ad43d21b80">
<img width="230"  src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/dbfb190d-b5e9-47d9-806b-b63e75bd7dea">
<img width="445" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/b88c7e95-f6f0-483d-a433-079d0c5a6e27"><br/><br/>



<img width="1000" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/0d54f9c4-e186-4aff-9fe3-e98da70fe1c8">
<img width="322" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/39618d3f-ae66-4961-9357-5d5562cfb0a6">
<img width="350" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/50824258-80c8-46fe-bfee-28e174d79425">
<img width="330" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/5f060f07-09f5-456a-808c-a1efb0b5438c">
