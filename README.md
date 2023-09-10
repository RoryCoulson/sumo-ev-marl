# sumo-ev-marl

Electric vehicle charging station demand handling using multi-agent reinforcement learning. The project aims to improve the queueing problem at EV charging stations by providing smarter distribution. The work includes a cooperative custom environment and agents that are deployed using PPO, DQN and greedy algorithms within a SUMO simulator. The agents' aim is to learn which EVs are best to charge and which agent is best situated to charge them. Although the simulation won't fully match reality, the findings provide an insight into the potential capability of such a system.


## Setup
* Python 3.9
* Install SUMO (https://sumo.dlr.de/docs/Installing/index.html)
* `pip install -r $SUMO_HOME/tools/requirements.txt`
* `pip install .` in the root to build the package.
* See requirements.txt for additional libraries


## Networks

Built three test networks to experiment with different scenarios:

<img width="407" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/9ffccb4e-0f33-436b-ab18-e696522508cd">
<img width="326" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/38261253-aa8c-4018-8910-8455f3511446">
<img width="269" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/f8de548d-748b-4d9f-839d-ce4306cffe74"><br/><br/>

Created a final example network using OpenStreetMap which is a part of Berlin's road network:

<img width="308" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/c4e108a3-392f-4e6c-b5df-40ad43d21b80">
<img width="210"  src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/dbfb190d-b5e9-47d9-806b-b63e75bd7dea">
<img width="425" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/b88c7e95-f6f0-483d-a433-079d0c5a6e27"><br/><br/>

## Berlin results:
A 87.2% and 88.1% reduction in mean wait times for the multi-agent PPO and DQN implementations were achieved against a greedy approach where each EV goes to the nearest station at a certain battery level.

<img width="1000" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/0d54f9c4-e186-4aff-9fe3-e98da70fe1c8">
<img width="302" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/39618d3f-ae66-4961-9357-5d5562cfb0a6">
<img width="330" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/50824258-80c8-46fe-bfee-28e174d79425">
<img width="300" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/5f060f07-09f5-456a-808c-a1efb0b5438c">


## References:
* Gymnasium: https://gymnasium.farama.org/
* PettingZoo: https://pettingzoo.farama.org/index.html
* Ray RLlib: https://docs.ray.io/en/latest/rllib/index.html
* Sumo-rl: https://github.com/LucasAlegre/sumo-rl


