# sumo-ev-marl

Electric vehicle charging station demand handling using multi-agent reinforcement learning. The project aims to improve the queueing problem at EV charging stations by providing smarter distribution. The work includes a cooperative custom environment and agents that are deployed using PPO, DQN and greedy algorithms within a SUMO simulator. The agents' aim is to learn which EVs are best to charge and which agent is best suited to charge them. Although the simulation won't fully match reality, the findings provide an insight into the potential capability of such a system.

Paper presented at EVS37: https://eprints.soton.ac.uk/487939/

## Setup
* Python 3.9
* Install SUMO (https://sumo.dlr.de/docs/Installing/index.html)
* `pip install -r $SUMO_HOME/tools/requirements.txt`
* `pip install .` in the root to build the package
* See requirements.txt for additional libraries


## Networks

Built three test networks to experiment with different scenarios:

<img width="335" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/cb5690d8-8933-4ae7-8ed2-489dcebb45ee">
<img width="266" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/dda8cb9b-d89d-474c-9630-b47de0b2cf78">
<img width="217" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/06a30581-0cc6-4d49-ab4a-00eb82f5128c"><br/><br/>

Created a final example network using OpenStreetMap which is a part of Berlin's road network:

<img width="265" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/942dcdcc-9a54-4125-aa7b-c78258a3215f">
<img width="186"  src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/45a1b1ea-aa96-450d-9f27-6d979b2a0457">
<img width="360" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/30a5dbb9-4c40-442f-aa87-c94354427379"><br/><br/>

## Berlin results:
An 87.2% and 88.1% reduction in mean wait times for the multi-agent PPO and DQN implementations respectively were achieved against a greedy approach where each EV goes to the nearest station at a low charge.

<img width="1000" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/6abebfa3-0fac-4f4e-bcb5-a23176a5b14f">

<img width="263" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/fcd8314c-2ef0-498e-975f-c9c1896f5ee4">
<img width="287" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/2e7a5669-6804-46e8-a9db-e8f8eb821f9a">
<img width="269" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/27367618-f4bd-4b77-bc20-f83ed9f6fab7">


## References:
* Gymnasium: https://gymnasium.farama.org/
* PettingZoo: https://pettingzoo.farama.org/index.html
* Ray RLlib: https://docs.ray.io/en/latest/rllib/index.html
* Sumo-rl: https://github.com/LucasAlegre/sumo-rl

