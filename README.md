# sumo-ev-marl

Electric vehicle charging station demand handling using multi-agent reinforcement learning. The project aims to improve the queueing problem at EV charging stations by providing smarter distribution. The work includes a cooperative custom environment and agents that are deployed using PPO, DQN and greedy algorithms within a SUMO simulator. The agents' aim is to learn which EVs are best to charge and which agent is best suited to charge them. Although the simulation won't fully match reality, the findings provide an insight into the potential capability of such a system.


## Setup
* Python 3.9
* Install SUMO (https://sumo.dlr.de/docs/Installing/index.html)
* `pip install -r $SUMO_HOME/tools/requirements.txt`
* `pip install .` in the root to build the package
* See requirements.txt for additional libraries


## Networks

Built three test networks to experiment with different scenarios:

<img width="335" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/1dc4d03c-3865-4fc3-a477-3bf180e20fc5">
<img width="266" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/9b15f68e-8a4e-47be-90ac-75e86cd45747">
<img width="217" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/f4af4e34-4e14-4d06-a87d-8ef45ead95e9"><br/><br/>

Created a final example network using OpenStreetMap which is a part of Berlin's road network:

<img width="265" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/5d6ed44c-b9a6-45cc-abda-4af8c05e5e40">
<img width="186"  src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/c88aef51-a21b-4096-890a-e0894636be86">
<img width="360" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/23dd856d-6eb5-4eda-99a5-574049f39759"><br/><br/>

## Berlin results:
An 87.2% and 88.1% reduction in mean wait times for the multi-agent PPO and DQN implementations respectively were achieved against a greedy approach where each EV goes to the nearest station at a low charge.

<img width="1000" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/bcc35ec4-84e6-4b74-800e-0c2efae3f3bb">

<img width="263" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/45d05bb8-761a-49b3-9dd9-880649192e8f">
<img width="287" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/6a9aca83-49e9-4d74-ba26-3d77a3591855">
<img width="269" src="https://github.com/RoryCoulson/sumo-ev-marl/assets/52762734/b5a1f358-8c29-4e88-ba5a-3cf6f11e62d8">


## References:
* Gymnasium: https://gymnasium.farama.org/
* PettingZoo: https://pettingzoo.farama.org/index.html
* Ray RLlib: https://docs.ray.io/en/latest/rllib/index.html
* Sumo-rl: https://github.com/LucasAlegre/sumo-rl

