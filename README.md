An Ontology-based Intelligent Traffic Signal Control Model

Reinforcement Learning (RL) can enhance the adjustment of the traffic signals' phases to improve the traffic flow. RL methods use ontologies and reasoning to enrich the controllers' domain knowledge, enabling them to interpret the traffic data, and ultimately improving their performance. 

Various RL methods are proposed for signal controllers with assumptions such as operating in non-stochastic environments with a predictable traffic flow and observing the fine-grained information of all vehicles. Such methods have not examined the robustness of the trained RL controllers' action selection when deployed in dynamic environments with partial detection of vehicles. However, in the real world, not all vehicles can be detectable, and not all events can be predicted.

In this paper, we propose an Ontology-based Intelligent Traffic Signal Control (OITSC) model that augments the RL controllers' observation using an environment ontology model, which improves their action selection particularly in dynamic, partially observable environments with stochastic traffic flow. The decreased vehicles' waiting time in various traffic scenarios with partial detection of vehicles, noisy sensor data, and unexpected traffic events shows that the performance of the controllers is significantly improved in all tested RL algorithms (i.e., Q-learning, SARSA, and Deep Q-Network).

This repository provides the implementation of Ontology-based Intelligent Traffic Signal Control (OITSC) model.

References

Baseline Code

https://github.com/LucasAlegre/sumo-rl

## Cite
If you use this repository in your research, please cite:
```
@inproceedings{ghanadbashi2021ontology,
  title={An ontology-based intelligent traffic signal control model},
  author={Ghanadbashi, Saeedeh and Golpayegani, Fatemeh},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
  pages={2554--2561},
  year={2021},
  organization={IEEE}
}
```
