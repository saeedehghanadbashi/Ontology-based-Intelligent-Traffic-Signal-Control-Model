import gym

import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
from gym import spaces
import numpy as np
from DQN.env import SumoEnvironment
from sumo_rl.util.gen_route import write_route_file
import traci

from DQN.dqnp1 import DQN

env = SumoEnvironment(net_file='nets/4x4-Saeedeh/4x4.net.xml',
                        single_agent=False,
                        route_file='nets/4x4-Saeedeh/4x4c1c2c1c2.rou.xml',
                        out_csv_name='DQN/SampleObservation/Sample-Observation-DQN-output-OUR-ALGORITHM-run1.csv',
                        use_gui=False,
                        num_seconds=1000,
                        yellow_time=4,
                        min_green=5,
                        max_green=100,
                        max_depart_delay=0,
                        tripinfo_output='DQN/SampleObservation/Sample-Observation-DQN-tripinfo-OUR-ALGORITHM-run1.xml',
                        time_to_load_vehicles=300,
                        phases=[
                          traci.trafficlight.Phase(35, "GGGrrr"),   # north-south
                          traci.trafficlight.Phase(2, "yyyrrr"),
                          traci.trafficlight.Phase(35, "rrrGGG"),   # west-east
                          traci.trafficlight.Phase(2, "rrryyy")
                          ])

model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    buffer_size=50000,
    exploration_fraction=0.05,
    exploration_final_eps=0.005
)

model.learn(total_timesteps=500)
'''
DQ_agents = {ts: DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    buffer_size=50000,
    exploration_fraction=0.05,
    exploration_final_eps=0.02
) for ts in env.ts_ids}

for agent_id in DQ_agents.keys():
    DQ_agents[agent_id].learn(total_timesteps=100000)

'''
