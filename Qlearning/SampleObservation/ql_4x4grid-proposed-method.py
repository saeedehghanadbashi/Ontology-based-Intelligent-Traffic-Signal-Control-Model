import argparse
import os
import sys
import pandas as pd

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
from collections import defaultdict
import csv
import random
from datetime import datetime
import numpy as np

class Dict(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, Dict)
    def __repr__(self):
        return dict.__repr__(self)
#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from Qlearning.env import SumoEnvironment, prettyfloat
from sumo_rl.agents.ql_agent import QLAgent
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

if __name__ == '__main__':

    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1
    
    out_csv = 'Qlearning/SampleObservation/Sample-Observation-Qlearning-output-OUR-ALGORITHM-run1.csv'.format(alpha, gamma, decay)

    env = SumoEnvironment(net_file='nets/4x4-Saeedeh/4x4.net.xml',
                          route_file='nets/4x4-Saeedeh/4x4c1c2c1c2.rou.xml',
                          use_gui=False,
                          num_seconds=1000,
                          time_to_load_vehicles=300,
                          max_depart_delay=0,
                          tripinfo_output='Qlearning/SampleObservation/Sample-Observation-Qlearning-tripinfo-OUR-ALGORITHM-run1.xml',
                          phases=[
                            traci.trafficlight.Phase(35, "GGGrrr"),   # north-south
                            traci.trafficlight.Phase(2, "yyyrrr"),
                            traci.trafficlight.Phase(35, "rrrGGG"),   # west-east
                            traci.trafficlight.Phase(2, "rrryyy")
                            ])

    for run in range(1, runs+4):
        initial_states = env.reset()

        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts]),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)) for ts in env.ts_ids}
        infos = []
        done = {'__all__': False}
        waiting_time_data = []

        while not done['__all__']:

            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            neighbors_list, time_on_phase_list, phase_id_list, step_num, veh_position_data, veh_waiting_time, veh_complete_data, s, r, done, info = env.step(action=actions)
            infos.append(info)

            now = datetime.now()
            minute_time = now.minute

            print("step_num:", step_num)

            if step_num == 1: 
                waiting_time_entry = "step_num,p_num,vehicle_name,type_name,position,lane_num,waiting_time".split(",")
                waiting_time_data.append(waiting_time_entry)

            for aa in sorted(veh_complete_data.keys()):

                sampling_size = 2
                sampling_waiting_time = 0
                ee_pre = ''

                for bb in sorted(veh_complete_data[aa].keys()):
                    for cc in sorted(veh_complete_data[aa][bb].keys()):
                        for dd in sorted(veh_complete_data[aa][bb][cc].keys()):
                            for ee in sorted(veh_complete_data[aa][bb][cc][dd].keys()):
                                #print (step_num,",", aa,",",bb,",",cc,",",dd,",",ee,",",prettyfloat(veh_complete_data[aa][bb][cc][dd][ee]))

                                if minute_time % 2 == 0:

                                    #'''Sampling using weights
                                    if veh_complete_data[aa][bb][cc][dd][ee] != 0 and sampling_size > 0 and (ee_pre == '' or ee != ee_pre):
                                        sampling_waiting_time = veh_complete_data[aa][bb][cc][dd][ee]  
                                        sampling_size = sampling_size - 1
                                        ee_pre = ee
                                    #'''
                                    
                                    '''Sampling-baseline
                                    if sampling_size > 0 and random.randint(0, 1) == 1:
                                        sampling_waiting_time = veh_complete_data[aa][bb][cc][dd][ee]  
                                        sampling_size = sampling_size - 1
                                    '''

                                    r[aa] = r[aa] + veh_complete_data[aa][bb][cc][dd][ee] - sampling_waiting_time

                                waiting_time_entry = step_num,aa,bb,cc,dd,ee,prettyfloat(veh_complete_data[aa][bb][cc][dd][ee])
                                waiting_time_data.append(waiting_time_entry)

            for agent_id in ql_agents.keys():
                reward = r[agent_id]                   
                next_state = str(env.encode(s[agent_id]))

                ql_agents[agent_id].learn(next_state, reward)

        outfile = open('Qlearning/SampleObservation/Sample-Observation-Qlearning-result-OUR-ALGORITHM-run1.csv','w')
        writer=csv.writer(outfile)
        writer.writerows(waiting_time_data)
        outfile.close()

        #env.save_csv(out_csv, run)
        env.close()

        df = pd.DataFrame(infos)


