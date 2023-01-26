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

#'''Selecting part-uncertainty  
from sumo_rl.environment.HistoryNamespace import HistoryNamespace
history_flag_pre = HistoryNamespace()
history_flag_pre.__setitem__('flag_pre','None')
history_step_num_pre = HistoryNamespace()
history_step_num_pre.__setitem__('step_num_pre','None')
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
    
    out_csv = 'Qlearning/ImportantObservation/Important-Observation-Qlearning-output-OUR-ALGORITHM-run1.csv'.format(alpha, gamma, decay)

    env = SumoEnvironment(net_file='nets/4x4-Saeedeh/4x4.net.xml',
                          route_file='nets/4x4-Saeedeh/4x4c1c2c1c2.rou.xml',
                          use_gui=False,
                          num_seconds=1000,
                          time_to_load_vehicles=300,
                          max_depart_delay=0,
                          tripinfo_output='Qlearning/ImportantObservation/Important-Observation-Qlearning-tripinfo-OUR-ALGORITHM-run1.xml',
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

            now = datetime.now()
            minute_time = now.minute

            #'''Selecting part-uncertainty            
            flag_pre = history_flag_pre.ns['flag_pre'][-1]
            step_num_pre = history_step_num_pre.ns['step_num_pre'][-1]

            if step_num_pre != 'None' and minute_time % 2 == 0:
                for agent_id in ql_agents.keys():
                    #actions[agent_id] = random.randint(0, 1)
                    if flag_pre != 'None': actions[agent_id] = flag_pre[agent_id]
                    else: actions[agent_id] = random.randint(0, 1)
            #'''Selecting part-uncertainty

            ambulance_lane_list, trailer_lane_list, fueltruck_lane_list, importance_weight_list, neighbors_list, time_on_phase_list, phase_id_list, step_num, veh_position_data, veh_waiting_time, veh_complete_data, s, r, done, info = env.step(action=actions)
            infos.append(info)

            print("step_num:", step_num)

            #'''Selecting part-uncertainty
            flag = {}
            for aa in sorted(importance_weight_list.keys()):
                cnt = 0
                for bb in sorted(importance_weight_list[aa].keys()):
                    if cnt == 0: first_lane_weight = importance_weight_list[aa][bb]
                    if cnt != 0: second_lane_weight = importance_weight_list[aa][bb]
                    cnt = cnt + 1
                    
                if aa == '0': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '1': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 1
                    else: flag[aa] = 0
                if aa == '10': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '11': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 1
                    else: flag[aa] = 0
                if aa == '12': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 1
                    else: flag[aa] = 0
                if aa == '13': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 1
                    else: flag[aa] = 0
                if aa == '14': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '15': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '2': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '3': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '4': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '5': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '6': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '7': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1
                if aa == '8': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 1
                    else: flag[aa] = 0
                if aa == '9': 
                    if first_lane_weight > second_lane_weight: flag[aa] = 0
                    else: flag[aa] = 1

            history_flag_pre.__setitem__('flag_pre',flag)
            history_step_num_pre.__setitem__('step_num_pre',step_num)
            #'''

            if step_num == 1: 
                waiting_time_entry = "step_num,p_num,vehicle_name,type_name,position,lane_num,waiting_time".split(",")
                waiting_time_data.append(waiting_time_entry)

            for aa in sorted(veh_complete_data.keys()):
                for bb in sorted(veh_complete_data[aa].keys()):
                    for cc in sorted(veh_complete_data[aa][bb].keys()):
                        for dd in sorted(veh_complete_data[aa][bb][cc].keys()):
                            for ee in sorted(veh_complete_data[aa][bb][cc][dd].keys()):
                                #print (step_num,",", aa,",",bb,",",cc,",",dd,",",ee,",",prettyfloat(veh_complete_data[aa][bb][cc][dd][ee]))
                                waiting_time_entry = step_num,aa,bb,cc,dd,ee,prettyfloat(veh_complete_data[aa][bb][cc][dd][ee])
                                waiting_time_data.append(waiting_time_entry)

            for agent_id in ql_agents.keys():
                reward = r[agent_id]                   
                next_state = str(env.encode(s[agent_id]))

                ql_agents[agent_id].learn(next_state, reward)

        outfile = open('Qlearning/ImportantObservation/Important-Observation-Qlearning-result-OUR-ALGORITHM-run1.csv','w')
        writer=csv.writer(outfile)
        writer.writerows(waiting_time_data)
        outfile.close()

        env.save_csv(out_csv, run)
        env.close()

        df = pd.DataFrame(infos)


