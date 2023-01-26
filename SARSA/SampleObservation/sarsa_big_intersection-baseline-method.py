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
from SARSA.env import SumoEnvironment, prettyfloat
from sumo_rl.agents.sarsa_lambda import TrueOnlineSarsaLambda


if __name__ == '__main__':

    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'SARSA/SampleObservation/Sample-Observation-SARSA-output-BASELINE-run1.csv' #+ experiment_time*

    env = SumoEnvironment(net_file='nets/4x4-Saeedeh/4x4.net.xml',
                          single_agent=False,
                          route_file='nets/4x4-Saeedeh/4x4c1c2c1c2.rou.xml',
                          out_csv_name=out_csv,
                          use_gui=False,
                          num_seconds=1000,
                          yellow_time=4,
                          min_green=5,
                          max_green=100,
                          max_depart_delay=0,
                          tripinfo_output='SARSA/SampleObservation/Sample-Observation-SARSA-tripinfo-BASELINE-run1.xml',
                          time_to_load_vehicles=300,
                          phases=[
                            traci.trafficlight.Phase(35, "GGGrrr"),   # north-south
                            traci.trafficlight.Phase(2, "yyyrrr"),
                            traci.trafficlight.Phase(35, "rrrGGG"),   # west-east
                            traci.trafficlight.Phase(2, "rrryyy")
                            ])

    fixed_tl = False

    waiting_time_data = []

    for run in range(1, 4 +1):           
        obs = {}
        action = {}
        obs = env.reset()
        agents = {ts: TrueOnlineSarsaLambda(env.observation_space, env.action_space, alpha=0.000000001, gamma=0.95, epsilon=0.05, lamb=0.1, fourier_order=7) for ts in env.ts_ids}
        done = {'__all__': False}

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if fixed_tl:
            while not done:
                _, _, done, _ = env.step(None)

        else:
            while not done['__all__']:

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                for agent_id in agents.keys():
                    action[agent_id] = agents[agent_id].act(agents[agent_id].get_features(obs[agent_id]))

                neighbors_list, time_on_phase_list, phase_id_list, step_num, veh_position_data, veh_waiting_time, veh_complete_data, next_obs, r, done, _ = env.step(action=action)

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

                                        '''Sampling using weights
                                        if veh_complete_data[aa][bb][cc][dd][ee] != 0 and sampling_size > 0 and (ee_pre == '' or ee != ee_pre):
                                            sampling_waiting_time = veh_complete_data[aa][bb][cc][dd][ee]
                                            sampling_size = sampling_size - 1
                                            ee_pre = ee
                                        '''
                                    
                                        #'''Sampling-baseline
                                        if sampling_size > 0: # and random.randint(0, 1) == 1:
                                            sampling_waiting_time = veh_complete_data[aa][bb][cc][dd][ee]  
                                            sampling_size = sampling_size - 1
                                        #'''

                                        r[aa] = r[aa] + veh_complete_data[aa][bb][cc][dd][ee] - sampling_waiting_time

                                    waiting_time_entry = step_num,aa,bb,cc,dd,ee,prettyfloat(veh_complete_data[aa][bb][cc][dd][ee])
                                    waiting_time_data.append(waiting_time_entry)

                for agent_id in agents.keys():

                    agents[agent_id].learn(state=obs[agent_id], action=action[agent_id], reward=r[agent_id], next_state=next_obs[agent_id], done=done['__all__'])
                    obs[agent_id] = next_obs[agent_id]

            outfile = open('SARSA/SampleObservation/Sample-Observation-SARSA-result-BASELINE-run1.csv','w')
            writer=csv.writer(outfile)
            writer.writerows(waiting_time_data)
            outfile.close()

            env.save_csv(out_csv, run)


 



