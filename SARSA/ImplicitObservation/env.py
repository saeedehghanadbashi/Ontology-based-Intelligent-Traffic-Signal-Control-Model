import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from gym import Env
import traci.constants as tc
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd

from .traffic_signal import TrafficSignal

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import random
from datetime import datetime
from collections import defaultdict

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

class Dict(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, Dict)
    def __repr__(self):
        return dict.__repr__(self)
#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class SumoEnvironment(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param time_to_load_vehicles: (int) Number of simulation seconds ran before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(self, net_file, route_file, tripinfo_output, phases, out_csv_name=None, use_gui=False, num_seconds=20000, max_depart_delay=100000,
                 time_to_teleport=-1, time_to_load_vehicles=0, delta_time=5, yellow_time=2, min_green=5, max_green=100, single_agent=False):

        self._net = net_file
        self._route = route_file
        self._tripinfo = tripinfo_output
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.single_agent = single_agent
        self.ts_ids = traci.trafficlight.getIDList()
        self.lanes_per_ts = len(set(traci.trafficlight.getControlledLanes(self.ts_ids[0])))
        self.traffic_signals = dict()
        self.phases = phases
        self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.vehicles = dict()
        self.last_measure = dict()  # used to reward function remember last measure
        self.last_reward = {i: 0 for i in self.ts_ids}
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles  # number of simulation seconds ran in reset() before learning starts
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.vehicles_details = Dict()

        self.ambulance_count = 0
        self.fueltruck_count = 0
        self.trailer_count = 0
        self.ambulance_count_0to1 = 0
        self.ambulance_count_4to5 = 0
        self.ambulance_count_9to10 = 0
        self.ambulance_count_14to15 = 0
        self.ambulance_count_21to4 = 0
        self.ambulance_count_0to4 = 0
        self.ambulance_count_8to9 = 0
        self.ambulance_count_5to9 = 0
        self.ambulance_count_13to14 = 0
        self.ambulance_count_10to14 = 0

        self.distance_val_per_road = {}
        self.default_vehicles_distance_val_per_road = {}
        self.last_distance = defaultdict(dict)   # used to distance function remember last distance
        self.last_distance_reward = dict()
        self.neighbors_list = dict()
        self.controlledlanes_list = dict()
        self.importance_weight_list = dict()
        self.last_default_vehicles_distance_reward = dict()
        self.last_default_vehicles_distance = defaultdict(dict)
#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        """
        Default observation space is a vector R^(#greenPhases + 2 * #lanes)
        s = [current phase one-hot encoded, density for each lane, queue for each lane]
        You can change this by modifing self.observation_space and the method _compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds
        """
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 2*self.lanes_per_ts), high=np.ones(self.num_green_phases + 2*self.lanes_per_ts))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                         # Green Phase
            #spaces.Discrete(self.max_green//self.delta_time),               # Elapsed time of phase
            *(spaces.Discrete(10) for _ in range(2*self.lanes_per_ts))      # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = ''

        self.radix_factors = [s.n for s in self.discrete_observation_space.spaces]
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.counter = 0
#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        traci.close()
        
    def reset(self):
        if self.run != 0:
            traci.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--tripinfo-output', self._tripinfo,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(self.time_to_teleport),
                     '--random']
        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green, self.phases)
            self.last_measure[ts] = 0.0

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            lanes = self.traffic_signals[ts]._getControlledLanes()
            self.last_distance[ts] = {}
            self.last_default_vehicles_distance[ts] = {}
            for lane in lanes:
                key = lane[:-2]
                self.distance_val_per_road[key] = {}
                self.default_vehicles_distance_val_per_road = {}
#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


        self.vehicles = dict()

        # Load vehicles
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    def step(self, action):

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        random_id_list = random_sample(120, 2, 10000000000000000000000000000000)
        self.counter = self.counter+1

        now = datetime.now()
        experiment_time = now.minute
        
        #'''Intuitive part-generated randomly-scenario 5-high frequency
        if  experiment_time % 2 == 0:
            while self.ambulance_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count]), "routedist1", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
                self.ambulance_count = self.ambulance_count + 1

        if  experiment_time % 2 == 0:
            while self.fueltruck_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[5+self.fueltruck_count]), "routedist1", "fueltruck", None, "random", "base", "0", "current", "max","current","","","",0, 0)
                self.fueltruck_count = self.fueltruck_count + 1

        if  experiment_time % 2 == 0:
            while self.trailer_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[10+self.trailer_count]), "routedist1", "trailer", None, "random", "base", "0", "current", "max","current","","","",0, 0)
                self.trailer_count = self.trailer_count + 1
     
        if  experiment_time % 2 != 0: 
            self.ambulance_count = 0
            self.fueltruck_count = 0
            self.trailer_count = 0
        #''' 
       
        '''generated randomly-scenario 5-high frequency
        if  self.counter % 5 == 0:
            while self.ambulance_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count]), "routedist1", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
                self.ambulance_count = self.ambulance_count + 1

        if  self.counter % 5 == 0:
            while self.fueltruck_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[5+self.fueltruck_count]), "routedist1", "fueltruck", None, "random", "base", "0", "current", "max","current","","","",0, 0)
                self.fueltruck_count = self.fueltruck_count + 1

        if  self.counter % 5 == 0:
            while self.trailer_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[10+self.trailer_count]), "routedist1", "trailer", None, "random", "base", "0", "current", "max","current","","","",0, 0)
                self.trailer_count = self.trailer_count + 1
     
        if  self.counter % 5 != 0: 
            self.ambulance_count = 0
            self.fueltruck_count = 0
            self.trailer_count = 0

        '''
        '''generated-scenario 1-high frequency
        if experiment_time % 2 == 0 and self.ambulance_count_0to1 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_0to1]), "route0to1", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_0to1 = self.ambulance_count_0to1 + 1
        if experiment_time % 2 != 0: self.ambulance_count_0to1 = 0
        '''
        '''generated-scenario 2-high frequency
        if experiment_time % 2 == 0 and self.ambulance_count_0to1 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_0to1]), "route0to1", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_0to1 = self.ambulance_count_0to1 + 1
        if experiment_time % 2 == 0 and self.ambulance_count_4to5 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_4to5]), "route4to5", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_4to5 = self.ambulance_count_4to5 + 1
        if experiment_time % 2 != 0: self.ambulance_count_0to1 = 0
        if experiment_time % 2 != 0: self.ambulance_count_4to5 = 0
        '''
        '''generated-scenario 3-high frequency
        if experiment_time % 2 == 0 and self.ambulance_count_0to1 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_0to1]), "route0to1", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_0to1 = self.ambulance_count_0to1 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_4to5 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_4to5]), "route4to5", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_4to5 = self.ambulance_count_4to5 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_9to10 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_9to10]), "route9to10", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_9to10 = self.ambulance_count_9to10 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_14to15 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_14to15]), "route14to15", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_14to15 = self.ambulance_count_14to15 + 1

        if experiment_time % 2 != 0: 
            self.ambulance_count_0to1 = 0
            self.ambulance_count_4to5 = 0
            self.ambulance_count_9to10 = 0
            self.ambulance_count_14to15 = 0
        '''
        '''generated-scenario 4-high frequency
        if experiment_time % 2 == 0 and self.ambulance_count_21to4 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_21to4]), "route21to4", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_21to4 = self.ambulance_count_21to4 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_0to4 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[5+self.ambulance_count_0to4]), "route0to4", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_0to4 = self.ambulance_count_0to4 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_8to9 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[10+self.ambulance_count_8to9]), "route8to9", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_8to9 = self.ambulance_count_8to9 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_5to9 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[15+self.ambulance_count_5to9]), "route5to9", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_5to9 = self.ambulance_count_5to9 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_13to14 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[20+self.ambulance_count_13to14]), "route13to14", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_13to14 = self.ambulance_count_13to14 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_10to14 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[25+self.ambulance_count_10to14]), "route10to14", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_10to14 = self.ambulance_count_10to14 + 1


        if experiment_time % 2 != 0: 
            self.ambulance_count_21to4 = 0
            self.ambulance_count_0to4 = 0
            self.ambulance_count_8to9 = 0
            self.ambulance_count_5to9 = 0
            self.ambulance_count_13to14 = 0
            self.ambulance_count_10to14 = 0
        '''
#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()

        else:
            self._apply_actions(action)

            for _ in range(self.yellow_time):
                self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update_phase()
            for _ in range(self.delta_time - self.yellow_time):
                self._sumo_step()

        # observe new state and reward
        observation = self._compute_observations()
        phase_id_list = self._compute_phase_id_list()
        time_on_phase_list = self._compute_time_on_phase_list()
        reward = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        info = self._compute_step_info()
        self.metrics.append(info)
        self.last_reward = reward

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        veh_complete_data = self._get_complete_data()
        distance_reward,distance_val_per_road = self._compute_distance()
        default_vehicles_distance_reward, default_vehicles_distance_val_per_road = self._compute_default_vehicles_distance()
        self.last_distance_reward = distance_reward
        self.last_default_vehicles_distance_reward = default_vehicles_distance_reward
        phase_id_list = self._compute_phase_id_list()
        importance_weight_list = self._compute_importance_weight_list()
        test_keep_list, test_give_up_list = self._compute_test_keep_list()
        time_on_phase_list = self._compute_time_on_phase_list()
        self.neighbors_list = self._get_neighbors_list_importance()
        self.controlledlanes_list = self._get_controlledlanes_list()
        self.importance_weight_list = self._get_importance_weight_list()
        veh_position_data = self._get_position_data()
        veh_waiting_time = self._get_waiting_time()
#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        if self.single_agent:
            return observation[self.ts_ids[0]], reward[self.ts_ids[0]], done['__all__'], {}
        else:
            return  self.neighbors_list, time_on_phase_list, phase_id_list, self.counter, veh_position_data, veh_waiting_time, veh_complete_data, observation, reward, done, {}

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].set_next_phase(action)

    def _compute_observations(self):
        """
        Return the current observation for each traffic signal
        """
        observations = {}
        phase_id_list = {}
        for ts in self.ts_ids:
            phase_id = [1 if self.traffic_signals[ts].phase//2 == i else 0 for i in range(self.num_green_phases)]  #one-hot encoding
            #elapsed = self.traffic_signals[ts].time_on_phase / self.max_green
            density = self.traffic_signals[ts].get_lanes_density()
            queue = self.traffic_signals[ts].get_lanes_queue()
            observations[ts] = phase_id + density + queue
            phase_id_list[ts] = phase_id
        return observations

    def _compute_phase_id_list(self):
        """
        Return the current phase indicator for each traffic signal
        """
        phase_id_list = {}
        for ts in self.ts_ids:
            phase_id = self.traffic_signals[ts].phase
            phase_id_list[ts] = phase_id
        return phase_id_list

    def _compute_rewards(self):
        return self._waiting_time_reward()
        #return self._pressure_reward()
        #return self._queue_reward()
        #return self._waiting_time_reward2()
        #return self._queue_average_reward()
    
    def _pressure_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = -self.traffic_signals[ts].get_pressure()
        return rewards

    def _queue_average_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            new_average = np.mean(self.traffic_signals[ts].get_stopped_vehicles_num())
            rewards[ts] = self.last_measure[ts] - new_average
            self.last_measure[ts] = new_average
        return rewards

    def _queue_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - (sum(self.traffic_signals[ts].get_stopped_vehicles_num()))**2
        return rewards

    def _waiting_time_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time_per_lane())
            rewards[ts] = self.last_measure[ts] - ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _waiting_time_reward2(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            self.last_measure[ts] = ts_wait
            if ts_wait == 0:
                rewards[ts] = 1.0
            else:
                rewards[ts] = 1.0/ts_wait
        return rewards

    def _waiting_time_reward3(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            rewards[ts] = -ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': self.last_reward[self.ts_ids[0]],
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(self.last_measure[ts] for ts in self.ts_ids)
            #'total_wait_time': sum([sum(self.traffic_signals[ts].get_waiting_time()) for ts in self.ts_ids])
        }

    def close(self):
        traci.close()
    
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state):
        phase = state[:self.num_green_phases].index(1)
        #elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self._discretize_density(d) for d in state[self.num_green_phases:]]
        return self.radix_encode([phase] + density_queue)

    def _discretize_density(self, density):
        return min(int(density*10), 9)

    def _discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green
        for i in range(self.max_green//self.delta_time):
            if elapsed <= self.delta_time + i*self.delta_time:
                return i
        return self.max_green//self.delta_time -1

    def radix_encode(self, values):
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

    def radix_decode(self, value):
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def _get_complete_data(self):
        """
        Return the current observation for each traffic signal
        """
        veh_complete_data = Dict()
        for ts in self.ts_ids: # ts means all traffic signals in the network       
            veh_complete_data_retval = self.traffic_signals[ts]._get_complete_data()
            veh_complete_data [ts] = veh_complete_data_retval
        return veh_complete_data

    def _get_position_data(self):
        """
        Return the current observation for each traffic signal
        """
        veh_position_data = Dict()
        for ts in self.ts_ids: # ts means all traffic signals in the network       
            veh_position_retval = self.traffic_signals[ts]._get_position_data()
            veh_position_data [ts] = veh_position_retval
        return veh_position_data

    def _get_waiting_time(self):
        """
        Return the current observation for each traffic signal
        """
        veh_waiitng_time = Dict()
        for ts in self.ts_ids: # ts means all traffic signals in the network       
            veh_waiting_time_retval = self.traffic_signals[ts]._get_waiting_time()
            veh_waiitng_time [ts] = veh_waiting_time_retval
        return veh_waiitng_time

    def _compute_distance(self): 
        rewards = {}
        ts_distance = 0
        for ts in self.ts_ids:
            distance_val_per_road = self.traffic_signals[ts]._get_distance_val()
            for key in distance_val_per_road: 
                distance_val_per_vehicle = distance_val_per_road[key]
                rewards[key] = 0
                set_reward_zero = 0
                for veh in distance_val_per_vehicle:
                    ts_distance = distance_val_per_vehicle[veh]
                    reward = 0
                    if veh in self.last_distance[ts]: reward = ts_distance - self.last_distance[ts][veh]
                    else: reward = ts_distance
                    self.last_distance[ts][veh] = ts_distance
                    if reward == 0: set_reward_zero = 1
                    rewards[key] += reward
                if set_reward_zero == 1: rewards[key] = 0
        return rewards,distance_val_per_road

    def _compute_default_vehicles_distance(self): 
        rewards = {}
        ts_distance = 0
        for ts in self.ts_ids:
            default_vehicles_distance_val_per_road = self.traffic_signals[ts]._get_default_vehicles_distance_val()
            for key in default_vehicles_distance_val_per_road: 
                default_vehicles_distance_val_per_vehicle = default_vehicles_distance_val_per_road[key]
                rewards[key] = 0
                set_reward_zero = 0
                for veh in default_vehicles_distance_val_per_vehicle:
                    ts_distance = default_vehicles_distance_val_per_vehicle[veh]
                    reward = 0
                    if veh in self.last_default_vehicles_distance[ts]: reward = ts_distance - self.last_default_vehicles_distance[ts][veh]
                    else: reward = ts_distance
                    self.last_default_vehicles_distance[ts][veh] = ts_distance
                    if reward == 0: set_reward_zero = 1
                    rewards[key] += reward
                if set_reward_zero == 1: rewards[key] = 0
        return rewards,default_vehicles_distance_val_per_road

    def _compute_phase_id_list(self):
        """
        Return the current phase indicator for each traffic signal
        """
        phase_id_list = {}
        for ts in self.ts_ids:
            phase_id = self.traffic_signals[ts].phase
            phase_id_list[ts] = phase_id
        return phase_id_list

    def _compute_importance_weight_list(self):
        """
        Return the current phase indicator for each traffic signal
        """
        importance_weight_list = {}
        for ts in self.ts_ids:
            importance_weight = self.traffic_signals[ts]._get_importance_weight_val()
            importance_weight_list[ts] = importance_weight
        return importance_weight_list


    def _compute_test_keep_list(self):
        """
        Return the current phase indicator for each traffic signal
        """
        test_keep_list = {}
        test_give_up_list = {}
        for ts in self.ts_ids:
            keep, give_up = self.traffic_signals[ts].test_keep()
            test_keep_list[ts] = keep
            test_give_up_list[ts] = give_up
        return test_keep_list, test_give_up_list

    def _compute_time_on_phase_list(self):
        """
        Return the current phase indicator for each traffic signal
        """
        time_on_phase_list = {}
        for ts in self.ts_ids:
            time_on_phase = self.traffic_signals[ts].time_on_phase
            time_on_phase_list [ts] = time_on_phase
        return time_on_phase_list 

    def _get_neighbors_list_random(self):
        rnd = str(random.randint(0, 15)) 
        number_of_neighbors = 1
        neighbors_list = {}
        for ts in self.ts_ids:
            neighbors_list_temp = []
            cnt = 0
            while cnt < number_of_neighbors:
                while rnd == ts or rnd in neighbors_list_temp: rnd = str(random.randint(0, 15))
                neighbors_list_temp.append(rnd)
                cnt = cnt + 1
            neighbors_list[ts] = neighbors_list_temp
        return neighbors_list

    def _get_neighbors_list_closeness(self):
        number_of_neighbors = 3
        neighbors_list = {}
        #defining list include position of each traffic signal
        point = {}
        point['0'] = np.array((1,4))
        point['1'] = np.array((2,4))
        point['2'] = np.array((3,4))
        point['3'] = np.array((4,4))
        point['4'] = np.array((1,3))
        point['5'] = np.array((2,3))
        point['6'] = np.array((3,3))
        point['7'] = np.array((4,3))
        point['8'] = np.array((1,2))
        point['9'] = np.array((2,2))
        point['10'] = np.array((3,2))
        point['11'] = np.array((4,2))
        point['12'] = np.array((1,1))
        point['13'] = np.array((2,1))
        point['14'] = np.array((3,1))
        point['15'] = np.array((4,1))     

        for ts in self.ts_ids:
            distance = {}
            neighbors_list_temp = []
            for neighbor_id in self.ts_ids:
                distance[neighbor_id] = np.linalg.norm(point[ts] - point[neighbor_id])
            sorted_closeness_list = sorted(distance.items(), key=lambda x: x[1], reverse=False)             
            cnt = 0
            for key in sorted_closeness_list:
                if cnt < number_of_neighbors and key[0] != ts:
                    neighbors_list_temp.append(key[0])
                    cnt = cnt + 1 
            neighbors_list[ts] = neighbors_list_temp
        return neighbors_list

    def _get_neighbors_list_importance(self):
        number_of_neighbors_selected_in_first_stage = 1
        number_of_neighbors_selected_in_second_stage = 1
        number_of_neighbors_selected_in_third_stage = 1
        neighbors_list = {}  
        importance_weight_list_per_ts = self._get_importance_weight_list_per_ts()

        #defining list include position of each traffic signal
        point = {}
        point['0'] = np.array((1,4))
        point['1'] = np.array((2,4))
        point['2'] = np.array((3,4))
        point['3'] = np.array((4,4))
        point['4'] = np.array((1,3))
        point['5'] = np.array((2,3))
        point['6'] = np.array((3,3))
        point['7'] = np.array((4,3))
        point['8'] = np.array((1,2))
        point['9'] = np.array((2,2))
        point['10'] = np.array((3,2))
        point['11'] = np.array((4,2))
        point['12'] = np.array((1,1))
        point['13'] = np.array((2,1))
        point['14'] = np.array((3,1))
        point['15'] = np.array((4,1))   
 
        for ts in self.ts_ids:
            importance_weight_list_with_distance = {}
            sorted_importance_weight_list_with_distance = {} 
            neighbors_list_temp = []
            for neighbor_id in self.ts_ids:
                distance = np.linalg.norm(point[ts] - point[neighbor_id])
                if distance!= 0: reverse_distance = 1/distance
                else: reverse_distance = 0
                importance_weight_list_with_distance[neighbor_id] = [reverse_distance,importance_weight_list_per_ts[neighbor_id]]
            
            sorted_importance_weight_list_with_distance = sorted(importance_weight_list_with_distance.items(), key=lambda x: x[1], reverse=True)
            
            cnt = 0 
            for key in sorted_importance_weight_list_with_distance:
                if cnt < number_of_neighbors_selected_in_first_stage and key[0] != ts:
                    neighbors_list_temp.append(key[0])
                    first_selected_neighbor = key[0]
                    cnt = cnt + 1 

            importance_weight_list_with_distance = {}
            sorted_importance_weight_list_with_distance = {}  
            for neighbor_id in self.ts_ids:
                distance = np.linalg.norm(point[first_selected_neighbor] - point[neighbor_id])
                if distance!= 0: reverse_distance = 1/distance
                else: reverse_distance = 0
                importance_weight_list_with_distance[neighbor_id] = [reverse_distance,importance_weight_list_per_ts[neighbor_id]]
            
            sorted_importance_weight_list_with_distance = sorted(importance_weight_list_with_distance.items(), key=lambda x: x[1], reverse=True)
            
            cnt = 0 
            for key in sorted_importance_weight_list_with_distance:
                if cnt < number_of_neighbors_selected_in_second_stage and key[0] != ts:
                    neighbors_list_temp.append(key[0])
                    second_selected_neighbor = key[0]
                    cnt = cnt + 1 

            importance_weight_list_with_distance = {}
            sorted_importance_weight_list_with_distance = {}  
            for neighbor_id in self.ts_ids:
                distance = np.linalg.norm(point[second_selected_neighbor] - point[neighbor_id])
                if distance!= 0: reverse_distance = 1/distance
                else: reverse_distance = 0
                importance_weight_list_with_distance[neighbor_id] = [reverse_distance,importance_weight_list_per_ts[neighbor_id]]
            
            sorted_importance_weight_list_with_distance = sorted(importance_weight_list_with_distance.items(), key=lambda x: x[1], reverse=True)
            
            cnt = 0 
            for key in sorted_importance_weight_list_with_distance:
                if cnt < number_of_neighbors_selected_in_third_stage and key[0] != ts:
                    neighbors_list_temp.append(key[0])
                    cnt = cnt + 1 
            neighbors_list[ts] = neighbors_list_temp
        return neighbors_list

    def _get_controlledlanes_list(self):
        controlledlanes_list = {}
        for ts in self.ts_ids:
            lanes = self.traffic_signals[ts]._getControlledLanes()
            controlledlanes_list[ts] = lanes
        return controlledlanes_list

    def _get_importance_weight_list(self):
        importance_weight_list = {}
        for ts in self.ts_ids:
            importance_weight = self.traffic_signals[ts]._get_importance_weight_val()
            importance_weight_list[ts] = importance_weight
        return importance_weight_list

    def _get_importance_weight_list_per_ts(self):
        importance_weight_list_per_ts = {}
        for ts in self.ts_ids:
            importance_weight = self.traffic_signals[ts]._get_importance_weight_val_per_ts()
            importance_weight_list_per_ts[ts] = importance_weight
        return importance_weight_list_per_ts

def random_sample(count, start, stop, step=1):
    def gen_random():
        while True:
            yield random.randrange(start, stop, step)

    def gen_n_unique(source, n):
        seen = set()
        seenadd = seen.add
        for i in (i for i in source() if i not in seen and not seenadd(i)):
            yield i
            if len(seen) == n:
                break

    return [i for i in gen_n_unique(gen_random,
                                    min(count, int(abs(stop - start) / abs(step))))]

