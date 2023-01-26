import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
from collections import defaultdict

class Dict(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, Dict)
    def __repr__(self):
        return dict.__repr__(self)
#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, phases):
        self.id = ts_id
        self.env = env
        self.time_on_phase = 0.0
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.num_green_phases = len(phases) // 2
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id)]
        self.out_lanes = list(set(self.out_lanes))

        logic = traci.trafficlight.Logic("new-program", 0, 0, phases=phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases] 
        """

        keep, give_up = self.test_keep() 
        new_phase *= 2
        if (self.phase == new_phase or self.time_on_phase < self.min_green): # or keep == 1) and give_up != 1:
            self.time_on_phase += self.delta_time
            self.green_phase = self.phase
        else:
            self.time_on_phase = self.delta_time - self.yellow_time
            self.green_phase = new_phase
            traci.trafficlight.setPhase(self.id, self.phase + 1)  # turns yellow
           
    def update_phase(self):
        """
        Change the next green_phase after it is set by set_next_phase method
        """
        traci.trafficlight.setPhase(self.id, self.green_phase)

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_pressure(self):
        return abs(sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]
    
    def get_total_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self, p):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

#new added ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def _get_complete_data(self):
        veh_complete_data = Dict()
        veh_lane_list = []
        veh_list = []
        total_veh_list = []
        p_list = []
        for p in range(self.num_green_phases):
            if p not in p_list: 
                p_list.append(p) 
            veh_list = self._get_veh_list(p)
            total_veh_list += veh_list
            wait_time = 0.0
            for veh in veh_list:  
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_position = traci.vehicle.getLanePosition(veh)
                veh_type = traci.vehicle.getTypeID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles_details[veh][veh_type][veh_position] = {veh_lane: acc}		
                else:
                    self.env.vehicles_details[veh][veh_type][veh_position][veh_lane] = acc - sum([self.env.vehicles_details[veh][veh_type][veh_position][lane] for lane in self.env.vehicles_details[veh][veh_type][veh_position].keys() if lane != veh_lane])
                veh_complete_data[veh][veh_type][veh_position][veh_lane] = self.env.vehicles_details[veh][veh_type][veh_position][veh_lane]
        return veh_complete_data

    def _get_position_data(self):
        veh_position_data = {}
        for p in range(self.num_green_phases):
            veh_list = self._get_veh_list(p)
            for veh in veh_list:  
                veh_position = traci.vehicle.getLanePosition(veh)
                veh_position_data[veh] = veh_position
        return veh_position_data

    def _get_waiting_time(self):
        veh_waiting_time = Dict()
        veh_lane_list = []
        veh_list = []
        for p in range(self.num_green_phases):
            veh_list = self._get_veh_list(p)
            wait_time = 0.0
            for veh in veh_list:  
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_position = traci.vehicle.getLanePosition(veh)
                veh_type = traci.vehicle.getTypeID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles_details[veh][veh_type][veh_position] = {veh_lane: acc}		
                else:
                    self.env.vehicles_details[veh][veh_type][veh_position][veh_lane] = acc - sum([self.env.vehicles_details[veh][veh_type][veh_position][lane] for lane in self.env.vehicles_details[veh][veh_type][veh_position].keys() if lane != veh_lane])
                veh_waiting_time[veh] = self.env.vehicles_details[veh][veh_type][veh_position][veh_lane]
        return veh_waiting_time

    def get_edge_id(self, lane):
        ''' Get edge Id from lane Id
        :param lane: id of the lane
        :return: the edge id of the lane
        '''
        return lane[:-2]

    def test_keep(self):
        """
        Return the current observation for each traffic signal
        """
        keep = 0
        give_up = 0

        ambulance_lane_list, trailer_lane_list, fueltruck_lane_list = self._get_important_objects_lane_list()
        keep = 0
        give_up = 0
        for lane in self.lanes:
            k = lane[:-2]
            if k in self.env.last_distance_reward:
                if self.env.last_distance_reward[k] > 0: keep = 1 
                if self.env.last_distance_reward[k] == 0 and k in ambulance_lane_list: give_up = 1
        return keep, give_up

    def _get_distance_val(self):
        veh_list_part1,veh_list_part2 = self._get_veh_list_per_edge()
        for key in veh_list_part1:
            self.env.distance_val_per_road[key] = {}
            for veh in veh_list_part1[key]:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_type = traci.vehicle.getTypeID(veh)               
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    self.env.distance_val_per_road[key][veh] = traci.vehicle.getLanePosition(veh) 
        for key in veh_list_part2:
            for veh in veh_list_part2[key]:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_type = traci.vehicle.getTypeID(veh)               
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    self.env.distance_val_per_road[key][veh] = traci.vehicle.getLanePosition(veh) 
        return self.env.distance_val_per_road

    def _get_default_vehicles_distance_val(self):
        veh_list_part1,veh_list_part2 = self._get_veh_list_per_edge()
        for key in veh_list_part1:
            self.env.default_vehicles_distance_val_per_road[key] = {}
            for veh in veh_list_part1[key]:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_type = traci.vehicle.getTypeID(veh)               
                if veh_type == "DEFAULT_VEHTYPE":
                    self.env.default_vehicles_distance_val_per_road[key][veh] = traci.vehicle.getLanePosition(veh) 
        for key in veh_list_part2:
            for veh in veh_list_part2[key]:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_type = traci.vehicle.getTypeID(veh)               
                if veh_type == "DEFAULT_VEHTYPE":
                    self.env.default_vehicles_distance_val_per_road[key][veh] = traci.vehicle.getLanePosition(veh) 
        return self.env.default_vehicles_distance_val_per_road

    def _getControlledLanes(self):
        return list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))

    def _get_veh_list_per_edge(self):
        getControlledLanes = self._getControlledLanes()
        veh_list_part1 = {}
        veh_list_part2 = {}
        veh_list = {}
        cnt_part = 0
        for p in getControlledLanes:            
            pl = p[:-2]
            if cnt_part % 2 == 0: veh_list_part1[pl] = traci.lane.getLastStepVehicleIDs(p)
            elif cnt_part % 2 != 0: veh_list_part2[pl] = traci.lane.getLastStepVehicleIDs(p)
            veh_list[pl] = traci.lane.getLastStepVehicleIDs(p)
            cnt_part += 1
        return veh_list_part1,veh_list_part2

    def _get_importance_weight_val(self):
        veh_list_part1,veh_list_part2 = self._get_veh_list_per_edge()
        importance_weight_val_per_road = {}
        cnt = {}        
        for key in veh_list_part1:
            cnt[key] = 0
            if cnt[key] == 0: importance_weight_val_per_road[key] = 0  
            for veh in veh_list_part1[key]:
                veh_type = traci.vehicle.getTypeID(veh)               
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    importance_weight_val_per_road[key] = importance_weight_val_per_road[key] + 1
                    cnt[key] = 1
        for key in veh_list_part2:
            if cnt[key] == 0: importance_weight_val_per_road[key] = 0  
            for veh in veh_list_part2[key]:
                veh_type = traci.vehicle.getTypeID(veh)               
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    importance_weight_val_per_road[key] = importance_weight_val_per_road[key] + 1
                    cnt[key] = 1
        return importance_weight_val_per_road

    def _get_importance_weight_val_per_ts(self):
        veh_list_part1,veh_list_part2 = self._get_veh_list_per_edge()   
        importance_weight_val_per_ts = 0   
        for key in veh_list_part1:
            for veh in veh_list_part1[key]:
                veh_type = traci.vehicle.getTypeID(veh)               
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    importance_weight_val_per_ts = importance_weight_val_per_ts + 1
        for key in veh_list_part2:
            for veh in veh_list_part2[key]:
                veh_type = traci.vehicle.getTypeID(veh)               
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    importance_weight_val_per_ts = importance_weight_val_per_ts + 1
        return importance_weight_val_per_ts

    def _get_important_objects_lane_list(self):
        getControlledLanes = self._getControlledLanes()
        ambulance_lane_list = []
        trailer_lane_list = []
        fueltruck_lane_list = []
        for p in getControlledLanes:            
            pl = p[:-2]
            veh_list = self._get_veh_list(p)
            for veh in veh_list:  
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "Ambulance": ambulance_lane_list.append(veh_lane)
                if veh_type == "trailer": trailer_lane_list.append(veh_lane)
                if veh_type == "fueltruck": fueltruck_lane_list.append(veh_lane)
        return ambulance_lane_list, trailer_lane_list, fueltruck_lane_list

