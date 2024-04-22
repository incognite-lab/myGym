import numpy as np
import matplotlib.pyplot as plt
#from stable_baselines import results_plotter
import os
import math
from math import sqrt, fabs, exp, pi, asin
from myGym.utils.vector import Vector
import random

class Reward:
    """
    Reward base class for reward signal calculation and visualization

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task=None):
        self.env = env
        self.task = task
        self.rewards_history = []
        self.current_network = 0
        self.num_networks = env.num_networks
        #self.check_num_networks()
        self.network_rewards = [0] * self.num_networks

    def network_switch_control(self, observation):
        if self.env.num_networks <= 1:
            print("Cannot switch networks in a single-network scenario")
        else:
           if self.env.network_switcher == "gt":
                self.current_network = self.decide(observation)
           else:
               raise NotImplementedError("Currently only implemented ground truth ('gt') network switcher")
        return self.current_network

    def compute(self, observation=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def visualize_reward_over_steps(self):
        """
        Plot and save a graph of reward values assigned to individual steps during an episode. Call this method after the end of the episode.
        """
        save_dir = os.path.join(self.env.logdir, "rewards")
        os.makedirs(save_dir, exist_ok=True)
        if self.env.episode_steps > 0:
            #results_plotter.EPISODES_WINDOW=50
            #results_plotter.plot_curves([(np.arange(self.env.episode_steps),np.asarray(self.rewards_history[-self.env.episode_steps:]))],'step','Step rewards')
            plt.ylabel("reward")
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(save_dir + "/reward_over_steps_episode{}.png".format(self.env.episode_number))
            plt.close()

    def visualize_reward_over_episodes(self):
        """
        Plot and save a graph of cumulative reward values assigned to individual episodes. Call this method to plot data from the current and all previous episodes.
        """
        save_dir = os.path.join(self.env.logdir, "rewards")
        os.makedirs(save_dir, exist_ok=True)
        if self.env.episode_number > 0:
            #results_plotter.EPISODES_WINDOW=10
            #results_plotter.plot_curves([(np.arange(self.env.episode_number),np.asarray(self.env.episode_final_reward[-self.env.episode_number:]))],'episode','Episode rewards')
            plt.ylabel("reward")
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(save_dir + "/reward_over_episodes_episode{}.png".format(self.env.episode_number))
            plt.close()

    def get_magnetization_status(self):
        return self.env.robot.use_magnet
    


# PROTOREWARDS

class Protorewards(Reward):

    def reset(self):
        self.last_owner = None
        self.last_find_dist  = None
        self.last_approach_dist  = None
        self.last_grip_dist  = None
        self.last_lift_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.last_rot_dist = None
        self.subgoaloffset_dist = None
        self.last_leave_dist = None
        self.prev_object_position = None
        self.was_near = False
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
        self.has_left = False
        self.last_traj_idx = 0
        self.last_traj_dist = 0
        self.offset = [0.3,0.0,0.0]
        self.offsetleft = [0.2,0.0,-0.1]
        self.offsetright = [-0.2,0.0,-0.1]
        self.offsetcenter = [0.0,0.0,-0.1]
        self.grip_threshold = 0.1
        self.approached_threshold = 0.05
        self.opengr_threshold = 0.07
        self.closegr_threshold = 0.001
        self.near_threshold = 0.1
        self.lift_threshold = 0.1
        self_above_threshold = 0.1
    
    def compute(self, observation=None):
        #inherit and define your sequence of protoactions here
        pass

    def decide(self, observation=None):
        #inherit and define subgoals checking and network switching here
        pass
    
    def find_compute(self, gripper, object):
        self.env.p.addUserDebugText("find object", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        self.env.robot.set_magnetization(True)
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_find_dist is None:
            self.last_find_dist = dist
        
        reward = self.last_find_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.63, 0.8,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        
        self.last_find_dist = dist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,0.6,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    def approach_compute(self, gripper, object,gripper_states):
        self.env.p.addUserDebugText("approach object", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        self.env.robot.set_magnetization(False)
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) + ((gripdist - self.last_grip_dist)*0.2)
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.63, 0.8,0.55], lifeTime=0.5, textColorRGB=[0,125,0])     
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,0.6,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    def withdraw_compute(self, gripper, object,gripper_states):
        self.env.p.addUserDebugText("withdraw object", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        self.env.robot.set_magnetization(False)
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (dist - self.last_approach_dist) + ((gripdist - self.last_grip_dist)*0.2)
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.63, 0.8,0.55], lifeTime=0.5, textColorRGB=[0,125,0])     
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,0.6,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    def grasp_compute(self, gripper, object,gripper_states):
        self.env.p.addUserDebugText("grasp object", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        self.env.robot.set_magnetization(False)
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) + ((self.last_grip_dist - gripdist)*0.2)
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.63, 0.8,0.55], lifeTime=0.5, textColorRGB=[0,125,0])     
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,0.6,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        print(self.last_approach_dist)
        return reward
    
    def drop_compute(self, gripper, object,gripper_states):
        self.env.p.addUserDebugText("drop object", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        #self.env.robot.set_magnetization(False)
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) + ((gripdist - self.last_grip_dist)*0.2)
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.63, 0.8,0.55], lifeTime=0.5, textColorRGB=[0,125,0])     
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,0.6,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        print(self.last_approach_dist)
        return reward

    def move_compute(self, object, goal, gripper_states):
        self.env.robot.set_magnetization(True)
        self.env.p.addUserDebugText("move", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        object_XY = object[:3]
        goal_XY   = goal[:3]
        gripdist = sum(gripper_states)
        dist = self.task.calc_distance(object_XY, goal_XY)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_move_dist is None:
           self.last_move_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_move_dist - dist) + ((self.last_grip_dist - gripdist)*0.2)
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61, 0.8,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime = 0.1)
        self.last_move_dist = dist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65, 0.6,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    
    def rotate_compute(self, object, goal):
        self.env.p.addUserDebugText("rotate", [0.63,1,0.5], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.robot.set_magnetization(False)
        dist = self.task.calc_distance(object, goal)
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist

        rot = self.task.calc_rot_quat(object, goal)
        self.env.p.addUserDebugText("Rotation: {}".format(round(rot,3)), [0.65,1,0.6], lifeTime=0.5, textColorRGB=[0, 222, 100])
        if self.last_rot_dist is None: 
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        
        reward = reward + rewardrot
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        self.last_rot_dist = rot
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward

    def transform_compute(self, object, goal, trajectory, magnetization = True):
        """Calculate reward based on following a trajectory
        params: object: self-explanatory
                goal: self-explanatory
                trajectory: (np.array) 3D trajectory, lists of points x, y, z
                magnetization: (boolean) sets magnetization on or off
        Reward is calculated based on distance of object from goal and square distance of object from trajectory.
        That way, object tries to approach goal while trying to stay on trajectory path.
        """
        self.env.robot.set_magnetization(magnetization)
        self.env.p.addUserDebugText("transform", [0.7, 0.7, 0.7], lifeTime=0.1, textColorRGB=[125, 125, 0])
        dist_g = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist_g

        reward_g_dist = self.last_place_dist - dist_g #distance from goal
        self.env.p.addUserDebugText(f"RewardDist:{reward_g_dist}", [0.61, 1, 0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])

        pos = object[:3]
        dist_t, self.last_traj_idx = self.task.trajectory_distance(trajectory, pos, self.last_traj_idx, 10)

        self.env.p.addUserDebugText(f"Traj_Dist:{dist_t}", [0.61, 1, 0.45], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_traj_dist is None:
            self.last_traj_dist = dist_t
        reward_t_dist = self.last_traj_dist - dist_t #distance from trajectory
        reward = reward_g_dist + 4*reward_t_dist
        self.env.p.addUserDebugLine(trajectory[:3,0], trajectory[:3, -1], lifeTime = 0.1)
        self.env.p.addUserDebugText(f"reward:{reward}", [0.61, 1, 0.35], lifeTime=0.5, textColorRGB=[0, 125, 0])

        self.last_place_dist = dist_g
        self.last_traj_dist = dist_t
        self.network_rewards[self.current_network] += reward
        return reward
    
    def follow_compute(self, object, goal, trajectory, magnetization = True):
        """Calculate reward based on following a trajectory
        params: object: self-explanatory
                goal: self-explanatory
                trajectory: (np.array) 3D trajectory, lists of points x, y, z
                magnetization: (boolean) sets magnetization on or off
        Reward is calculated based on distance of object from goal and square distance of object from trajectory.
        That way, object tries to approach goal while trying to stay on trajectory path.
        """
        self.env.robot.set_magnetization(magnetization)
        self.env.p.addUserDebugText("transform", [0.7, 0.7, 0.7], lifeTime=0.1, textColorRGB=[125, 125, 0])
        dist_g = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist_g

        reward_g_dist = self.last_place_dist - dist_g #distance from goal
        self.env.p.addUserDebugText(f"RewardDist:{reward_g_dist}", [0.61, 1, 0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])

        pos = object[:3]
        dist_t, self.last_traj_idx = self.task.trajectory_distance(trajectory, pos, self.last_traj_idx, 10)

        self.env.p.addUserDebugText(f"Traj_Dist:{dist_t}", [0.61, 1, 0.45], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_traj_dist is None:
            self.last_traj_dist = dist_t
        reward_t_dist = self.last_traj_dist - dist_t #distance from trajectory
        reward = reward_g_dist + 4*reward_t_dist
        self.env.p.addUserDebugLine(trajectory[:3,0], trajectory[:3, -1], lifeTime = 0.1)
        self.env.p.addUserDebugText(f"reward:{reward}", [0.61, 1, 0.35], lifeTime=0.5, textColorRGB=[0, 125, 0])

        self.last_place_dist = dist_g
        self.last_traj_dist = dist_t
        self.network_rewards[self.current_network] += reward
        return reward

    

    # PREDICATES

    def get_positions(self, observation):
        goal_position = observation["goal_state"]
        object_position = observation["actual_state"]
        #gripper_name = [x for x in self.env.task.obs_template["additional_obs"] if "endeff" in x][0]
        gripper_position = self.env.robot.get_accurate_gripper_position() #observation["additional_obs"][gripper_name][:3]
        gripper_position = observation["additional_obs"]["endeff_xyz"]
        gripper_states = self.env.robot.get_gjoints_states()
        if self.prev_object_position is None:
            self.prev_object_position = object_position
        return goal_position,object_position,gripper_position,gripper_states
    
    def gripper_reached_object(self, gripper, object):
        if self.task.calc_distance(gripper, object) <= self.grip_threshold:
            self.env.robot.magnetize_object(self.env.env_objects["actual_state"])
            return True
        return False
    
    def gripper_approached_object (self, gripper, object):
        if self.task.calc_distance(gripper, object) <= self.approached_threshold:
            return True
        return False
    
    def gripper_opened(self, gripper_states):
        if sum(gripper_states) >= self.opengr_threshold:
            print("Gripper opened")
            return True
        return False
    
    def gripper_closed(self, gripper_states):
        #print(self.env.robot.gripper_active)
        if sum(gripper_states) <= self.closegr_threshold:
            self.env.robot.grasp_object(self.env.env_objects["actual_state"])
            self.env.robot.set_magnetization(True)
            print("Gripper closed and magnetized")
            return True
        return False

    def object_lifted(self, object, object_before_lift):
        lifted_position = [object_before_lift[0], object_before_lift[1], object_before_lift[2]+0.1] # position of object before lifting but hightened with its height
        self.task.calc_distance(object, lifted_position)
        if object[2] < self.lift_threshold:
            self.lifted = False # object has fallen
            self.object_before_lift = object
        else:
            self.lifted = True
            return True
        return False

    def object_above_goal(self, object, goal):
        goal_XY   = [goal[0], goal[1], goal[2]+0.2]
        object_XY = object
        distance  = self.task.calc_distance(goal_XY, object_XY)
        if distance < self.above_threshhold:
            return True
        return False
    
    def left_out_of_threshold(self, gripper, object, threshold = 0.2):
        distance = self.task.calc_distance(gripper ,object)
        if distance > threshold:
            return True
        return False

    def object_near_goal(self, object, goal):
        distance  = self.task.calc_distance(goal, object)
        if distance < self.near_threshold:
            return True
        return False
    
    def subgoal_offset(self, goal_position,offset):
        
        subgoal = [goal_position[0]-offset[0],goal_position[1]-offset[1],goal_position[2]-offset[2],goal_position[3],goal_position[4],goal_position[5],goal_position[6]]
        return subgoal


# RDDL

class A(Protorewards):
    
    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(goal_position, object_position, gripper_position, gripper_states)
        target = [[object_position,goal_position,gripper_states]][owner]
        reward = [self.approach_compute][owner](*target)
        self.last_owner = owner        
        self.rewards_history.append(reward)
        return reward
         
    def decide(self,goal_position, object_position, gripper_position, gripper_states):
        if self.gripper_approached_object(object_position, goal_position):
            if self.gripper_opened(gripper_states):
                #self.current_network += 1
                self.task.check_goal()
        return self.current_network

class AaG(Protorewards):
    
    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(goal_position, object_position, gripper_position, gripper_states)
        target = [[object_position,goal_position,gripper_states], [object_position,goal_position,gripper_states]][owner]
        reward = [self.approach_compute, self.grasp_compute][owner](*target)
        self.last_owner = owner
        self.rewards_history.append(reward)
        return reward
    
    def decide(self, goal_position, object_position, gripper_position, gripper_states):
        if self.gripper_approached_object(object_position, goal_position):
            if self.gripper_opened(gripper_states):
                self.current_network = 1
                #self.task.check_goal()
            if self.current_network == 1:
                if self.gripper_closed(gripper_states):
                    self.task.check_goal()

        return self.current_network

class AaGaM(Protorewards):
    
    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(goal_position, object_position, gripper_position, gripper_states)
        target = [[gripper_position,object_position,gripper_states], [gripper_position,object_position,gripper_states], [object_position,goal_position,gripper_states]][owner]
        reward = [self.approach_compute, self.grasp_compute, self.move_compute][owner](*target)
        self.last_owner = owner
        self.rewards_history.append(reward)
        return reward
    
    def decide(self, goal_position, object_position, gripper_position, gripper_states):
        print(self.get_magnetization_status())
        if self.current_network == 0:
            if self.gripper_approached_object(gripper_position, object_position):
                if self.gripper_opened(gripper_states):
                    self.current_network = 1
                    #self.task.check_goal()
        if self.current_network == 1:
            if self.gripper_approached_object(gripper_position, object_position):
                if self.gripper_closed(gripper_states):
                    self.current_network = 2
        if self.current_network == 2:
            if self.object_near_goal(object_position, goal_position):
                print("Goal reached")
                self.task.check_goal()

        return self.current_network

class AaGaMaD(Protorewards):
    
    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(goal_position, object_position, gripper_position, gripper_states)
        target = [[gripper_position,object_position,gripper_states], [gripper_position,object_position,gripper_states], [object_position,goal_position,gripper_states]][owner]
        reward = [self.approach_compute, self.grasp_compute, self.move_compute, self.drop_compute][owner](*target)
        self.last_owner = owner
        self.rewards_history.append(reward)
        return reward
    
    def decide(self, goal_position, object_position, gripper_position, gripper_states):
        print(self.get_magnetization_status())
        if self.current_network == 0:
            if self.gripper_approached_object(gripper_position, object_position):
                if self.gripper_opened(gripper_states):
                    self.current_network = 1
                    #self.task.check_goal()
        if self.current_network == 1:
            if self.gripper_approached_object(gripper_position, object_position):
                if self.gripper_closed(gripper_states):
                    self.current_network = 2
        if self.current_network == 2:
            if self.object_near_goal(object_position, goal_position):
                print("Goal reached")
                self.task.check_goal()

class AaGaMaDaW(Protorewards):
    
    def compute(self, observation=None):
        owner = 0
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position]][owner]
        reward = [self.approach_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

class F(Protorewards):
    
    def compute(self, observation=None):
        owner = 0
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position]][owner]
        reward = [self.find_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward
    
class FaM(Protorewards):
    
    def compute(self, observation=None):
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        return self.current_network

class FaMaR(Protorewards):
    
    def compute(self, observation=None):
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, goal_position], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, goal_position) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network



class FaROaM(Protorewards):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position,self.offset)], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.rotate_compute, self.move_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,self.offset)) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network

class FaMOaR(Protorewards):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position,self.offset)], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,self.offset)) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network

class FaMOaM(Protorewards):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position,self.offset)], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.move_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,self.offset)) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network

class FaMOaT(Protorewards):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        #[object_position, goal_position, self.task.create_line(self.subgoal_offset(goal_position), [0.3,0.0,0.0], goal_position), True]
        transform_line = self.task.create_line(self.subgoal_offset(goal_position, self.offset), goal_position)
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position,self.offset)], [object_position, goal_position, transform_line]][owner]
        reward = [self.find_compute,self.move_compute, self.transform_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,self.offset)) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network

class FaROaT(Protorewards):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        transform_line = self.task.create_line(self.subgoal_offset(goal_position, self.offset), goal_position)
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position, self.offset)], [object_position, goal_position, transform_line]][owner]
        reward = [self.find_compute,self.rotate_compute, self.transform_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,self.offset)) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network
    

class FaMaLaFaR(Protorewards):
    def check_num_networks(self):
        assert self.num_networks <= 4, "Find&move&leave&find&rotate reward can work with maximum of 4 networks"

    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, goal_position], [gripper_position, object_position], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.leave_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position) and self.current_network == 0:
            self.current_network = 1
        if self.object_near_goal(object_position, goal_position) and self.has_left == False and self.current_network == 1:
            self.current_network = 2
        if self.left_out_of_threshold(gripper_position, goal_position, threshold = 0.3) and self.current_network == 2:
            self.current_network = 0
            self.has_left = True
        if self.gripper_reached_object(gripper_position, object_position) and self.has_left:
            self.current_network = 3
        return self.current_network

class SwitchRewardNew(Protorewards):
    def check_num_networks(self):
        assert self.num_networks <= 2, "Switch reward can work with maximum of 2 networks"

    def gripper_reached_object(self, gripper, object):
        if self.task.calc_distance(gripper, object) <= 0.1:
            return True
        return False


    def compute(self, observation=None):
        owner = self.decide(observation)
        #print("owner = ", owner)
        self.offsetleft=[0.2,0.0,-0.3]
        self.offsetright=[-0.2,0.0,-0.3]
        goal_position, object_position, gripper_position = self.get_positions(observation)
        transform_line = self.task.create_line(self.subgoal_offset(goal_position, self.offsetleft), self.subgoal_offset(goal_position, self.offsetright))
        target = [[gripper_position, self.subgoal_offset(goal_position, self.offsetleft)],
               [gripper_position, self.subgoal_offset(goal_position, self.offsetright), transform_line]][owner]
        reward = [self.find_compute, self.transform_compute][owner](*target)
        self.env.p.addUserDebugLine(self.subgoal_offset(goal_position, self.offsetleft)[:3], self.subgoal_offset(goal_position, self.offsetright)[:3])
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        #print("goal:", goal_position)
        #print("gripper:", gripper_position)
        if self.gripper_reached_object(gripper_position, self.subgoal_offset(goal_position, self.offsetleft)):
            self.current_network = 1
        return self.current_network



class TurnRewardNew(Protorewards):
    def check_num_networks(self):
        assert self.num_networks <= 2, "Switch reward can work with maximum of 2 networks"


    def gripper_reached_object(self, gripper, object):
        if self.task.calc_distance(gripper, object) <= 0.1:
            return True
        return False

    def compute(self, observation=None):
        owner = self.decide(observation)
        # print("owner = ", owner)
        self.offsetleft=[0.2,0.0,-0.1]
        self.offsetright=[-0.2,0.0,-0.1]
        self.offsetcenter=[0.0,0.0,-0.1]
        goal_position, object_position, gripper_position = self.get_positions(observation)
        transform_circle = self.task.create_circular_trajectory(self.subgoal_offset(goal_position, self.offsetcenter)[:3], 0.2, arc =np.pi, direction = -1)
        target = [[gripper_position, self.subgoal_offset(goal_position, self.offsetleft)],
                  [gripper_position, self.subgoal_offset(goal_position, self.offsetright), transform_circle]][owner]
        reward = [self.find_compute, self.transform_compute][owner](*target)
        self.env.p.addUserDebugLine(self.subgoal_offset(goal_position, self.offsetleft)[:3],
                                    self.subgoal_offset(goal_position, self.offsetright)[:3])
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        # print("goal:", goal_position)
        # print("gripper:", gripper_position)
        if self.gripper_reached_object(gripper_position, self.subgoal_offset(goal_position, self.offsetleft)):
            self.current_network = 1
        return self.current_network