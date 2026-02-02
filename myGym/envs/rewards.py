import numpy as np
import matplotlib.pyplot as plt
# from stable_baselines import results_plotter
import os
import math
from math import sqrt, fabs, exp, pi, asin
from myGym.utils.vector import Vector
import random
import time

GREEN = [0, 125, 0]
RED = [125, 0, 0]


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
        self.network_rewards = [0] * self.num_networks

    def network_switch_control(self, observation):

        if self.env.network_switcher == "gt":
            self.current_network = self.decide(observation)
        elif self.env.network_switcher == "keyboard":
            keypress = self.env.p.getKeyboardEvents()
            if 107 in keypress.keys() and keypress[107] == 1:  # K
                if self.current_network < self.num_networks - 1:
                    self.current_network += 1
            elif 106 in keypress.keys() and keypress[106] == 1:  # J
                if self.current_network > 0:
                    self.current_network -= 1
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
            # results_plotter.EPISODES_WINDOW=50
            # results_plotter.plot_curves([(np.arange(self.env.episode_steps),np.asarray(self.rewards_history[-self.env.episode_steps:]))],'step','Step rewards')
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
            # results_plotter.EPISODES_WINDOW=10
            # results_plotter.plot_curves([(np.arange(self.env.episode_number),np.asarray(self.env.episode_final_reward[-self.env.episode_number:]))],'episode','Episode rewards')
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
        self.last_find_dist = None
        self.last_approach_dist = None
        self.last_grip_dist = None
        self.last_lift_dist = None
        self.last_move_dist = None
        self.last_place_dist = None
        self.last_rot_dist = None
        self.subgoaloffset_dist = None
        self.last_leave_dist = None
        self.prev_object_position = None
        self.was_near = False
        self.current_network = 0
        self.eval_network_rewards = self.network_rewards
        self.network_rewards = [0] * self.num_networks
        self.has_left = False
        self.last_traj_idx = 0
        self.last_traj_dist = 0
        self.offset = [0.3, 0.0, 0.0]
        self.offsetleft = [0.2, 0.0, -0.1]
        self.offsetright = [-0.2, 0.0, -0.1]
        self.offsetcenter = [0.0, 0.0, -0.1]
        self.grip_threshold = 0.1
        self.approached_threshold = 0.045
        self.withdraw_threshold = 0.3
        self.near_threshold = 0.05
        self.lift_threshold = 0.1
        self_above_threshold = 0.1
        self.above_offset = 0.02
        self.reward_name = None
        self.iter = 1

    def compute(self, observation=None):
        # inherit and define your sequence of protoactions here
        pass

    def decide(self, observation=None):
        # inherit and define subgoals checking and network switching here
        pass

    def show_every_n_iters(self, text, value, n):
        """
        Every n iterations (steps) show given text and value.
        """
        if self.iter % n == 0:
            print(text, value)
        self.iter += 1

    def disp_reward(self, reward, owner):
        "Display reward in green if it's positive or in red if it's negative"
        if reward > 0:
            color = GREEN
        else:
            color = RED
        if self.network_rewards[owner] > 0:
            color_sum = GREEN
        else:
            color_sum = RED
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.63, 0.8, 0.55], lifeTime=0.5, textColorRGB=color)
        self.env.p.addUserDebugText(f"Reward sum for network{owner}, :{self.network_rewards[owner]}", [0.65, 0.6, 0.7],
                                    lifeTime=0.5,
                                    textColorRGB=color_sum)

    def change_network_based_on_key(self):
        keypress = self.env.p.getKeyboardEvents()
        if 107 in keypress.keys() and keypress[107] == 1:  # K
            if self.current_network < self.num_networks - 1:
                self.current_network += 1
                time.sleep(0.1)
        elif 106 in keypress.keys() and keypress[106] == 1:  # J
            if self.current_network > 0:
                self.current_network -= 1
                time.sleep(0.1)

    def get_distance_error(self, observation):
        gripper = observation["additional_obs"]["endeff_xyz"]
        object = observation["actual_state"]
        goal = observation["goal_state"]
        object_goal_distance = self.task.calc_distance(object, goal)
        if self.current_network == 0:
            gripper_object_distance = self.task.calc_distance(gripper, object)
            final_distance = object_goal_distance + gripper_object_distance
        else:
            final_distance = object_goal_distance
        return final_distance

    def get_positions(self, observation):
        goal_position = observation["goal_state"]
        object_position = observation["actual_state"]
        # gripper_name = [x for x in self.env.task.obs_template["additional_obs"] if "endeff" in x][0]
        #gripper_position = self.env.robot.get_accurate_gripper_position()  # observation["additional_obs"][gripper_name][:3]
        gripper_position = observation["additional_obs"]["endeff_xyz"]
        gripper_states = self.env.robot.get_gjoints_states()

        if self.prev_object_position is None:
            self.prev_object_position = object_position
        if self.__class__.__name__ in ["AaGaM", "AaGaMaD", "AaGaMaDaW"]:
            goal_position[2] += self.above_offset
        return goal_position, object_position, gripper_position, gripper_states

    def check_endeff_z_penalty(self, gripper_position):
        """
        Check if end effector z position is below threshold and apply penalty
        Only applies when fixed base is false
        
        Parameters:
            :param gripper_position: (array) End effector position [x, y, z]
        Returns:
            :return penalty: (float) Penalty value (0 if no penalty, negative if penalty applied)
        """
        penalty = 0
        # Only penalize when fixed base is false
        if not self.env.robot.use_fixed_base:
            # Check if end effector z position is lower than 0.01
            if gripper_position[2] < 0.01:
                penalty = -1.0  # Apply penalty
        return penalty

    #### PROTOREWARDS DEFINITIONS  ####

    def approach_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(False)
        #self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        dist = self.task.calc_distance(gripper[:3], object[:3])        
        gripdist = sum(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) + ((gripdist - self.last_grip_dist) * 0.2)
        # self.env.p.addUserDebugText(f"Reward:{reward}", [0.63, 0.8, 0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        # self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65, 0.6, 0.7], lifeTime=0.5,
        #                            textColorRGB=[0, 0, 125])
        self.reward_name = "approach"
        return reward

    def withdraw_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(False)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        if dist >= self.withdraw_threshold: #rewarding only up to distance which should finish the task
            reward = 0 #This is done so that robot doesnt learn to drop object out of goal and then get the biggest reward by
            #withdrawing without finishing the task
        else:
            reward = (dist - self.last_approach_dist) + ((gripdist - self.last_grip_dist) * 0.2)
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "withdraw"
        return reward

    def grasp_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(True)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) * 0.2 + ((self.last_grip_dist - gripdist) * 10)
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "grasp"
        return reward

    def drop_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(True)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        gripdist = sum(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward = (self.last_approach_dist - dist) * 0.2 + ((gripdist - self.last_grip_dist) * 10)
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "drop"
        return reward

    def move_compute(self, object, goal, gripper_states):
        self.env.robot.set_magnetization(False)
        object_XY = object[:3]
        goal_XY = goal[:3]
        gripdist = sum(gripper_states)
        dist = self.task.calc_distance(object_XY, goal_XY)
        if self.last_move_dist is None:
            self.last_move_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        distance_rew = (self.last_move_dist - dist)
        gripper_rew = (self.last_grip_dist - gripdist) * 0.1
        reward = distance_rew + gripper_rew
        self.last_move_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "move"
        return reward

    def rotate_compute(self, object, goal, gripper_states):
        self.env.robot.set_magnetization(False)
        dist = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist
        gripdist = sum(gripper_states)
        gripper_rew = (self.last_grip_dist - gripdist) * 0.1
        reward = self.last_place_dist - dist
        rot = self.task.calc_rot_quat(object, goal)
        if self.last_rot_dist is None:
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        reward = reward + rewardrot
        self.last_place_dist = dist
        self.last_rot_dist = rot
        self.network_rewards[self.current_network] += reward
        self.reward_name = "rotate"
        return reward

    def transform_compute(self, object, goal, trajectory, magnetization=True):
        """Calculate reward based on following a trajectory
        params: object: self-explanatory
                goal: self-explanatory
                trajectory: (np.array) 3D trajectory, lists of points x, y, z
                magnetization: (boolean) sets magnetization on or off
        Reward is calculated based on distance of object from goal and square distance of object from trajectory.
        That way, object tries to approach goal while trying to stay on trajectory path.
        """
        self.env.robot.set_magnetization(magnetization)
        dist_g = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist_g
        reward_g_dist = self.last_place_dist - dist_g  # distance from goal
        pos = object[:3]
        dist_t, self.last_traj_idx = self.task.trajectory_distance(trajectory, pos, self.last_traj_idx, 10)
        if self.last_traj_dist is None:
            self.last_traj_dist = dist_t
        reward_t_dist = self.last_traj_dist - dist_t  # distance from trajectory
        reward = reward_g_dist + 4 * reward_t_dist
        self.last_place_dist = dist_g
        self.last_traj_dist = dist_t
        self.network_rewards[self.current_network] += reward
        self.reward_name = "transform"
        return reward

    def follow_compute(self, object, goal, trajectory, magnetization=True):
        """Calculate reward based on following a trajectory
        params: object: self-explanatory
                goal: self-explanatory
                trajectory: (np.array) 3D trajectory, lists of points x, y, z
                magnetization: (boolean) sets magnetization on or off
        Reward is calculated based on distance of object from goal and square distance of object from trajectory.
        That way, object tries to approach goal while trying to stay on trajectory path.
        """
        self.env.robot.set_magnetization(magnetization)
        dist_g = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist_g
        reward_g_dist = self.last_place_dist - dist_g  # distance from goal
        pos = object[:3]
        dist_t, self.last_traj_idx = self.task.trajectory_distance(trajectory, pos, self.last_traj_idx, 10)
        if self.last_traj_dist is None:
            self.last_traj_dist = dist_t
        reward_t_dist = self.last_traj_dist - dist_t  # distance from trajectory
        reward = reward_g_dist + 4 * reward_t_dist
        self.last_place_dist = dist_g
        self.last_traj_dist = dist_t
        self.network_rewards[self.current_network] += reward
        self.reward_name = "transform"
        return reward

    # PREDICATES

    def gripper_approached_object(self, gripper, object):
        if self.task.calc_distance(gripper, object) <= self.approached_threshold:
            return True
        return False

    def gripper_withdraw_object(self, gripper, object):
        if self.task.calc_distance(gripper, object) >= self.withdraw_threshold:
            return True
        return False

    def gripper_opened(self, gripper_states):
        gripper_status = self.env.robot.check_gripper_status(gripper_states)
        if gripper_status == "open":
            self.env.robot.release_object(self.env.env_objects["actual_state"])
            self.env.robot.set_magnetization(False)
            return True
        return False

    def gripper_closed(self, gripper_states):
        gripper_status = self.env.robot.check_gripper_status(gripper_states)
        if gripper_status == "close":
            try:
                #self.env.robot.magnetize_object(self.env.env_objects["actual_state"])
                self.env.robot.set_magnetization(True)
                return True
            except:
                return True
        return False

    def object_near_goal(self, object, goal):
        goal[2] += self.above_offset
        distance = self.task.calc_distance(goal, object)
        if distance < self.near_threshold:
            return True
        return False


# ATOMIC ACTIONS - Examples of 1-5 protorewards

class A(Protorewards):

    def reset(self):
        super().reset()
        self.network_names = ["approach"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)
        target = [[object_position, goal_position, gripper_states]][owner]
        reward = [self.approach_compute][owner](*target)
        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)
        if self.env.episode_terminated:
            reward += 0.2 #Adding reward for succesful finish of episode
        self.last_owner = owner
        #self.disp_reward(reward, owner)
        self.rewards_history.append(reward)
        self.rewards_num = 1
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        if self.gripper_approached_object(object_position, goal_position):
            if self.gripper_opened(gripper_states):
                self.task.check_goal()
        self.task.check_episode_steps()
        return self.current_network


class AaG(Protorewards):

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)
        target = [[object_position, goal_position, gripper_states], [object_position, goal_position, gripper_states]][
            owner]
        reward = [self.approach_compute, self.grasp_compute][owner](*target)
        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)
        if self.env.episode_terminated:
            reward += 0.2 #Adding reward for succesful finish of episode
        #self.disp_reward(reward, owner)
        self.last_owner = owner
        self.rewards_history.append(reward)
        self.rewards_num = 2
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        else:
            if self.gripper_approached_object(object_position, goal_position):
                if self.gripper_opened(gripper_states):
                    self.current_network = 1
        if self.current_network == 1:
            if self.gripper_closed(gripper_states):
                self.task.check_goal()
        self.task.check_episode_steps()
        return self.current_network


class AaGaM(Protorewards):

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp", "move"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)
        target = \
        [[gripper_position, object_position, gripper_states], [gripper_position, object_position, gripper_states],
         [object_position, goal_position, gripper_states]][owner]
        reward = [self.approach_compute, self.grasp_compute, self.move_compute][owner](*target)
        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)
        if self.env.episode_terminated:
            reward += 0.2 #Adding reward for succesful finish of episode
        #self.disp_reward(reward, owner)
        self.last_owner = owner
        self.rewards_history.append(reward)
        self.rewards_num = 3
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        else:
            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
            if self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self.current_network = 2
        if self.current_network == 2:
            if self.object_near_goal(object_position, goal_position):
                self.task.check_goal()

        # Change after 100 and 250 steps:
        # if self.env.episode_steps == 100:
        #     print("changed network to 1")
        #     self.current_network = 1
        # elif self.env.episode_steps == 250:
        #     self.current_network = 2
        #     print("changed network to 2")

        # Random
        # self.current_network=np.random.randint(0, self.num_networks)
        self.task.check_episode_steps()
        return self.current_network

class AaGaR(Protorewards):
    """
    Reward sequence: Approach -> Grasp -> Rotate
    Uses the 'rotate_compute' protoreward for the third step.
    """
    def reset(self):
        super().reset()
        # Updated network names to include "rotate"
        self.network_names = ["approach", "grasp", "rotate"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        # Define targets for each protoreward function
        # Note: rotate_compute needs object_position and goal_position
        target = [
            [gripper_position, object_position, gripper_states],  # approach
            [gripper_position, object_position, gripper_states],  # grasp
            [object_position, goal_position, gripper_states]                      # rotate
        ][owner]

        # List of protoreward functions corresponding to network_names
        reward_func_list = [
            self.approach_compute,
            self.grasp_compute,
            self.rotate_compute  # Using rotate_compute here
        ]

        # Calculate reward using the function for the current network owner
        reward = reward_func_list[owner](*target)

        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)
        
        # Add bonus for successful episode termination
        if self.env.episode_terminated:
            reward += 0.2

        # Display, log, and update history
        #self.disp_reward(reward, owner)
        self.last_owner = owner
        self.rewards_history.append(reward)
        self.rewards_num = 3 # Total number of networks is 3
        return reward


    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        else:
            # Ground truth logic for switching networks based on predicates
            if self.current_network == 0: # approach
                # Switch to grasp if gripper is near object and open
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                    self.current_network = 1
            elif self.current_network == 1: # grasp
                # Switch to rotate if gripper is near object and closed (grasped)
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_closed(gripper_states):
                    self.current_network = 2
            # Network 2 (rotate) is the final stage

        # Check for task completion in the final network stage (rotate)
        if self.current_network == 2:
            # Check if the object is near the goal (rotation might also affect position)
            if self.object_near_goal(object_position, goal_position):
                 # Additionally, you might want to check rotation alignment if applicable
                 # if self.task.check_rotation_alignment(object_position, goal_position):
                 self.task.check_goal() # Check if the overall task goal is met

        # Check for episode step limits
        self.task.check_episode_steps()
        return self.current_network


class AaGaMaD(Protorewards):

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp", "move", "drop"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)
        target = [[gripper_position, object_position, gripper_states],
                  [gripper_position, object_position, gripper_states],
                  [object_position, goal_position, gripper_states],
                  [gripper_position, goal_position, gripper_states]][owner]
        reward = [self.approach_compute, self.grasp_compute, self.move_compute, self.drop_compute][owner](*target)
        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)
        if self.env.episode_terminated:
            reward += 0.2 #Adding reward for succesful finish of episode
        #self.disp_reward(reward, owner)
        self.last_owner = owner
        self.rewards_history.append(reward)
        self.rewards_num = 4
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        else:
            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
            if self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self.current_network = 2
            if self.current_network == 2:
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
        if self.current_network == 3:
            # if self.gripper_approached_object(gripper_position, object_position):
            if self.gripper_opened(gripper_states):
                self.task.check_goal()
        self.task.check_episode_steps()
        return self.current_network


class AaGaMaDaW(Protorewards):

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp", "move", "drop", "withdraw"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)
        target = [[gripper_position, object_position, gripper_states],
                  [gripper_position, object_position, gripper_states],
                  [object_position, goal_position, gripper_states],
                  [gripper_position, object_position, gripper_states],
                  [gripper_position, goal_position, gripper_states]][owner]
        reward = \
        [self.approach_compute, self.grasp_compute, self.move_compute, self.drop_compute, self.withdraw_compute][owner](
            *target)
        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)
        if self.env.episode_terminated:
            reward += 0.2 #Adding reward for succesful finish of episode
        #self.disp_reward(reward, owner)
        self.last_owner = owner
        self.rewards_history.append(reward)
        self.rewards_num = 5
        return reward


    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        else:
            if self.current_network == 0:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
            elif self.current_network == 1:
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self.current_network = 2
            elif self.current_network == 2:
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
            elif self.current_network == 3:
                # if self.gripper_approached_object(gripper_position, object_position):
                if self.gripper_opened(gripper_states):
                    self.current_network = 4
        if self.current_network == 4:
            if self.gripper_withdraw_object(gripper_position, object_position):
                if self.gripper_opened(gripper_states):
                    self.task.check_goal()

        self.task.check_episode_steps()
        # self.current_network = np.random.randint(0, self.num_networks)
        return self.current_network
    
class AaGaRaDaW(Protorewards):

    def reset(self):
        super().reset()
        # Updated network names to include "rotate" instead of "move"
        self.network_names = ["approach", "grasp", "rotate", "drop", "withdraw"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)
        # Updated target list for the new sequence of protorewards
        target = [[gripper_position, object_position, gripper_states],  # approach
                  [gripper_position, object_position, gripper_states],  # grasp
                  [object_position, goal_position, gripper_states],                     # rotate (takes object, goal)
                  [gripper_position, object_position, gripper_states],  # drop
                  [gripper_position, goal_position, gripper_states]][owner] # withdraw
        # Updated list of protoreward functions to call
        reward = \
        [self.approach_compute, self.grasp_compute, self.rotate_compute, self.drop_compute, self.withdraw_compute][owner](
            *target)
        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)
        if self.env.episode_terminated:
            reward += 0.2 #Adding reward for succesful finish of episode
        #self.disp_reward(reward, owner)
        self.last_owner = owner
        self.rewards_history.append(reward)
        self.rewards_num = 5 # Still 5 networks
        return reward


    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        else:
            # Logic for switching networks based on predicates
            if self.current_network == 0: # approach
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_opened(gripper_states):
                        self.current_network = 1
            elif self.current_network == 1: # grasp
                if self.gripper_approached_object(gripper_position, object_position):
                    if self.gripper_closed(gripper_states):
                        self.current_network = 2
            elif self.current_network == 2: # rotate
                # Switch from rotate to drop when the object is near the goal
                # (Rotation might also affect position, or this is a proxy)
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
            elif self.current_network == 3: # drop
                if self.gripper_approached_object(gripper_position, object_position): # Check approach to object again for dropping
                    if self.gripper_opened(gripper_states):
                        self.current_network = 4
            # Network 4 (withdraw) is the last step before checking goal

        # Check for task completion in the final network stage
        if self.current_network == 4: # withdraw
            if self.gripper_withdraw_object(gripper_position, object_position): # Check withdraw distance
                 if self.gripper_opened(gripper_states): # Ensure gripper is open after withdrawing
                    self.task.check_goal()

        self.task.check_episode_steps()
        return self.current_network

class AaGaFaDaW(Protorewards):
    """
    Reward sequence: Approach -> Grasp -> Follow Trajectory -> Drop -> Withdraw
    Uses the 'follow_compute' protoreward for the third step.
    """
    def reset(self):
        super().reset()
        # Updated network names to include "follow"
        self.network_names = ["approach", "grasp", "follow", "drop", "withdraw"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        # --- Get Trajectory for Follow ---
        # NOTE: Assumes the task object has a method or attribute to get the relevant trajectory
        # You might need to adjust this based on your specific Task implementation
        try:
            # Example: Get trajectory from the task object
            trajectory = self.task.get_current_trajectory(observation)
            if trajectory is None:
                 print("Warning: get_current_trajectory returned None. Follow reward might be incorrect.")
                 # Provide a default or handle appropriately if trajectory isn't available
                 trajectory = np.array([]) # Example default
        except AttributeError:
            print("Warning: Task object does not have 'get_current_trajectory' method. Using empty trajectory for 'follow'.")
            trajectory = np.array([]) # Default empty trajectory if method doesn't exist
        except Exception as e:
            print(f"Warning: Error getting trajectory: {e}. Using empty trajectory.")
            trajectory = np.array([])

        # Define targets for each protoreward function
        # Note: follow_compute needs object_position, goal_position, and the trajectory
        # Note: drop_compute target uses object_position like in AaGaMaDaW
        target = [
            [gripper_position, object_position, gripper_states],  # approach
            [gripper_position, object_position, gripper_states],  # grasp
            [object_position, goal_position, trajectory],         # follow
            [gripper_position, object_position, gripper_states],  # drop
            [gripper_position, goal_position, gripper_states]   # withdraw (away from goal)
        ][owner]

        # List of protoreward functions corresponding to network_names
        reward_func_list = [
            self.approach_compute,
            self.grasp_compute,
            self.follow_compute,  # Using follow_compute here
            self.drop_compute,
            self.withdraw_compute
        ]

        # Calculate reward using the function for the current network owner
        reward = reward_func_list[owner](*target)

        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)

        # Add bonus for successful episode termination
        if self.env.episode_terminated:
            reward += 0.2

        # Display, log, and update history
        #self.disp_reward(reward, owner)
        self.last_owner = owner
        self.rewards_history.append(reward)
        self.rewards_num = 5 # Total number of networks remains 5
        return reward


    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        else:
            # Ground truth logic for switching networks based on predicates
            if self.current_network == 0: # approach
                # Switch to grasp if gripper is near object and open
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                    self.current_network = 1
            elif self.current_network == 1: # grasp
                # Switch to follow if gripper is near object and closed (grasped)
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_closed(gripper_states):
                    self.current_network = 2
            elif self.current_network == 2: # follow
                # Switch to drop when the object following the trajectory is near the goal
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
            elif self.current_network == 3: # drop
                # Switch to withdraw if gripper is near the object (or goal) and opened
                # Using object position check consistent with AaGaMaDaW's drop->withdraw transition
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                    self.current_network = 4
            # Network 4 (withdraw) is the final stage before checking goal completion

        # Check for task completion in the final network stage (withdraw)
        if self.current_network == 4:
            # Check if gripper has withdrawn sufficiently from the object and is open
            if self.gripper_withdraw_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                self.task.check_goal() # Check if the overall task goal is met

        # Check for episode step limits
        self.task.check_episode_steps()
        return self.current_network

class AaGaTaDaW(Protorewards):
    """
    Reward sequence: Approach -> Grasp -> Transform Trajectory -> Drop -> Withdraw
    Uses the 'transform_compute' protoreward for the third step.
    """
    def reset(self):
        super().reset()
        # Updated network names to include "transform"
        self.network_names = ["approach", "grasp", "transform", "drop", "withdraw"]

    def compute(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)
        owner = self.decide(observation)

        # --- Get Trajectory for Transform ---
        # NOTE: Assumes the task object has a method or attribute to get the relevant trajectory
        # This logic is similar to AaGaFaDaW, adjust if needed for transform's specific trajectory requirements.
        try:
            # Example: Get trajectory from the task object
            trajectory = self.task.get_current_trajectory(observation)
            if trajectory is None:
                 print("Warning: get_current_trajectory returned None. Transform reward might be incorrect.")
                 trajectory = np.array([]) # Example default
        except AttributeError:
            print("Warning: Task object does not have 'get_current_trajectory' method. Using empty trajectory for 'transform'.")
            trajectory = np.array([]) # Default empty trajectory
        except Exception as e:
            print(f"Warning: Error getting trajectory: {e}. Using empty trajectory.")
            trajectory = np.array([])

        # Define targets for each protoreward function
        # Note: transform_compute needs object_position, goal_position, and the trajectory
        target = [
            [gripper_position, object_position, gripper_states],  # approach
            [gripper_position, object_position, gripper_states],  # grasp
            [object_position, goal_position, trajectory],         # transform
            [gripper_position, object_position, gripper_states],  # drop
            [gripper_position, goal_position, gripper_states]   # withdraw (away from goal)
        ][owner]

        # List of protoreward functions corresponding to network_names
        reward_func_list = [
            self.approach_compute,
            self.grasp_compute,
            self.transform_compute, # Using transform_compute here
            self.drop_compute,
            self.withdraw_compute
        ]

        # Calculate reward using the function for the current network owner
        # Pass magnetization=True explicitly if needed by transform_compute, otherwise default is used
        if owner == 2: # transform network
             reward = reward_func_list[owner](*target) # Uses default magnetization=True
             # Or explicitly: reward = reward_func_list[owner](*target, magnetization=True)
        else:
             reward = reward_func_list[owner](*target)

        # Apply penalty for low end effector z position when fixed base is false
        reward += self.check_endeff_z_penalty(gripper_position)

        # Add bonus for successful episode termination
        if self.env.episode_terminated:
            reward += 0.2

        # Display, log, and update history
        #self.disp_reward(reward, owner)
        self.last_owner = owner
        self.rewards_history.append(reward)
        self.rewards_num = 5 # Total number of networks remains 5
        return reward


    def decide(self, observation=None):
        goal_position, object_position, gripper_position, gripper_states = self.get_positions(observation)

        if self.env.network_switcher == "keyboard":
            self.change_network_based_on_key()
        else:
            # Ground truth logic for switching networks based on predicates
            if self.current_network == 0: # approach
                # Switch to grasp if gripper is near object and open
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                    self.current_network = 1
            elif self.current_network == 1: # grasp
                # Switch to transform if gripper is near object and closed (grasped)
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_closed(gripper_states):
                    self.current_network = 2
            elif self.current_network == 2: # transform
                # Switch to drop when the object being transformed is near the goal
                if self.object_near_goal(object_position, goal_position):
                    self.current_network = 3
            elif self.current_network == 3: # drop
                # Switch to withdraw if gripper is near the object (or goal) and opened
                if self.gripper_approached_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                    self.current_network = 4
            # Network 4 (withdraw) is the final stage

        # Check for task completion in the final network stage (withdraw)
        if self.current_network == 4:
            # Check if gripper has withdrawn sufficiently from the object and is open
            if self.gripper_withdraw_object(gripper_position, object_position) and self.gripper_opened(gripper_states):
                self.task.check_goal() # Check if the overall task goal is met

        # Check for episode step limits
        self.task.check_episode_steps()
        return self.current_network
