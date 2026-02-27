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


class UniversalReward:
    """
    Universal reward calculator for task distance and gripper rewards.

    Computes reward based on actual and goal state (translation and rotation).
    Provides absolute, relative, and temporal reward components for both
    task distance and gripper, along with progress tracking.

    Parameters:
        :param task: (object) Task instance with calc_distance and calc_rot_quat methods
        :param robot: (object) Robot instance with check_gripper_status, close_gripper, open_gripper
        :param window_size: (int) Size of the sliding window for temporal reward calculation
        :param solved_threshold: (float) Progress percentage threshold to consider task solved (0-100)
    """

    def __init__(self, task, robot, window_size=10, solved_threshold=90.0):
        self.task = task
        self.robot = robot
        self.window_size = window_size
        self.solved_threshold = solved_threshold
        self.reset()

    def reset(self):
        """Reset all internal state for a new episode."""
        self.step = 0
        # Task distance tracking
        self.max_trans_dist = None
        self.min_trans_dist = None
        self.max_rot_dist = None
        self.min_rot_dist = None
        self.prev_trans_dist = None
        self.prev_rot_dist = None
        # Task reward history for temporal calculation
        self.absolute_reward_history = []
        self.relative_reward_history = []
        # Gripper tracking
        self.max_grip_dist = None
        self.min_grip_dist = None
        self.prev_grip_dist = None
        self.grip_status = None
        # Gripper reward history for temporal calculation
        self.grip_absolute_reward_history = []
        self.grip_relative_reward_history = []

    def _compute_absolute_reward(self, current, min_val, max_val):
        """
        Compute absolute reward rescaled to range [-1, 1].
        Returns: 1 for min distance, 0 for max distance, -1 when distance exceeds max by (max-min).
        Reward decreases linearly beyond max_val.
        """
        range_val = max_val - min_val
        if range_val <= 0:
            return 0.0
        normalized = (current - min_val) / range_val
        return 1.0 - normalized

    def _compute_relative_reward(self, prev_dist, current_dist):
        """
        Compute relative reward from difference between previous and current distance.
        Positive if distance decreases, negative if increases, zero if same.
        """
        return prev_dist - current_dist

    def _compute_temporal_reward(self, absolute_history, relative_history):
        """
        Compute temporal reward using sliding window mean of absolute and relative rewards.
        Returns negative reward when mean is negative, positive when positive.
        """
        window = self.window_size
        abs_window = absolute_history[-window:]
        rel_window = relative_history[-window:]
        if len(abs_window) == 0:
            return 0.0
        abs_mean = np.mean(abs_window)
        rel_mean = np.mean(rel_window)
        return (abs_mean + rel_mean) / 2.0

    def _compute_progress(self, current_dist, max_dist):
        """
        Compute progress percentage (0-100).
        Returns (progress, solved) where solved is True when progress >= solved_threshold.
        """
        if max_dist <= 0:
            return 100.0, True
        progress = max(0.0, min(100.0, (1.0 - current_dist / max_dist) * 100.0))
        solved = progress >= self.solved_threshold
        return progress, solved

    def compute(self, actual_state, goal_state, gripper_states, rot=True, gripper="Close"):
        """
        Compute universal reward for the current step.

        Parameters:
            :param actual_state: (array) Actual object state [x, y, z, qx, qy, qz, qw]
            :param goal_state: (array) Goal state [x, y, z, qx, qy, qz, qw]
            :param gripper_states: (list) Current gripper joint values
            :param rot: (bool) If True, rotational error is included in all task rewards.
                If False, only translational error is calculated.
            :param gripper: (str) "Open" or "Close". If "Open", gripper reward increases
                when gripper reaches maximal values, and progress/solved thresholds
                are based on maximal values. If "Close", maximal reward is for
                minimal values, and progress/solved thresholds are based on minimal values.
        Returns:
            :return result: (dict) Dictionary containing:
                - task_absolute_reward: Rescaled task distance reward (0=max dist, 1=min dist)
                - task_relative_reward: Difference-based task reward
                - task_temporal_reward: Sliding window temporal task reward
                - task_progress: Task progress percentage (0-100)
                - task_solved: Boolean, True when task progress >= 90%
                - gripper_absolute_reward: Rescaled gripper reward (0=max dist, 1=min dist)
                - gripper_relative_reward: Difference-based gripper reward
                - gripper_temporal_reward: Sliding window temporal gripper reward
                - gripper_progress: Gripper progress percentage (0-100)
                - gripper_solved: Boolean, True when gripper progress >= 90%
                - total_reward: Combined reward from all components
        """
        #if gripper not in ("Open", "Close"):
        #    raise ValueError(f"gripper must be 'Open' or 'Close', got '{gripper}'")

        # -- Task distance (translation + rotation) --
        trans_dist = self.task.calc_distance(actual_state, goal_state)
        rot_dist = self.task.calc_rot_quat(actual_state, goal_state) if rot else 0.0

        # -- Gripper distance --
        status, grip_dist = self.robot.check_gripper_status(gripper_states)

        if self.step == 0:
            # First step: record max and min, absolute reward = 0
            self.max_trans_dist = trans_dist
            self.min_trans_dist = self.task.calc_distance(goal_state, goal_state)
            self.max_rot_dist = rot_dist
            self.min_rot_dist = self.task.calc_rot_quat(goal_state, goal_state) if rot else 0.0
            self.prev_trans_dist = trans_dist
            self.prev_rot_dist = rot_dist

            self.max_grip_dist = 1
            self.min_grip_dist = 0
            self.prev_grip_dist = grip_dist
            self.grip_status = status

            self.step += 1

            result = {
                "task_absolute_reward": 0.0,
                "task_relative_reward": 0.0,
                "task_temporal_reward": 0.0,
                "task_progress": 0.0,
                "task_solved": False,
                "gripper_absolute_reward": 0.0,
                "gripper_relative_reward": 0.0,
                "gripper_temporal_reward": 0.0,
                "gripper_progress": 0.0,
                "gripper_solved": False,
                "total_reward": 0.0,
            }
            return result

        # Update min/max distances
        #self.min_trans_dist = min(self.min_trans_dist, trans_dist)
        #self.min_rot_dist = min(self.min_rot_dist, rot_dist)
        #self.min_grip_dist = min(self.min_grip_dist, grip_dist)
        #self.max_grip_dist = max(self.max_grip_dist, grip_dist)

        # -- Task absolute reward --
        task_abs_trans = self._compute_absolute_reward(trans_dist, self.min_trans_dist, self.max_trans_dist)
        if rot:
            task_abs_rot = self._compute_absolute_reward(rot_dist, self.min_rot_dist, self.max_rot_dist)
            task_absolute_reward = (task_abs_trans + task_abs_rot) / 2.0
        else:
            task_absolute_reward = task_abs_trans

        # -- Task relative reward --
        rel_trans = self._compute_relative_reward(self.prev_trans_dist, trans_dist)
        if rot:
            rel_rot = self._compute_relative_reward(self.prev_rot_dist, rot_dist)
            task_relative_reward = rel_trans + rel_rot
        else:
            task_relative_reward = rel_trans

        # Log for temporal
        self.absolute_reward_history.append(task_absolute_reward)
        self.relative_reward_history.append(task_relative_reward)

        # -- Task temporal reward --
        task_temporal_reward = self._compute_temporal_reward(
            self.absolute_reward_history, self.relative_reward_history
        )

        # -- Task progress --
        trans_progress, _ = self._compute_progress(trans_dist, self.max_trans_dist)
        if rot:
            rot_progress, _ = self._compute_progress(rot_dist, self.max_rot_dist)
            task_progress = (trans_progress + rot_progress) / 2.0
        else:
            task_progress = trans_progress
        task_solved = task_progress >= self.solved_threshold

        # -- Gripper rewards (direction depends on gripper mode) --
        # Compute base values using helper methods (Close behavior)
        gripper_absolute_reward = self._compute_absolute_reward(grip_dist, self.min_grip_dist, self.max_grip_dist)
        # Rescale from [0, 1] to [-1, 1]: new = old * 2 - 1
        gripper_absolute_reward = gripper_absolute_reward * 2.0 - 1.0
        gripper_relative_reward = self._compute_relative_reward(self.prev_grip_dist, grip_dist)
        gripper_progress, gripper_solved = self._compute_progress(grip_dist, self.max_grip_dist)

        if gripper == "Open":
            # Invert rewards for opening behavior
            gripper_absolute_reward = -gripper_absolute_reward
            gripper_relative_reward = -gripper_relative_reward
            # Invert progress to measure openness instead of closeness
            gripper_progress = 100.0 - gripper_progress
            gripper_solved = gripper_progress >= self.solved_threshold

        # Log for temporal
        self.grip_absolute_reward_history.append(gripper_absolute_reward)
        self.grip_relative_reward_history.append(gripper_relative_reward)

        # -- Gripper temporal reward --
        gripper_temporal_reward = self._compute_temporal_reward(
            self.grip_absolute_reward_history, self.grip_relative_reward_history
        )

        # Update previous distances for next step
        self.prev_trans_dist = trans_dist
        self.prev_rot_dist = rot_dist
        self.prev_grip_dist = grip_dist
        self.step += 1

        total_reward = (task_absolute_reward + task_relative_reward + task_temporal_reward +
                        gripper_absolute_reward + gripper_relative_reward + gripper_temporal_reward)

        result = {
            "task_absolute_reward": task_absolute_reward,
            "task_relative_reward": task_relative_reward,
            "task_temporal_reward": task_temporal_reward,
            "task_progress": task_progress,
            "task_solved": task_solved,
            "gripper_absolute_reward": gripper_absolute_reward,
            "gripper_relative_reward": gripper_relative_reward,
            "gripper_temporal_reward": gripper_temporal_reward,
            "gripper_progress": gripper_progress,
            "gripper_solved": gripper_solved,
            "total_reward": total_reward,
        }
        return result


# PROTOREWARDS

class Protorewards(Reward):

    def reset(self):
        self.last_owner = None
        self.last_find_dist = None
        self.last_approach_dist = None
        self.last_grip_dist = None
        self.last_lift_dist = None
        self.first_move_grip_dist = None
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
        # Thresholds for end effector z position checking
        self.endeff_z_threshold = 0.01
        self.endeff_z_penalty = -1.0
        # Grasp distance tracking variables
        self.ideal_grasp_dist = None
        self.grasp_within_threshold_count = 0
        # Universal reward calculator
        self.universal_reward = UniversalReward(self.task, self.env.robot)

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
            # Check if end effector z position is lower than threshold
            if gripper_position[2] < self.endeff_z_threshold:
                penalty = self.endeff_z_penalty
        return penalty


    def test_compute(self, object_state, goal_state, gripper_states):
        """
        Compute universal reward with rot=False and gripper="Open".

        Parameters:
            :param object_state: (array) Actual object state [x, y, z, qx, qy, qz, qw]
            :param goal_state: (array) Goal state [x, y, z, qx, qy, qz, qw]
            :param gripper_states: (list) Current gripper joint values
        Returns:
            :return result: (dict) Dictionary with all reward components, progress, and solved status
        """
        result = self.universal_reward.compute(object_state, goal_state, gripper_states, rot=False, gripper="Open")
        reward = result["total_reward"]
        self.network_rewards[self.current_network] += reward
        self.reward_name = "test_compute"
        return result

    #### PROTOREWARDS DEFINITIONS  ####

    def approach_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(False)
        #self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        dist = self.task.calc_distance(gripper[:3], object[:3])        
        status, gripdist = self.env.robot.check_gripper_status(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        # Approach: reward opening (positive when metric increases toward 1
        reward = (self.last_approach_dist - dist) + (gripdist - self.last_grip_dist)
        #print(f"Approach reward: {reward:.5f}, Distance: {(self.last_approach_dist - dist):.5f}, Gripdist: {(gripdist - self.last_grip_dist):.5f}, Status: {status}", end='\r')
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
        _, gripdist = self.env.robot.check_gripper_status(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        if dist >= self.withdraw_threshold: #rewarding only up to distance which should finish the task
            reward = 0 #This is done so that robot doesnt learn to drop object out of goal and then get the biggest reward by
            #withdrawing without finishing the task
        else:
            # Withdraw: reward opening (positive when metric increases toward 1)
            reward = (dist - self.last_approach_dist) + (gripdist - self.last_grip_dist)
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "withdraw"
        return reward

    def grasp_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(True)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        _, gripdist = self.env.robot.check_gripper_status(gripper_states)
        
        # Initialize ideal distance on first frame
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
            self.ideal_grasp_dist = dist  # Store initial distance as ideal
            self.grasp_within_threshold_count = 0
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        
        # Calculate distance reward based on deviation from ideal
        deviation = abs(dist - self.ideal_grasp_dist)
        threshold = 0.01
        
        if deviation > threshold:
            # Penalize deviation beyond threshold
            dist_reward = -(deviation - threshold)
        else:
            # Reward staying within threshold, accumulate over time
            self.grasp_within_threshold_count += 1
            dist_reward = 0.01 * self.grasp_within_threshold_count
        
        # Grasp: reward closing (positive when metric decreases toward 0)
        grip_reward = (self.last_grip_dist - gripdist)
        reward = dist_reward + grip_reward
        
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "grasp"
        return reward

    def drop_compute(self, gripper, object, gripper_states):
        self.env.robot.set_magnetization(True)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        _, gripdist = self.env.robot.check_gripper_status(gripper_states)
        if self.last_approach_dist is None:
            self.last_approach_dist = dist
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        # Drop: reward opening (positive when metric increases toward 1)
        reward = (self.last_approach_dist - dist) + (gripdist - self.last_grip_dist)
        self.last_approach_dist = dist
        self.last_grip_dist = gripdist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "drop"
        return reward

    def move_compute(self, object, goal, gripper_states):
        self.env.robot.set_magnetization(True)
        _, gripdist = self.env.robot.check_gripper_status(gripper_states)
        dist = self.task.calc_distance(object[:3], goal[:3])
        
        if self.last_move_dist is None:
            self.last_move_dist = dist
        if self.first_move_grip_dist is None:
            self.first_move_grip_dist = gripdist
        # Reward for moving object closer to goal, penalize deviation from initial gripper distance
        reward = (self.last_move_dist - dist) - abs(gripdist - self.first_move_grip_dist)
        self.last_move_dist = dist
        self.network_rewards[self.current_network] += reward
        self.reward_name = "move"
        return reward

    def rotate_compute(self, object, goal, gripper_states):
        self.env.robot.set_magnetization(False)
        dist = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist
        _, gripdist = self.env.robot.check_gripper_status(gripper_states)
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        # Rotate: reward keeping closed (positive when metric stays low/decreases toward 0)
        gripper_rew = self.last_grip_dist - gripdist
        reward = self.last_place_dist - dist
        rot = self.task.calc_rot_quat(object, goal)
        if self.last_rot_dist is None:
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        reward = reward + rewardrot + gripper_rew
        self.last_place_dist = dist
        self.last_rot_dist = rot
        self.last_grip_dist = gripdist
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
        _, gripdist = self.env.robot.check_gripper_status(gripper_states)
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward_g_dist = self.last_grip_dist - gripdist  # distance from goal
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
        _, gripdist = self.env.robot.check_gripper_status(gripper_states)
        if self.last_grip_dist is None:
            self.last_grip_dist = gripdist
        reward_g_dist = self.last_grip_dist - gripdist  # distance from goal
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
        gripper_status,metric = self.env.robot.check_gripper_status(gripper_states)
        if gripper_status == "open":
            self.env.robot.release_object(self.env.env_objects["actual_state"])
            self.env.robot.set_magnetization(False)
            return True
        return False

    def gripper_closed(self, gripper_states):
        gripper_status,metric = self.env.robot.check_gripper_status(gripper_states)
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
    
    def protoreward_params (self, name):
        if name == "approach":
            return {"rot": False, "gripper": "Open"}
        elif name == "withdraw":
            return {"rot": False, "gripper": "Open"}
        elif name == "grasp":
            return {"rot": True, "gripper": "Close"}
        elif name == "drop":
            return {"rot": True, "gripper": "Open"}
        elif name == "move":
            return {"rot": False, "gripper": "Close"}
        elif name == "rotate":
            return {"rot": True, "gripper": "Close"}
        elif name == "transform":
            return {"rot": False, "gripper": "Close"}
        elif name == "follow":
            return {"rot": False, "gripper": "Close"}
        else:
            raise ValueError(f"Unknown protoreward name: {name}")


# ATOMIC ACTIONS - Examples of 1-5 protorewards

####NEW

class AGM(Protorewards):

    def __init__(self, env, task=None):
        super().__init__(env, task)
        self.reset()

    def reset(self):
        super().reset()
        self.network_names = ["approach", "grasp", "move"]
        self.owner = 0

    def compute(self, observation=None):
        params = self.protoreward_params(self.network_names[self.owner])
        result = self.universal_reward.compute(observation["actual_state"], observation["goal_state"], self.env.robot.get_gjoints_states(), **params)
        reward = result["total_reward"]
        
        # Check if task is solved and progress to next network
        if result["task_solved"] and self.owner < len(self.network_names) - 1:
            self.owner += 1
        
        self.current_network = self.owner
        self.network_rewards[self.current_network] += reward
        self.last_owner = self.owner
        self.rewards_history.append(reward)
        self.rewards_num = 3
        return reward



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
