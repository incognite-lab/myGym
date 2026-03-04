import numpy as np
import matplotlib.pyplot as plt
# from stable_baselines import results_plotter
import os
import math
from math import sqrt, fabs, exp, pi, asin
from myGym.utils.vector import Vector
import random
import time

class UniversalReward:
    """
    Universal reward calculator for task distance and gripper rewards.

    Computes reward based on actual and goal state (translation and rotation).
    Provides absolute, relative, and temporal reward components for both
    task distance and gripper, along with progress tracking.

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task instance with calc_distance and calc_rot_quat methods
        :param window_size: (int) Size of the sliding window for temporal reward calculation
        :param solved_threshold: (float) Progress percentage threshold to consider task solved (0-100)
    """

    def __init__(self, env, task=None, window_size=10, solved_threshold=90.0):
        # Initialize only UniversalReward-specific attributes
        self.env = env
        self.task = task
        self.window_size = window_size
        self.solved_threshold = solved_threshold

    def reset(self):
        """Empty reset - Rewarder will handle all reset logic."""
        pass

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

    def compute(self, observation=None):
        """Default compute method that calls calculate with default parameters."""
        raise NotImplementedError("Subclasses should override compute() method")

    def calculate(self, observation, rot=True, gripper="Close"):
        """
        Calculate universal reward for the current step.

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
                - arm_absolute_reward: Rescaled task distance reward (0=max dist, 1=min dist)
                - arm_relative_reward: Difference-based task reward
                - arm_temporal_reward: Sliding window temporal task reward
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
        trans_dist = self.task.calc_distance(observation["actual_state"], observation["goal_state"])
        rot_dist = self.task.calc_rot_quat(observation["actual_state"], observation["goal_state"]) if rot else 0.0
        
        # -- Absolute non-normalized distance --
        absolute_distance = trans_dist + rot_dist if rot else trans_dist

        # -- Gripper distance --
        status, grip_dist = self.env.robot.check_gripper_status(observation["additional_obs"]["gjoints_states"])

        if self.step == 0:
            # First step: record max and min, absolute reward = 0
            self.max_trans_dist = trans_dist
            self.min_trans_dist = self.task.calc_distance(observation["goal_state"], observation["goal_state"])
            self.max_rot_dist = rot_dist
            self.min_rot_dist = self.task.calc_rot_quat(observation["goal_state"], observation["goal_state"]) if rot else 0.0
            self.prev_trans_dist = trans_dist
            self.prev_rot_dist = rot_dist

            self.max_grip_dist = 1
            self.min_grip_dist = 0
            self.prev_grip_dist = grip_dist
            self.grip_status = status

            self.step += 1

            result = {
                "arm_absolute_reward": 0.0,
                "arm_relative_reward": 0.0,
                "arm_temporal_reward": 0.0,
                "task_progress": 0.0,
                "task_solved": False,
                "gripper_absolute_reward": 0.0,
                "gripper_relative_reward": 0.0,
                "gripper_temporal_reward": 0.0,
                "gripper_progress": 0.0,
                "gripper_solved": False,
                "total_reward": 0.0,
                "absolute_distance": absolute_distance,
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
            arm_absolute_reward = (task_abs_trans + task_abs_rot) / 2.0
        else:
            arm_absolute_reward = task_abs_trans

        # -- Task relative reward --
        rel_trans = self._compute_relative_reward(self.prev_trans_dist, trans_dist)
        if rot:
            rel_rot = self._compute_relative_reward(self.prev_rot_dist, rot_dist)
            arm_relative_reward = rel_trans + rel_rot
        else:
            arm_relative_reward = rel_trans

        # Log for temporal
        self.absolute_reward_history.append(arm_absolute_reward)
        self.relative_reward_history.append(arm_relative_reward)

        # -- Task temporal reward --
        arm_temporal_reward = self._compute_temporal_reward(
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

        total_reward = (arm_absolute_reward + arm_relative_reward + arm_temporal_reward +
                        gripper_absolute_reward + gripper_relative_reward + gripper_temporal_reward)

        result = {
            "arm_absolute_reward": arm_absolute_reward,
            "arm_relative_reward": arm_relative_reward,
            "arm_temporal_reward": arm_temporal_reward,
            "task_progress": task_progress,
            "task_solved": task_solved,
            "gripper_absolute_reward": gripper_absolute_reward,
            "gripper_relative_reward": gripper_relative_reward,
            "gripper_temporal_reward": gripper_temporal_reward,
            "gripper_progress": gripper_progress,
            "gripper_solved": gripper_solved,
            "total_reward": total_reward,
            "absolute_distance": absolute_distance,
        }
        return result

class Rewarder(UniversalReward):
    """
    Universal reward class that dynamically adapts to any task type.
    Initialized with task_subgoals which become the network names.
    Inherits from UniversalReward.
    """

    def __init__(self, env, task=None):
        # Call parent init first
        super().__init__(env, task)
        # Initialize Rewarder-specific attributes
        self.task_subgoals = task.get_subgoals_from_task_type() if task else []
        self.network_names = self.task_subgoals
        self.num_networks = len(self.network_names)
        self.current_network = 0
        self.owner = 0
        self.last_result = self._default_result()
        self.prev_owner = None
        self.last_owner = None
        self.rewards_history = []
        self.network_rewards = [0] * self.num_networks

    def _default_result(self):
        """Return a default result dictionary with all expected keys."""
        return {
            "arm_absolute_reward": 0.0,
            "arm_relative_reward": 0.0,
            "arm_temporal_reward": 0.0,
            "task_progress": 0.0,
            "task_solved": False,
            "gripper_absolute_reward": 0.0,
            "gripper_relative_reward": 0.0,
            "gripper_temporal_reward": 0.0,
            "gripper_progress": 0.0,
            "gripper_solved": False,
            "total_reward": 0.0,
            "absolute_distance": 0.0,
        }

    def reset(self):
        """Reset all state for both UniversalReward and Rewarder."""
        # Reset UniversalReward state variables
        self.step = 0
        self.max_trans_dist = None
        self.min_trans_dist = None
        self.max_rot_dist = None
        self.min_rot_dist = None
        self.prev_trans_dist = None
        self.prev_rot_dist = None
        self.absolute_reward_history = []
        self.relative_reward_history = []
        self.max_grip_dist = None
        self.min_grip_dist = None
        self.prev_grip_dist = None
        self.grip_status = None
        self.grip_absolute_reward_history = []
        self.grip_relative_reward_history = []
        # Reset Rewarder-specific state variables
        self.owner = 0
        self.current_network = 0
        print(f"Resetting Rewarder: starting with network {self.owner} ({self.network_names[self.owner] if self.network_names else 'N/A'})")
        self.last_owner = None
        self.last_result = self._default_result()
        self.prev_owner = None
        self.network_rewards = [0] * self.num_networks

    def compute(self, observation=None):
        if not self.network_names:
            return 0.0
        
        params = self.protoreward_params(self.network_names[self.owner])
        result = self.calculate(observation, **params)
        self.last_result = result
        reward = result["total_reward"]

        self.prev_owner = self.last_owner

        # Check if task is solved and progress to next network
        if result["task_solved"] and self.owner < len(self.network_names) - 1:
            self.owner += 1
            print(f"Switching to network {self.owner} ({self.network_names[self.owner]})")

        self.current_network = self.owner
        self.network_rewards[self.current_network] += reward
        self.last_owner = self.owner
        self.rewards_history.append(reward)
        return reward

    def protoreward_params(self, name):
        if name == "approach" or name == "A":
            return {"rot": False, "gripper": "Open"}
        elif name == "withdraw" or name == "W":
            return {"rot": False, "gripper": "Open"}
        elif name == "grasp" or name == "G":
            return {"rot": True, "gripper": "Close"}
        elif name == "drop" or name == "D":
            return {"rot": True, "gripper": "Open"}
        elif name == "move" or name == "M":
            return {"rot": False, "gripper": "Close"}
        elif name == "rotate" or name == "R":
            return {"rot": True, "gripper": "Close"}
        elif name == "transform" or name == "T":
            return {"rot": False, "gripper": "Close"}
        elif name == "follow" or name == "F":
            return {"rot": False, "gripper": "Close"}
        else:
            raise ValueError(f"Unknown protoreward name: {name}")
