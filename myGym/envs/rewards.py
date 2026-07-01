import numpy as np
import matplotlib.pyplot as plt
# from stable_baselines import results_plotter
import os
import math
from math import sqrt, fabs, exp, pi, asin
from myGym.utils.vector import Vector
import random
import time
import json
from pyquaternion import Quaternion  
from myGym.envs.predicates import GoalPredicateResolver, SubgoalPredicateResolver

class UniversalReward:
    """
    Universal reward calculator for task distance and gripper rewards.

    Computes reward based on actual and goal state (translation and rotation).
    Provides absolute, relative, and temporal reward components for both
    arm distance and gripper, along with progress tracking.

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task instance with calc_distance and calc_rot_quat methods
        :param window_size: (int) Size of the sliding window for temporal reward calculation
        :param solved_threshold: (float) Progress percentage threshold to consider task solved (0-100)
    """

    def __init__(self, env, task=None, window_size=10, solved_threshold=85.0):
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
        #print(f"Current distance: {current_dist:.4f}, Max distance: {max_dist:.4f}, Progress: {progress:.1f}%")
        solved = progress >= self.solved_threshold
        return progress, solved

    def compute(self, observation=None):
        """Default compute method that calls calculate with default parameters."""
        raise NotImplementedError("Subclasses should override compute() method")

    def calculate(self, observation, rot=True, gripper="close", actual_state=None, goal_state=None, 
                  armweight=1, gripperweight=1, absoluteweight=1, relativeweight=1, temporalweight=1):
        """
        Calculate universal reward for the current step.

        Parameters:
            :param observation: (dict) Observation dictionary from environment
            :param rot: (bool) If True, rotational error is included in all task rewards.
                If False, only translational error is calculated.
            :param gripper: (str) "open" or "close". If "open", gripper reward increases
                when gripper reaches maximal values, and progress/solved thresholds
                are based on maximal values. If "Close", maximal reward is for
                minimal values, and progress/solved thresholds are based on minimal values.
            :param actual_state: (list) Path to actual_state in observation dict (e.g., ["actual_state"] or ["additional_obs", "endeff_6D"])
            :param goal_state: (list) Path to goal_state in observation dict (e.g., ["goal_state"] or ["actual_state"])
            :param armweight: (float) Weight multiplier for all arm rewards (default: 1)
            :param gripperweight: (float) Weight multiplier for all gripper rewards (default: 1)
            :param absoluteweight: (float) Weight multiplier for all absolute rewards (default: 1)
            :param relativeweight: (float) Weight multiplier for all relative rewards (default: 1)
            :param temporalweight: (float) Weight multiplier for all temporal rewards (default: 1)
        Returns:
            :return result: (dict) Dictionary containing:
                - arm_absolute_reward: Rescaled arm distance reward (0=max dist, 1=min dist)
                - arm_relative_reward: Difference-based arm reward
                - arm_temporal_reward: Sliding window temporal arm reward
                - arm_progress: Task progress percentage (0-100)
                - arm_solved: Boolean, True when arm progress >= 90%
                - gripper_absolute_reward: Rescaled gripper reward (0=max dist, 1=min dist)
                - gripper_relative_reward: Difference-based gripper reward
                - gripper_temporal_reward: Sliding window temporal gripper reward
                - gripper_progress: Gripper progress percentage (0-100)
                - gripper_solved: Boolean, True when gripper progress >= 90%
                - total_reward: Combined reward from all components
        """
        #if gripper not in ("Open", "Close"):
        #    raise ValueError(f"gripper must be 'Open' or 'Close', got '{gripper}'")

        # Get actual_state and goal_state from observation using provided paths
        if actual_state is None:
            actual_state = ["actual_state"]
        if goal_state is None:
            goal_state = ["goal_state"]
        
        # Navigate observation dict using the paths with error handling
        actual_state_value = observation
        for key in actual_state:
            if isinstance(actual_state_value, dict) and key in actual_state_value:
                actual_state_value = actual_state_value[key]
            else:
                raise KeyError(f"Key '{key}' not found in observation path {actual_state}. Available keys: {list(actual_state_value.keys()) if isinstance(actual_state_value, dict) else 'not a dict'}")
        
        goal_state_value = observation
        for key in goal_state:
            if isinstance(goal_state_value, dict) and key in goal_state_value:
                goal_state_value = goal_state_value[key]
            else:
                raise KeyError(f"Key '{key}' not found in observation path {goal_state}. Available keys: {list(goal_state_value.keys()) if isinstance(goal_state_value, dict) else 'not a dict'}")

        # -- Task distance (translation + rotation) --
        trans_dist = self.task.calc_distance(actual_state_value, goal_state_value)
        rot_dist = self.task.calc_rot_quat(actual_state_value, goal_state_value) if rot else 0.0
        
        # -- Absolute non-normalized distance --
        absolute_distance = trans_dist + rot_dist if rot else trans_dist

        # -- Gripper distance --
        status, grip_dist = self.env.robot.check_gripper_status(observation["additional_obs"]["gjoints_angles"])

        # -- Task absolute reward --
        arm_abs_trans = self._compute_absolute_reward(trans_dist, self.min_trans_dist, self.max_trans_dist)
        if rot:
            arm_abs_rot = self._compute_absolute_reward(rot_dist, self.min_rot_dist, self.max_rot_dist)
            arm_absolute_reward = (arm_abs_trans + arm_abs_rot) / 2.0
        else:
            arm_absolute_reward = arm_abs_trans

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
            arm_progress = (trans_progress + rot_progress) / 2.0
        else:
            arm_progress = trans_progress
        arm_solved = arm_progress >= self.solved_threshold

        # -- Gripper rewards (direction depends on gripper mode) --
        # Compute base values using helper methods (Close behavior)
        gripper_absolute_reward = self._compute_absolute_reward(grip_dist, self.min_grip_dist, self.max_grip_dist)
        # Rescale from [0, 1] to [-1, 1]: new = old * 2 - 1
        gripper_absolute_reward = gripper_absolute_reward * 2.0 - 1.0
        gripper_relative_reward = self._compute_relative_reward(self.prev_grip_dist, grip_dist)
        gripper_progress, gripper_solved = self._compute_progress(grip_dist, self.max_grip_dist)

        if gripper == "open":
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


        #Calculate gripper orientation reward (fixed value - TODO: add as parameter)
        gripper_orientation_reward = 0.0  # Replace with actual orientation reward calculation
        #gripper_orientation_weight = 1  # Weight for orientation reward (can be adjusted)

        # Default gripper orientation (identity quaternion)
        default_gripper_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Current gripper orientation
        current_gripper_orientation = np.asarray(self.env.robot.get_orientation(), dtype=float)

        # Normalize quaternions
        q1 = default_gripper_orientation / np.linalg.norm(default_gripper_orientation)
        q2 = current_gripper_orientation / np.linalg.norm(current_gripper_orientation)

        # Smallest angular difference between orientations (0 to pi radians)
        angular_error = 2.0 * np.arccos(
            np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
        )

        # Reward: +1 (perfect alignment) to -1 (180° error)
        gripper_orientation_reward = 1.0 - 2.0 * (angular_error / np.pi)

        
        # Update previous distances for next step
        self.prev_trans_dist = trans_dist
        self.prev_rot_dist = rot_dist
        self.prev_grip_dist = grip_dist
        self.step += 1

        # Default gripper orientation (identity quaternion)
        default_gripper_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Current gripper orientation
        current_gripper_orientation = np.asarray(self.env.robot.get_orientation(), dtype=float)

        # Normalize quaternions
        q1 = default_gripper_orientation / np.linalg.norm(default_gripper_orientation)
        q2 = current_gripper_orientation / np.linalg.norm(current_gripper_orientation)

        # Convert quaternions to Euler angles (roll, pitch, yaw)
        # Using PyQuaternion for conversion
        q_default = Quaternion(q1[3], q1[0], q1[1], q1[2])  # (w, x, y, z)
        q_current = Quaternion(q2[3], q2[0], q2[1], q2[2])

        roll_default, _, _ = q_default.yaw_pitch_roll
        roll_current, _, _ = q_current.yaw_pitch_roll

        # Compute absolute difference in roll only
        roll_diff = abs(roll_current - roll_default)

        # Normalize roll difference to [0, π] range (max possible is π radians)
        roll_diff = min(roll_diff, 2 * pi - roll_diff)  # Handle wrap-around

        # Reward: +1 (perfect alignment) to -1 (π radian error)
        gripper_orientation_reward = 1.0 - 2.0 * (roll_diff / pi)

        total_reward = (arm_absolute_reward * armweight * absoluteweight + 
                        arm_relative_reward * armweight * relativeweight + 
                        arm_temporal_reward * armweight * temporalweight +
                        gripper_absolute_reward * gripperweight * absoluteweight + 
                        gripper_relative_reward * gripperweight * relativeweight + 
                        gripper_temporal_reward * gripperweight * temporalweight +
                        gripper_orientation_reward * gripperweight)

        result = {
            "arm_absolute_reward": arm_absolute_reward,
            "arm_relative_reward": arm_relative_reward,
            "arm_temporal_reward": arm_temporal_reward,
            "arm_progress": arm_progress,
            "arm_solved": arm_solved,
            "gripper_absolute_reward": gripper_absolute_reward,
            "gripper_relative_reward": gripper_relative_reward,
            "gripper_temporal_reward": gripper_temporal_reward,
            "gripper_progress": gripper_progress,
            "gripper_solved": gripper_solved,
            "total_reward": total_reward,
            "absolute_distance": absolute_distance,
            "goal_state": goal_state_value,
            "gripper_distance": grip_dist,
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
        self.prev_owner = None
        self.last_owner = None
        self.rewards_history = []
        self.network_rewards = [0] * self.num_networks
        self.finished = False
        
        # Load protorewards configuration from JSON file
        json_path = os.path.join(os.path.dirname(__file__), 'protorewards.json')
        with open(json_path, 'r') as f:
            self.protorewards_config = json.load(f)

    def reset(self, observation=None):
        """Reset all state for both UniversalReward and Rewarder."""
        # Reset UniversalReward state variables
        self.step = 0
        self.owner = 0
        self.current_network = 0
        self.network_name = self.network_names[self.owner]
        params = self.protoreward_params(self.network_name)
        
        # Get actual_state and goal_state values using paths from params
        actual_state_path = params.get("actual_state", ["actual_state"])
        goal_state_path = params.get("goal_state", ["goal_state"])
        
        actual_state_value = observation
        for key in actual_state_path:
            actual_state_value = actual_state_value[key]
        
        goal_state_value = observation
        for key in goal_state_path:
            goal_state_value = goal_state_value[key]
        
        self.max_trans_dist = self.task.calc_distance(actual_state_value, goal_state_value)
        self.min_trans_dist = self.task.calc_distance(goal_state_value, goal_state_value)
        self.max_rot_dist = self.task.calc_rot_quat(actual_state_value, goal_state_value) if params["rot"] else 0.0
        self.min_rot_dist = self.task.calc_rot_quat(goal_state_value, goal_state_value) if params["rot"] else 0.0
        self.prev_trans_dist = self.max_trans_dist
        self.prev_rot_dist = self.max_rot_dist
        self.absolute_reward_history = []
        self.relative_reward_history = []
        self.max_grip_dist = 1
        self.min_grip_dist = 0
        _,self.prev_grip_dist = self.env.robot.check_gripper_status(observation["additional_obs"]["gjoints_states"])
        self.grip_absolute_reward_history = []
        self.grip_relative_reward_history = []
        #This will calculate self.last_results
        result = self.calculate(observation, **params)
        self.last_result = result

        
        self.last_owner = None
        self.prev_owner = None
        self.network_rewards = [0] * self.num_networks

    def compute(self, observation=None):
        if not self.network_names:
            return 0.0
        
        # Ensure owner is within bounds
        if self.owner >= self.num_networks:
            self.owner = self.num_networks - 1
            
        self.network_name = self.network_names[self.owner]
        self.params = self.protoreward_params(self.network_name)

        result = self.calculate(observation, **self.params)
        self.last_result = result
        reward = result["total_reward"]
        

        self.prev_owner = self.last_owner

        # Check if arm is solved and progress to next network
        if result["arm_solved"] and result["gripper_solved"]:
            current_preds = self.env._get_current_predicates()
            if self.owner < self.num_networks - 1:
                if SubgoalPredicateResolver(self.task.current_subgoal).check(
                    placed_objects=self.env.env_objects,
                    env=self.env,
                    predicates=current_preds,
                ):
                    print(f"Subgoal {self.task.current_subgoal} satisfied, switching to ({self.network_names[self.owner + 1]})")
                    self.task.current_subgoal += 1
                    self.owner += 1

            else:
                if GoalPredicateResolver().check(
                    placed_objects=self.env.env_objects,
                    env=self.env,
                    predicates=current_preds,
                ):
                    self.finished = True
                    GREEN = "\033[92m"
                    RESET = "\033[0m"
                    print(f"{GREEN}Goal predicates satisfied{RESET}")
                self.task.check_goal()
                

        self.current_network = self.owner
        self.network_rewards[self.current_network] += reward
        self.last_owner = self.owner
        self.rewards_history.append(reward)
        return reward

    def protoreward_params(self, name):
        """Load protoreward parameters from JSON configuration file."""
        # Map single-letter abbreviations to full action names
        letter_to_name = {
            "A": "approach",
            "W": "withdraw",
            "G": "grasp",
            "D": "drop",
            "M": "move",
            "R": "rotate",
            "T": "transform",
            "F": "follow"
        }
        
        # Convert single letter to full name if applicable
        lookup_name = letter_to_name.get(name, name)
        
        if lookup_name in self.protorewards_config:
            return self.protorewards_config[lookup_name]
        else:
            raise ValueError(f"Unknown protoreward name: {name}. Available names: {list(self.protorewards_config.keys())}")
