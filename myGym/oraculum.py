from typing import Any, Dict, Optional

import numpy as np

# Constants
DEFAULT_WITHDRAW_OFFSET = np.array([0.0, 0.0, 0.35])

class Oraculum:

    def __init__(self, env: Any,  info: Dict[str, Any], robot_action: str, gripper_open, gripper_closed):
        self._env = env
        # self._max_episode_steps = max_steps
        self._info = info
        self._robot_action = robot_action
        # Convert gripper values to numpy arrays for consistency with robot.py
        self._gripper_open = np.array(gripper_open)
        self._gripper_closed = np.array(gripper_closed)
        
        # Validate that gripper_open and gripper_closed have the same length
        if len(self._gripper_open) != len(self._gripper_closed):
            raise ValueError(f"gripper_open and gripper_closed must have the same length. "
                           f"Got {len(self._gripper_open)} and {len(self._gripper_closed)}")


    def reset_env(self, env):
        self._env = env


    # def check_task_feasibility(self):
    #     """The main oraculum function"""
    #     action = None
    #     for t in range(self._max_episode_steps):
    #         action = self.perform_oraculum_task(t, self._env, action, self._info)
    #         observation, reward, terminated, truncated, info = self._env.step(action)
    #         done = terminated or truncated
    #         self._info = info
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break



    def perform_oraculum_task(self, t: int, env: Any,
                              action: Optional[np.ndarray] = None, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Perform the Oraculum task based on the current timestep and environment state.
        Uses protoreward_params from the Rewarder to determine rot and gripper settings
        for the current subgoal.

        Args:
            t (int): Current timestep in the simulation.
            env (Any): The simulation environment.
            action (Optional[np.ndarray]): The action array to modify, defaults to None.
            info (Optional[Dict[str, Any]]): Information returned from the environment, defaults to None.

        Returns:
            np.ndarray: Updated action array based on Oraculum logic.
        """
        # Return a random action for the first step (timestep 0)
        if t == 0:
            return env.action_space.sample()

        # Ensure info is not None (required for non-initial steps)
        if info is None:
            raise ValueError("Info dictionary must be provided for non-initial timesteps.")

        # Check for 'absolute' control mode in robot action
        if "absolute" not in self._robot_action:
            raise ValueError("Unsupported robot action type. Only 'absolute' is supported.")

        # Get current subgoal and params from Rewarder
        reward = env.env.unwrapped.reward
        current_subgoal = reward.network_names[reward.owner]
        params = reward.protoreward_params(current_subgoal)
        rot = params["rot"]
        gripper_param = params["gripper"]  # "Open" or "Close"
        has_gripper = "gripper" in self._robot_action

        # Set gripper state based on protoreward_params
        gripper_action = "open" if gripper_param == "Open" else "close"
        self._set_gripper_action(action, gripper_action, has_gripper)

        # Calculate arm goal state (position + orientation if rot, position only otherwise)
        arm_goal = self._compute_arm_goal(current_subgoal, env, info, rot)
        if arm_goal is not None:
            # Determine how many action indices to set (avoid overwriting gripper)
            gripper_len = len(self._gripper_open) if has_gripper else 0
            end_idx = min(len(arm_goal), len(action) - gripper_len)
            action[:end_idx] = arm_goal[:end_idx]

        return action


    def _compute_arm_goal(self, subgoal: str, env: Any, info: Dict[str, Any], rot: bool) -> Optional[np.ndarray]:
        """
        Compute the arm goal state for the current subgoal.
        If rot is True, returns position + orientation of goal state.
        If rot is False, returns only position.

        Args:
            subgoal (str): Name of the current subgoal (e.g., "approach", "grasp", "move").
            env (Any): The simulation environment.
            info (Dict[str, Any]): Information returned from the environment.
            rot (bool): Whether to include orientation in the goal state.

        Returns:
            Optional[np.ndarray]: Arm goal array, or None if no arm movement is needed.
        """
        goal_dim = len(info['o']["goal_state"]) if rot else 3

        if subgoal == "approach":
            return self._get_approach_action(env, info, goal_dim)
        elif subgoal == "grasp":
            return None  # No arm movement, just close gripper
        elif subgoal == "move":
            return np.array(info['o']["goal_state"][:goal_dim])
        elif subgoal == "drop":
            return None  # No arm movement, just open gripper
        elif subgoal == "withdraw":
            # Withdraw uses only position (3D) regardless of rot, as the offset is position-based
            return np.array(info['o']["actual_state"][:3]) + DEFAULT_WITHDRAW_OFFSET
        elif subgoal in ("rotate", "transform", "follow"):
            return np.array(info['o']["goal_state"][:goal_dim])
        else:
            return np.array(info['o']["goal_state"][:goal_dim])


    def _get_approach_action(self, env: Any, info: Dict[str, Any], goal_dim: int) -> np.ndarray:
        """
        Determine the approach action based on the environment's reward structure.
        Uses goal_dim to include orientation when rot is enabled.

        Args:
            env (Any): The simulation environment.
            info (Dict[str, Any]): Information returned from the environment.
            goal_dim (int): Number of dimensions for the goal (3 for position, 6 for position+orientation).

        Returns:
            np.ndarray: Updated approach action.
        """
        if len(env.env.unwrapped.reward.network_names) <= 2:
            return np.array(info['o']["goal_state"][:goal_dim])
        action = np.array(info['o']["actual_state"][:goal_dim])
        if info['o']["actual_state"][2] < -0.25:
            action[2] += 0.01
        return action


    def _set_gripper_action(self, action: np.ndarray, state: str, gripper: bool) -> None:
        """
        Set the gripper action (open or closed) using g_dict values.

        Args:
            action (np.ndarray): The action array to modify.
            state (str): The state of the gripper ("open" or "close").
            gripper (bool): Whether gripper control is enabled.
        
        Raises:
            ValueError: If state is not "open" or "close".
        """
        if gripper:
            if state == "close":
                action[-len(self._gripper_closed):] = self._gripper_closed
            elif state == "open":
                action[-len(self._gripper_open):] = self._gripper_open
            else:
                raise ValueError(f"Invalid gripper state '{state}'. Must be 'open' or 'close'.")
        else:
            print("No gripper to control, change 'robot_action' to contain 'gripper'.")


    def wait_n_steps(self, n):
        if not hasattr(self, '_wait_n_steps'):
            self._wait_n_steps = 1
        elif self._wait_n_steps >= n:
            del(self._wait_n_steps)
            return True
        else:
            self._wait_n_steps += 1
            return False
