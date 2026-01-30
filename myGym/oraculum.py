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
        self._gripper_open = gripper_open
        self._gripper_closed = gripper_closed


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

        Args:
            t (int): Current timestep in the simulation.
            env (Any): The simulation environment.
            arg_dict (Dict[str, Any]): Argument dictionary with configuration settings.
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
        if "absolute" in self._robot_action:
            reward_name = env.env.unwrapped.reward.reward_name
            gripper = "gripper" in self._robot_action
            if reward_name == "approach":
                self._set_gripper_action(action, "open", gripper)
                action[:3] = self._get_approach_action(env, info)
            elif reward_name == "grasp":
                self._set_gripper_action(action, "close", gripper)
            elif reward_name == "move":
                self._set_gripper_action(action, "close", gripper)
                # Check if the robot is close enough to the goal
                distance_to_goal = np.linalg.norm(np.array(info['o']["actual_state"][:2]) - np.array(info['o']["goal_state"][:2]))
                if info['o']["actual_state"][2] < -0.272:
                     action[:3] = info['o']["goal_state"][:3]
                     #action[2] += 0.1
                     #= i
                elif 0.22 > distance_to_goal > 0.13:  # Threshold for being "close enough"
                    action[:3] = info['o']["goal_state"][:3]
                    #action[2] += 0.065
                    #print(f"Close to goal, raising hand: {action[:3]}")
                else:
                    action[:3] = info['o']["goal_state"][:3]
            elif reward_name == "drop":
                if self.wait_n_steps(5):
                    self._set_gripper_action(action, "open", gripper)
                else:
                    pass
            elif reward_name == "withdraw":
                if self.wait_n_steps(25):
                    distance_to_goal = np.linalg.norm(
                        np.array(info['o']["goal_state"][:3]) - np.array(info['o']["actual_state"][:3]))
                    if distance_to_goal > 0.2:
                        self._set_gripper_action(action, "open", gripper)
                        action[:3] = np.array(info['o']["actual_state"][:3]) + DEFAULT_WITHDRAW_OFFSET
                    else:
                        action[:3] = info['o']["goal_state"][:3] + DEFAULT_WITHDRAW_OFFSET
                else:
                    action = action
            else:
                action[:3] = info['o']["goal_state"][:3]
        else:
            raise ValueError("Unsupported robot action type. Only 'absolute' is supported.")
        return action


    def _get_approach_action(self, env: Any, info: Dict[str, Any]) -> np.ndarray:
        """
        Determine the approach action based on the environment's reward structure.

        Args:
            env (Any): The simulation environment.
            info (Dict[str, Any]): Information returned from the environment.

        Returns:
            np.ndarray: Updated approach action.
        """
        if env.env.unwrapped.reward.rewards_num <= 2:
            action = info['o']["goal_state"][:3]
            return action
        action = info['o']["actual_state"][:3]
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
        """
        if gripper:
            if state == "close":
                action[-len(self._gripper_closed):] = self._gripper_closed
            elif state == "open":
                action[-len(self._gripper_open):] = self._gripper_open
        else:
            print("No gripper to control, change 'robot_action' to contain 'gripper'.")

    def check_gripper_status(self) -> str:
        """
        Check the current gripper status using the robot's check_gripper_status method.

        Returns:
            str: "open", "close", or "neutral" based on current gripper joint states.
        """
        robot = self._env.unwrapped.robot
        gripper_states = robot.get_gjoints_states()
        return robot.check_gripper_status(gripper_states)


    def wait_n_steps(self, n):
        if not hasattr(self, '_wait_n_steps'):
            self._wait_n_steps = 1
        elif self._wait_n_steps >= n:
            del(self._wait_n_steps)
            return True
        else:
            self._wait_n_steps += 1
            return False
