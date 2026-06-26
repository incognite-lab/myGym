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
            reward_name = env.env.unwrapped.reward.network_name
            reward_params = env.env.unwrapped.reward.params
            #print(f"Current reward: {reward_name} with params {reward_params}")
            action[:3] = env.env.unwrapped.reward.last_result["goal_state"][:3]
            gripper_state = reward_params["gripper"]
            gripper_values = env.env.unwrapped.robot.gripper_dict[gripper_state]
            #print(f"Oraculum setting gripper to {gripper_state} with values {gripper_values}")
            action[-len(gripper_values):] = gripper_values
            #print(f"Approach phase, setting gripper to {gripper_state}: {gripper_values}")


        else:
            raise ValueError("Unsupported robot action type. Only 'absolute' is supported.")
        return action



