from typing import Any, Dict, Optional

import numpy as np

# Constants
GRIPPER_OPEN = 1
GRIPPER_CLOSED = 0
DEFAULT_WITHDRAW_OFFSET = np.array([0.0, 0, 0.4])


def perform_oraculum_task(t: int, env: Any, arg_dict: Dict[str, Any],
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
    if "absolute" in arg_dict["robot_action"]:
        reward_name = env.env.unwrapped.reward.reward_name
        gripper = "gripper" in arg_dict["robot_action"]
        if reward_name == "approach":
            _set_gripper_action(action, GRIPPER_OPEN, gripper)
            action[:3] = _get_approach_action(env, info)
        elif reward_name == "grasp":
            _set_gripper_action(action, GRIPPER_CLOSED, gripper)
        elif reward_name == "move":
            _set_gripper_action(action, GRIPPER_CLOSED, gripper)
            # Check if the robot is close enough to the goal
            distance_to_goal = np.linalg.norm(np.array(info['o']["goal_state"][:2]) - np.array(info['o']["actual_state"][:2]))
            if info['o']["actual_state"][2] < -0.272:
                info['o']["goal_state"][2] += 0.1
                action[:3] = info['o']["goal_state"][:3]
            elif 0.20 > distance_to_goal > 0.09:  # Threshold for being "close enough"
                info['o']["goal_state"][2] += 0.23
                action[:3] = info['o']["goal_state"][:3]
                print(f"Close to goal, raising hand: {action[:3]}")
            else:
                action[:3] = info['o']["goal_state"][:3]
        elif reward_name == "drop":
            _set_gripper_action(action, GRIPPER_OPEN, gripper)
        elif reward_name == "rotate":
            action =np.zeros(9)
            action[:7] = info['o']["goal_state"] #rotate action includes both position and orientation
            _set_gripper_action(action, GRIPPER_CLOSED, gripper)
        elif reward_name == "withdraw":
            distance_to_goal = np.linalg.norm(
                np.array(info['o']["goal_state"][:3]) - np.array(info['o']["actual_state"][:3]))
            if distance_to_goal > 0.2:
                _set_gripper_action(action, GRIPPER_OPEN, gripper)
                action[:3] = info['o']["actual_state"][:3] + DEFAULT_WITHDRAW_OFFSET
            else:
                action[:3] = info['o']["goal_state"][2] + 0.1
        elif "reach" in arg_dict["task_type"]:
            action[:3] = _get_approach_action(env, info)
        else:
            action[:3] = info['o']["goal_state"][:3]
    else:
        raise ValueError("Unsupported robot action type. Only 'absolute' is supported.")
    return action


def _get_approach_action(env: Any, info: Dict[str, Any]) -> np.ndarray:
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
        # if info['o']["actual_state"][2] < -0.25:
        #     #action[2] += 0.05
        #     action[0] -= 0.05
        #     print("Too close to table, raising hand: {}".format(action))
        return action
    action = info['o']["actual_state"][:3]
    # if info['o']["actual_state"][2] < -0.25:
    #     #action[2] += 0.05
    #     # action[0] -= 0.05
    #     # print("Too close to table, raising hand: {}".format(action))
    return action

def _set_gripper_action(action: np.ndarray, state: int, gripper: bool) -> None:
    """
    Set the gripper action (open or closed).

    Args:
        action (np.ndarray): The action array to modify.
        state (int): The state of the gripper (0 for closed, 1 for open).
    """
    if gripper:
        if len(action) == 9:
            action[7:9] = (state, state)
        else:
            action[3:5] = (state, state)
    else:
        print("No gripper to control, change 'robot_action' to contain 'gripper'.")
