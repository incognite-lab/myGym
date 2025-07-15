import multiprocessing
import os
import subprocess
import time
from typing import Dict, Any

import cv2
import imageio
import numpy as np
import pybullet as p
import pybullet_data
from numpy import matrix
from sklearn.model_selection import ParameterGrid
import pandas as pd
import commentjson

from myGym import oraculum
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env, task_objects_replacement

clear = lambda: os.system('clear')

AVAILABLE_SIMULATION_ENGINES = ["mujoco", "pybullet"]
AVAILABLE_TRAINING_FRAMEWORKS = ["tensorflow", "pytorch"]

TASK_TYPE_MAPPING = {"A": "train_A_RDDL.json", "AG": "train_AG_RDDL.json", "AGM": "train_AGM_RDDL.json",
                     "AGR": "train_AGR_RDDL.json", "AGMD": "train_AGMD_RDDL.json", "AGMDW": "train_AGMDW_RDDL.json"}


def visualize_sampling_area(arg_dict: dict) -> None:
    task_object = arg_dict["task_objects"][0]
    goal_area = task_object["goal"]["sampling_area"]

    # Calculate the half-extents (rx, ry, rz)
    rx = (goal_area[0] - goal_area[1]) / 2
    ry = (goal_area[2] - goal_area[3]) / 2
    rz = (goal_area[4] - goal_area[5]) / 2

    # Create a visual shape and multi-body for the sampling area
    visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[rx, ry, rz], rgbaColor=[1, 0, 0, .2])
    collision = -1

    p.createMultiBody(
        baseVisualShapeIndex=visual,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=[goal_area[0] - rx, goal_area[2] - ry, goal_area[4] - rz],
    )


def visualize_trajectories(info: dict, action: list) -> None:
    visual_actual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 1, .3])
    collision = -1
    p.createMultiBody(
        baseVisualShapeIndex=visual_actual,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=info['o']['actual_state'],
    )

    visual_action = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, .3])
    p.createMultiBody(
        baseVisualShapeIndex=visual_action,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=action[:3],
    )


def visualize_goal(info: dict) -> None:
    visual_goal = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, .5])
    collision = -1
    p.createMultiBody(
        baseVisualShapeIndex=visual_goal,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=info['o']['goal_state'],
    )


def change_dynamics(cubex: int, lfriction: int, rfriction: int, ldamping: int, adamping: int) -> None:
    p.changeDynamics(cubex, -1, lateralFriction=p.readUserDebugParameter(lfriction))
    p.changeDynamics(cubex, -1, rollingFriction=p.readUserDebugParameter(rfriction))
    p.changeDynamics(cubex, -1, linearDamping=p.readUserDebugParameter(ldamping))
    p.changeDynamics(cubex, -1, angularDamping=p.readUserDebugParameter(adamping))


def visualize_infotext(action: list, env: object, info: dict) -> None:
    debug_params = [
        (f"Episode: {env.env.episode_number}", [0.65, 1., 0.45], [0.4, 0.2, .3]),
        (f"Step: {env.env.episode_steps}", [0.67, 1., .40], [0.2, 0.8, 1]),
        (f"Subtask: {env.env.task.current_task}", [0.69, 1., 0.35], [0.4, 0.2, 1]),
        (f"Network: {env.env.unwrapped.reward.current_network}", [0.71, 1., 0.3], [0.0, 0.0, 1]),
        (f"Action (Gripper): {matrix(np.around(np.array(action), 2))}", [0.73, 1., 0.25], [1, 0, 0]),
        (f"Actual State: {matrix(np.around(np.array(env.env.observation['task_objects']['actual_state'][:3]), 2))}",
         [0.75, 1., 0.2], [0.0, 1., 0.0]),
        (f"End Effector: {matrix(np.around(np.array(env.env.robot.end_effector_pos), 2))}", [0.77, 1., 0.15],
         [0.0, 1., 0.0]),
        (f"Object: {matrix(np.around(np.array(info['o']['actual_state']), 2))}", [0.8, 1., 0.10], [0.0, 0.0, 1]),
        (f"Velocity: {env.env.max_velocity}", [0.79, 1., 0.05], [0.6, 0.8, .3]),
        (f"Force: {env.env.max_force}", [0.81, 1., 0.00], [0.3, 0.2, .4]),
    ]

    for text, pos, color in debug_params:
        p.addUserDebugText(text, pos, textSize=1.0, lifeTime=0.5, textColorRGB=color)


# Function to detect key presses and update action accordingly
def detect_key(keypress: dict, arg_dict: dict, action: list) -> list:
    key_action_mapping = {
        97: (2, 0.03),  # A
        122: (2, -0.03),  # Z/Y
        65297: (1, -0.03),  # ARROW UP
        65298: (1, 0.03),  # ARROW DOWN
        65295: (0, 0.03),  # ARROW LEFT
        65296: (0, -0.03),  # ARROW RIGHT
        120: [(3, -0.03), (4, -0.03)],  # X
        99: [(3, 0.03), (4, 0.03)],  # C
        113: [(0, 0)],  # Q
        110: [(0, 0)]  # N - next task (useful when the robot gets stuck, no need to wait for episode end)
    }

    for key, value in key_action_mapping.items():
        if key in keypress.keys() and keypress[key] == 1:
            if key == 113:
                print("'Q' pressed, quitting")
                quit()
            elif (key == 99 or key == 120) and "gripper" not in arg_dict["robot_action"]:
                print("Gripper is not present, cannot perform actions 'C' or 'X'")
            elif isinstance(value, tuple):
                action[value[0]] += value[1]
            elif isinstance(value, list):
                for v in value:
                    action[v[0]] += v[1]

    if "step" in arg_dict["robot_action"]:
        action[:3] = np.multiply(action[:3], 10)
    elif "joints" in arg_dict["robot_action"]:
        print("Robot action: Joints - KEYBOARD CONTROL UNDER DEVELOPMENT, quitting")
        quit()

    return action


def n_pressed(last_call_time):
    """Function which detects, whether the key n was pressed and new episode should be launched"""
    keypress = p.getKeyboardEvents()
    now = time.time()
    if now - last_call_time > 0.5:
        for key in keypress.keys():
            if key == 110:
                print("N pressed, switching to next subtask")
                return True, now
        return False, last_call_time
    else:
        return False, last_call_time


def test_env(env: object, arg_dict: dict) -> list:
    env.reset()
    current_result = None
    global done
    # Prepare names for sliders
    joints = [f"Joint{i}" for i in range(1, 20)]
    jointparams = [f"Jnt{i}" for i in range(1, 20)]

    images = []
    action = None
    info = None

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    last_call_time = time.time()
    if arg_dict["control"] == "slider":
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        if "joints" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print("gripper is present")
                for i in range(env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i],
                                                            env.action_space.high[i], env.env.robot.init_joint_poses[i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i],
                                                            env.action_space.high[i], .02)
            else:
                for i in range(env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i], env.action_space.high[i],
                                                        env.env.robot.init_joint_poses[i])
        elif "absolute" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print("gripper is present")
                for i in range(env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, arg_dict["robot_init"][i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, .02)
            else:
                for i in range(env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], -1, 1, arg_dict["robot_init"][i])
        elif "step" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print("gripper is present")
                for i in range(env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, 0)
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, .02)
            else:
                for i in range(env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], -1, 1, 0)

    p.addUserDebugParameter("Lateral Friction", 0, 100, 0)
    p.addUserDebugParameter("Spinning Friction", 0, 100, 0)
    p.addUserDebugParameter("Linear Damping", 0, 100, 0)
    p.addUserDebugParameter("Angular Damping", 0, 100, 0)

    if arg_dict["vsampling"]:
        visualize_sampling_area(arg_dict)
    if arg_dict["control"] == "random":
        action = env.action_space.sample()
    if arg_dict["control"] == "keyboard":
        action = arg_dict["robot_init"]
        if "gripper" in arg_dict["robot_action"]:
            action.append(.1)
            action.append(.1)
    if arg_dict["control"] == "slider":
        action = []
        for i in range(env.action_space.shape[0]):
            jointparams[i] = p.readUserDebugParameter(joints[i])
            action.append(jointparams[i])

    eval_episodes = arg_dict.get("eval_episodes", 50)
    for e in range(eval_episodes):
        obs, info = env.reset()
        # Store oraculum results if selected:
        if arg_dict["results_report"]:
            if len(arg_dict["task_type"]) <=2: #A or AG task
                positions = [info['o']['actual_state'],  None,
                              info['o']['goal_state']]
            else:
                positions = [info['o']["additional_obs"]["endeff_xyz"],
                             info['o']['actual_state'],
                             info['o']['goal_state']]
            current_result = [arg_dict["task_type"], arg_dict["workspace"], arg_dict["robot"],
                            np.round(np.array(positions[0]), 2) if positions[0] is not None else None,
                            np.round(np.array(positions[1]), 2) if positions[1] is not None else None,
                            np.round(np.array(positions[2]), 2) if positions[2] is not None else None]
        for t in range(arg_dict["max_episode_steps"]):
            if arg_dict["control"] == "slider":
                action = []
                for i in range(env.action_space.shape[0]):
                    jointparams[i] = p.readUserDebugParameter(joints[i])
                    action.append(jointparams[i])

            if arg_dict["control"] == "observation":
                if t == 0:
                    action = env.action_space.sample()
                else:
                    if "joints" in arg_dict["robot_action"]:
                        action = info['o']["additional_obs"]["joints_angles"]
                    elif "absolute" in arg_dict["robot_action"]:
                        action = info['o']["actual_state"]
                    else:
                        action = [0, 0, 0]
            if arg_dict["control"] == "oraculum":
                action = oraculum.perform_oraculum_task(t, env, arg_dict, action, info)
            elif arg_dict["control"] == "keyboard":
                keypress = p.getKeyboardEvents()
                action = detect_key(keypress, arg_dict, action)
            elif arg_dict["control"] == "random":
                action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # print("Done step:", env.unwrapped.episode_steps, "for task type:", arg_dict["task_type"], "with robot", arg_dict["robot"])
            n_p, last_call_time = n_pressed(last_call_time)
            if n_p:  # If key 'n' is pressed, switch to next task - useful if robot gets stuck
                env.unwrapped.task.end_episode_fail("manual_switch")
                done = True

            if arg_dict["results_report"] and done:
                if terminated:
                    current_result.append(True)
                elif truncated:
                    current_result.append(False)
                else:
                    current_result.append(False)

            if arg_dict["vtrajectory"]:
                visualize_trajectories(info, action)
            if arg_dict["vinfo"]:
                visualize_infotext(action, env, info)

            if "step" in arg_dict["robot_action"]:
                action[:3] = [0, 0, 0]

            if arg_dict["visualize"]:
                visualizations = [[], []]
                env.render()
                for camera_id in range(arg_dict["camera"]):
                    camera_render = env.render()
                    image = cv2.cvtColor(camera_render[camera_id]["image"], cv2.COLOR_RGB2BGR)
                    depth = camera_render[camera_id]["depth"]
                    image = cv2.copyMakeBorder(image, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    cv2.putText(image, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 0, 0), 1, 0)
                    visualizations[0].append(image)
                    depth = cv2.copyMakeBorder(depth, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    cv2.putText(depth, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 0, 0), 1, 0)
                    visualizations[1].append(depth)

                if len(visualizations[0]) % 2 != 0:
                    visualizations[0].append(255 * np.ones(visualizations[0][0].shape, dtype=np.uint8))
                    visualizations[1].append(255 * np.ones(visualizations[1][0].shape, dtype=np.float32))
                fig_rgb = np.vstack((np.hstack((visualizations[0][0::2])), np.hstack((visualizations[0][1::2]))))
                fig_depth = np.vstack((np.hstack((visualizations[1][0::2])), np.hstack((visualizations[1][1::2]))))
                cv2.imshow('Camera RGB renders', fig_rgb)
                cv2.imshow('Camera depth renders', fig_depth)
                cv2.waitKey(1)

            if arg_dict["record"] > 0 and len(images) < 80000:
                if len(images) < 1:
                    avi_path = make_path(arg_dict, ".avi", False)
                    gif_path = make_path(arg_dict, ".gif", False)
                if arg_dict["record"] == 1:
                    record_video(images, arg_dict, env, gif_path)
                elif arg_dict["record"] == 2:
                    record_video(images, arg_dict, env, avi_path)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env_p = env.unwrapped.p
    env_p.disconnect()
    while env_p.isConnected():
        continue
    return current_result



def record_video(images: list, arg_dict: dict, env: object, path: str) -> None:
    if arg_dict["camera"] < 1:
        raise ValueError("Camera parameter must be set to > 0 to record!")

    render_info = env.render()
    image = render_info[arg_dict["camera"] - 1]["image"]
    images.append(image)
    print(f"appending image; total size: {len(images)}")
    if len(images) >= 80000:
        print(f"too many images; total size: {len(images)}")
    if ".gif" in path and done:
        imageio.mimsave(path, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], duration=65)
        os.system(
            './utils/gifopt -O3 --lossy=5 --colors 256 -o {dest} {source}'.format(source=path, dest=path))
        print("Record saved to " + path)
    elif ".avi" in path and done:
        height, width, layers = image.shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
        for img in images:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print("Record saved to " + path)


def make_path(arg_dict: dict, record_format: str, model: bool):
    counter = 0
    if model:
        model_logdir = os.path.dirname(arg_dict.get("model_path", ""))
        model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
        logdir = os.path.join(model_logdir, "train_record_" + model_name)
        print("Saving to " + logdir)
    else:
        if not os.path.exists(arg_dict["logdir"]):
            os.makedirs(arg_dict["logdir"])
        logdir = os.path.join(arg_dict["logdir"], "train_record_" + arg_dict["task_type"])

    video_path = logdir + "_" + str(counter) + record_format
    while os.path.exists(video_path):
        counter += 1
        video_path = logdir + "_" + str(counter) + record_format

    return video_path

def multiconfig_checker(multiconfig):
    """
    Tests whether given multiconfig is valid - checks for format and whether all the arrays have the same length.
    Parameters:
        "multiconfig" (str): location of multiconfig file
    """
    try:
        with open(multiconfig, "r") as f:
            multi_args = commentjson.load(f)
            arr_length = None
            for key, value in multi_args.items():
                if arr_length is not None:
                    if len(value) != arr_length:
                        print("Arrays in multiconfig do not have the same length, please change the config.")
                        raise ValueError
                else:
                    arr_length = len(value)
        return multi_args, arr_length
    except Exception as e:
        print("Error, failed to check multiconfig, error:", e)
        return None, None


def get_multitest_args(multiconfig, base_arg_dict, config, i):
    #2)Replace the base layer of arguments with arguments from retrieved config
    new_arg_dict = base_arg_dict.copy()
    with open(config, "r") as f:
        arg_dict = commentjson.load(f)
    for key, value in arg_dict.items():
        if value is not None and key != "config":
            if key in ["robot_init"] or key in ["end_effector_orn"]:
                new_arg_dict[key] = [float(arg_dict[key][i]) for i in range(len(arg_dict[key]))]
            elif type(value) is list and len(value) <= 1 and key != "task_objects":
                new_arg_dict[key] = value[0]
        if value is not None:
                new_arg_dict[key] = value
    #print("Arg dict task objects after config replacement:", new_arg_dict)
    #3)Last layer of replacement - replace current args with the last args from multiconfig (i.e. Robot, Gripper init..)
    for key, value in multiconfig.items():
        if value is not None and key != "Task type":
            new_arg_dict[key] = value[i]
    # print("-------------------------------------------------------------------")
    # print("Arg dict task objects after multiconfig replacement:", new_arg_dict["task_objects"])
    # print("-------------------------------------------------------------------")
    return new_arg_dict

def print_task_info(arg_dict):
    print("-----------------------------------")
    print("Task type:", arg_dict["task_type"])
    print("Robot type:", arg_dict["robot"])
    print("----------------------------------")



def main() -> None:
    """Main entry point for the testing script."""
    parser = get_parser()
    parser.add_argument("-ct", "--control", default="oraculum",
                        help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider.")
    parser.add_argument("-vs", "--vsampling", action="store_true", help="Visualize sampling area.")
    parser.add_argument("-vt", "--vtrajectory", action="store_true", help="Visualize gripper trajectory.")
    parser.add_argument("-vn", "--vinfo", action="store_true", help="Visualize info. Valid arguments: True, False")
    parser.add_argument("-ns", "--network_switcher", default="gt", help="How does a robot switch to next network (gt or keyboard)")
    parser.add_argument("-rr", "--results_report", default = False, help="Used only with oraculum - shows report of task feasibility at the end.")
    parser.add_argument("-tp", "--top_grasp", default = True, help="Use top grasp when reaching objects with oraculum.")
    #The most important argument for multitest:
    parser.add_argument("-mcfg", "--multiconfig", default = "./configs/multiconfig1.json", help="Config with a list of configs for all the tested tasks.")
    # parser.add_argument("-nl", "--natural_language", default=False, help="NL Valid arguments: True, False")
    #1) Get the first layer of arguments from parser
    base_arg_dict, commands = get_arguments(parser)
    parameters = {}
    # args = parser.parse_args()
    multiconfig, num_tasks = multiconfig_checker(base_arg_dict["multiconfig"])
    if multiconfig is not None:
        print("Multiconfig check passed!")
    else:
        print("Multiconfig check failed!")
        quit()

    results = pd.DataFrame(
        columns=["Task type", "Workspace", "Robot", "Gripper init", "Object init", "Object goal", "Success"])

    for i in range(num_tasks):
        current_task_type = multiconfig["Task type"][i]
        current_config = os.path.join("./configs/", TASK_TYPE_MAPPING[current_task_type])
        arg_dict = get_multitest_args(multiconfig, base_arg_dict, current_config, i)
        arg_dict["eval_episodes"] = 1#Maybe could put this into multiconfig
        #TODO: putting gui to 1 manually for now, but has to be fixed
        arg_dict["gui"] = 1
        if arg_dict["control"] == "oraculum":
            arg_dict["robot_action"] = "absolute_gripper"
        else:
            if arg_dict["results_report"]:
                print("Results report cannot be used without oraculum.")
                arg_dict["results_report"] = False
        print_task_info(arg_dict)
        model_logdir = os.path.dirname(arg_dict.get("model_path", ""))
        env = configure_env(arg_dict, model_logdir, for_train=False)

        current_result = test_env(env, arg_dict)
        results.loc[len(results)] = current_result  # Append result to pd dataframe
    if base_arg_dict["results_report"]:
        # results = results.round(2)
        i=1
        print(results.dtypes)
        print(results)
        print(type(results))
        while True:
            filename = f"./oraculum_results/results{i}.csv"
            if not(os.path.exists(filename)):
                break
            i+=1
        results.to_csv(filename, index = False)



if __name__ == "__main__":
    main()
