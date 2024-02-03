"""
This script serves the purpose of visualizing generated (noisy) trajectory of object along with its
filtered result. Filtering is done through the Kalman and particle filtering algorithm.
"""

from ast import arg
import gym
from myGym import envs
import cv2
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env
import os, imageio
import numpy as np
import time
from numpy import matrix
import pybullet as p
import pybullet_data
import pkg_resources
import random
import getkey
import json, commentjson
from stable_baselines.bench import Monitor


def get_arg_dict():
    #Retreive arguments from config test_filter.json
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config, "r") as f:
        arg_dict = commentjson.load(f)
    #for key, value in vars(args).items():
        #arg_dict[key] = value
    return arg_dict



def visualize_env(arg_dict):
    #TODO: Visualize gym_env. First, without any trajectories, later with them.
    env_arguments = {"render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"],
                     "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                     "robot_action": arg_dict["robot_action"], "max_velocity": arg_dict["max_velocity"],
                     "max_force": arg_dict["max_force"], "task_type": arg_dict["task_type"],
                     "action_repeat": arg_dict["action_repeat"],
                     "task_objects": arg_dict["task_objects"], "observation": arg_dict["observation"],
                     "distractors": arg_dict["distractors"],
                     "num_networks": arg_dict.get("num_networks", 1),
                     "network_switcher": arg_dict.get("network_switcher", "gt"),
                     "distance_type": arg_dict["distance_type"], "used_objects": arg_dict["used_objects"],
                     "active_cameras": arg_dict["camera"], "color_dict": arg_dict.get("color_dict", {}),
                     "max_steps": arg_dict["max_episode_steps"], "visgym": arg_dict["visgym"],
                     "reward": arg_dict["reward"], "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                     "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"],
                     "natural_language": bool(arg_dict["natural_language"])
                     }
    if arg_dict["gui"] == 0:
        arg_dict["gui"] = 1
    env_arguments["gui_on"] = arg_dict["gui"]
    model_logdir = os.path.dirname(arg_dict.get("model_path", ""))

    env = gym.make(arg_dict["env_name"], **env_arguments)
    #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    #p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    env.render("human")

    return env


def get_input():
    #Retreive user input - trajectory parameters
    type = int(input("Enter trajectory type (1 for line, 2 for circle, 3 for spline)"))
    trajectory_dict = dict()
    if type == 1:
        trajectory_dict["type"] = "line"
        trajectory_dict["start"] = input("Enter starting point coordinates x,y,z (format 'x y z'): ")
        trajectory_dict["end"] = input("Enter ending point coordinates x,y,z (format 'x y z'): ")
        trajectory_dict["stdv"] = input("Enter noise standard deviation in meters: ")
    elif type == 2:
        trajectory_dict["type"] = "circle"
        trajectory_dict["center"] = input("Enter center coordinates x,y,z (format 'x y z'): ")
        trajectory_dict["radius"] = input("Enter radius r: ")
        trajectory_dict["plane"] = input("Enter the norm vector of the circle plane (e.g. '1 0 0' for x = 0 yz plane): ")
    elif type == 3:
        trajectory_dict["type"] = "2D_spline"
        print("Enter sequence of maximum 15 points to make spline from. To end the sequence, enter 'f'")
        point_list = []
        i = 1
        while True:
            point_list.append(input(f"Enter point {i}:"))
            if len(point_list) >= 15 or point_list[-1] == 'f':
                break
        trajectory_dict["points"] = point_list
    return trajectory_dict


def add_noise(traj, sigma):
    #Add noise to ground_truth trajectory
    noise_x, noise_y, noise_z = sigma * np.random.randn(len(traj)), sigma * np.random.randn(len(traj)), sigma * np.random.randn(len(traj))
    ret = np.copy(traj)
    ret[:, 0] += noise_x
    ret[:, 1] += noise_y
    ret[:, 2] += noise_z
    return ret


def vis_anim(arg_dict, env):
    #TODO: Loop env episodes so that visualization lasts. Possibly add object movement animation
    for i in range(50):
        #arg_dict["max_episode_steps"]
        for e in range(1000000):
            action = [0, 0, 0]
            obs, reward, done, info = env.step(action)
            if done:
                break



def create_trajectory_points(parameters, n):
    #TODO: From retreived trajectory parameters, create set of n points on the trajectory along with noisy data.
    if parameters["type"] == "line":
        start = np.array(list(map(float, parameters["start"].split())))
        end = np.array(list(map(float, parameters["end"].split())))
        print(f"start: {start}")
        print(f"end: {end}")
        ground_truth = np.linspace(start, end, n)
        print(ground_truth)
        sigma = parameters["stdv"]
    elif parameters["type"] == "circle":
        ground_truth = np.zeros((n, 3))
    elif parameters["type"] == "spline":
        ground_truth = np.zeros((n, 3))
    else:
        ground_truth = np.zeros((n, 3))
    noisy_data = add_noise(ground_truth, float(sigma))
    print(noisy_data)
    return ground_truth, noisy_data


def visualize_trajectory(ground_truth, noisy_data):
    for i in range(ground_truth.shape[0] - 1):
        p.addUserDebugLine(noisy_data[i, :], noisy_data[i+1, :])
        p.addUserDebugLine(ground_truth[i, :], ground_truth[i+1, :])


if __name__ == "__main__":
    params = get_input()
    ground_truth, noisy_data = create_trajectory_points(params, 20)
    arg_dict = get_arg_dict()
    env = visualize_env(arg_dict)
    env.reset()
    visualize_trajectory(ground_truth, noisy_data)
    vis_anim(arg_dict, env)


