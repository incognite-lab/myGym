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
    #TODO: Retreive arguments from config test_filter.json
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config, "r") as f:
        arg_dict = commentjson.load(f)
    return arg_dict


def get_trajectory_parameters():
    #TODO: Somehow figure out how to retreive trajectory parameters. Arguments, config, GUI..
    pass


def visualize_env(arg_dict):
    # TODO: Visualize gym_env. First, without any trajectories, later with them.
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
    env = Monitor(env, model_logdir, info_keywords=tuple('d'))
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    env.render("human")
    time.sleep(10)

def add_noise(traj):
    #TODO: Add noise to ground_truth trajectory
    pass


def create_trajectory_points(parameters, n):
    #TODO: From retreived trajectory parameters, create set of n points on the trajectory along with noisy data.
    ground_truth = parameters
    noisy_data = add_noise(ground_truth)
    return ground_truth, noisy_data



if __name__ == "__main__":

    arg_dict = get_arg_dict()
    visualize_env(arg_dict)
    #parser.add_argument("-tr", "--trajectory", default = "line", help = "Type of trajectory (path of the object) to filter.")
    #parser.add_argument("")

