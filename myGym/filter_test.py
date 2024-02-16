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
from pyquaternion import Quaternion
import sys
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt



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
    #model_logdir = os.path.dirname(arg_dict.get("model_path", ""))

    env = gym.make(arg_dict["env_name"], **env_arguments)
    #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    global dt
    dt = arg_dict["dt"]
    #p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    env.render("human")

    return env


def get_parameters_from_config(type):
    #TODO: load trajectory_config.json and create correct trajectory dict
    trajectory_dict = dict()
    with open("./configs/trajectory_config.json", "r") as f:
        args = commentjson.load(f)
    if type == 1:
        traj = args["positions"][0]
        trajectory_dict["type"] = traj["type"]
        trajectory_dict["start"] = traj["start"]
        trajectory_dict["end"] = traj["end"]
    elif type == 2:
        traj = args["positions"][1]
        trajectory_dict["type"] = traj["type"]
        trajectory_dict["center"] = traj["center"]
        trajectory_dict["radius"] = traj["radius"]
        trajectory_dict["plane"] = traj["plane"]
    elif type == 3:
        traj = args["positions"][2]
        trajectory_dict["type"] = traj["type"]
        trajectory_dict["points"] = traj["points"]
    trajectory_dict["stdv"] = args["noise"]["position"]
    trajectory_dict["q_stdv"] = args["noise"]["orientation"]
    trajectory_dict["q_start"] = args["rotations"]["q_start"]
    trajectory_dict["q_end"] = args["rotations"]["q_end"]
    return trajectory_dict


def get_input():
    #Retreive user input - trajectory parameters
    input_option = int(input("Enter 1 to create a custom trajectory, 2 to load predefined from config: "))
    if input_option == 1:
        type = int(input("Enter trajectory type (1 for line, 2 for circle, 3 for spatial spline: )"))
        trajectory_dict = dict()
        if type == 1:
            trajectory_dict["type"] = "line"
            trajectory_dict["start"] = input("Enter starting point coordinates x,y,z (format 'x y z'): ")
            trajectory_dict["end"] = input("Enter ending point coordinates x,y,z (format 'x y z'): ")
            trajectory_dict["stdv"] = input("Enter position noise standard deviation in meters: ")
        elif type == 2:
            trajectory_dict["type"] = "circle"
            trajectory_dict["center"] = input("Enter center coordinates x,y,z (format 'x y z'): ")
            trajectory_dict["radius"] = input("Enter radius r: ")
            trajectory_dict["plane"] = input("Enter the norm vector of the circle plane (e.g. '1 0 0' for x = 0 yz plane): ")
        elif type == 3:
            trajectory_dict["type"] = "spline"
            print("Enter sequence of maximum 10 points to make spline from. To end the sequence, enter 'f' ")
            point_list = []
            i = 1
            while True:
                point_list.append(input(f"Enter point {i}:"))
                if len(point_list) >= 10:
                    break
                if point_list[-1] == 'f':
                    point_list.pop(-1)
                    break

                i += 1
            trajectory_dict["points"] = point_list
        elif type == 4:
            trajectory_dict["type"] = None
            return trajectory_dict
    else:
        type = int(input("Enter trajectory type (1 for line, 2 for circle, 3 for spatial spline: )"))
        trajectory_dict = get_parameters_from_config(type)
    #trajectory_dict["q_start"] = input("Enter starting rotation in axis-angle representation (format 'x y z theta')")
    #trajectory_dict["q_end"] = input("Enter ending rotation in axis-angle representation [rad] (format 'x y z theta')")
    #trajectory_dict["q_stdv"] = input("Enter rotation noise standard deviation in radians")
    return trajectory_dict


def add_noise(traj, sigma):
    #Add noise to ground_truth trajectory
    noise_x, noise_y, noise_z = sigma * np.random.randn(len(traj)), sigma * np.random.randn(len(traj)), sigma * np.random.randn(len(traj))
    ret = np.copy(traj)
    ret[:, 0] += noise_x
    ret[:, 1] += noise_y
    ret[:, 2] += noise_z
    return ret


def create_circle(radius, center, normal, n):
    """
    Create circular trajectory of n points
    """
    trajectory = np.zeros((n, 3))
    phi = np.arctan2(normal[1],normal[0]); #azimuth angle, in [-pi, pi]
    theta = np.arctan2(np.sqrt(normal[0]**2 + normal[1]**2) ,normal[2]); # zenith angle, in [0,pi]
    t = np.linspace(0, 2*np.pi, n)
    x = center[0] - radius * (np.cos(t)*np.sin(phi) + np.sin(t)*np.cos(theta)*np.cos(phi))
    y = center[1] + radius * (np.cos(t)*np.cos(phi) - np.sin(t)*np.cos(theta)*np.sin(phi))
    z = center[2] + radius * np.sin(t)*np.sin(theta)
    trajectory[:, 0] = x
    trajectory[:, 1] = y
    trajectory[:, 2] = z
    print(trajectory)
    #sys.exit()
    return trajectory


def fit_polynomial(points, n):
    """
    Make a polynomial of degree len(points) -1. Resulting trajectory will have n points
    """
    trajectory = np.zeros((n, 3))
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    print("orig x, y, z:", x, y, z)
    deg = points.shape[0] -1
    t = np.linspace(0, 3, deg + 1) #3 seconds for full trajectory traversal

    fitx = polynomial(np.flip(np.polyfit(t, x, deg)))
    fity = polynomial(np.flip(np.polyfit(t, y, deg)))
    fitz = polynomial(np.flip(np.polyfit(t, z, deg)))
    time = np.linspace(0, 3, n)

    trajectory[:, 0] = fitx(time)
    trajectory[:, 1] = fity(time)
    trajectory[:, 2] = fitz(time)
    print("Trajectory = ", trajectory)
    return trajectory


def add_rotation_noise(rotations, sigma_q):
    #TODO: Add gaussian noise to series of rotation Quaternions
    noisy_rotations = []
    for i in range(len(rotations)):
        rot = rotations[i]
        q = np.concatenate((rot.imaginary, [rot.scalar])).flatten()
        rot_euler = np.array(p.getEulerFromQuaternion(q)) #Conversion to Euler angles
        rot_euler[0] += np.random.randn()*sigma_q
        rot_euler[1] += np.random.randn()*sigma_q
        rot_euler[2] += np.random.randn()*sigma_q
        q_noisy = np.array(p.getQuaternionFromEuler(rot_euler))
        print("q_noisy:", q_noisy)
        quat_noisy = Quaternion(imaginary = q_noisy[:3], real = q_noisy[3])
        noisy_rotations.append(quat_noisy)
    return noisy_rotations


def create_trajectory_points(parameters, n):
    #TODO: From retreived trajectory parameters, create set of n 6D points on the trajectory along with noisy data.
    ground_truth = np.zeros((n, 3))
    noisy_data = np.zeros((n, 3))
    if parameters["type"] == None:
        ground_truth, noisy_data, rotations, noisy_rotations = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    elif parameters["type"] == "line":
        #Ground truth position points line
        start = np.array(list(map(float, parameters["start"].split())))
        end = np.array(list(map(float, parameters["end"].split())))
        ground_truth[:, :3] = np.linspace(start, end, n)

    elif parameters["type"] == "circle":
        # Ground truth position points circle
        radius = float(parameters["radius"])
        center = np.array(list(map(float, parameters["center"].split())))
        norm = np.array(list(map(float, parameters["plane"].split())))
        ground_truth[:, :3] = create_circle(radius, center, norm, n)
    elif parameters["type"] == "spline":
        points_amount = len(parameters["points"])
        point_list = np.zeros((points_amount, 3))
        for i in range(points_amount):
            point_list[i, :] = np.array(list(map(float, parameters["points"][i].split())))
        ground_truth[:, :3] = fit_polynomial(point_list, n)
    else:
        ground_truth = np.zeros((n, 3))

    if parameters["type"] is not None:
        # Ground truth rotation quaternions
        print("type:", parameters["type"])
        q1 = np.array(list(map(float, parameters["q_start"].split())))
        q2 = np.array(list(map(float, parameters["q_end"].split())))
        q_start = Quaternion(axis=q1[:3], angle=np.deg2rad(q1[3]))
        q_end = Quaternion(axis=q2[:3], angle=np.deg2rad(q2[3]))
        rotations = []
        for rot in Quaternion.intermediates(q_start, q_end, n):
            rotations.append(rot)
        print("ROTATIONS:", rotations)
        sigma_q = np.deg2rad(float(parameters["q_stdv"]))
        sigma = float(parameters["stdv"])
        #Adding noise to position and rotation
        noisy_data = add_noise(ground_truth, float(sigma))
        noisy_rotations = add_rotation_noise(rotations, sigma_q)


    return ground_truth, noisy_data, rotations, noisy_rotations


def visualize_errors(ground_truth, noisy_data):
    #visualization of position errors
    diffs = []
    for i in range(ground_truth.shape[0]):
        diffs.append(np.linalg.norm(ground_truth[i, :] - noisy_data[i, :]))
        #diffs.append(np.sqrt(np.dot(ground_truth[i, :], noisy_data[i, :])))
    diffs = np.array(diffs)
    k = np.arange(len(diffs))
    plt.plot(k, diffs)
    plt.show()


def visualize_rot_errors(rotations, noisy_rotations):
    """
    Visualization of rotation errors
    """
    diffs = []
    print("rotations", rotations)
    print("noisy rotations", noisy_rotations)
    for i in range(len(rotations)):
        diffs.append(Quaternion.absolute_distance(rotations[i], noisy_rotations[i]))
        # diffs.append(np.sqrt(np.dot(ground_truth[i, :], noisy_data[i, :])))
    diffs = np.array(diffs)
    k = np.arange(len(diffs))
    plt.plot(k, diffs)
    plt.title("Rotation errors")
    plt.show()


def polynomial(coeffs):
    return lambda x: sum(a*x**i for i, a in enumerate(coeffs))


def vis_anim(ground_truth, noisy_data, rotations, noisy_rotations):
    """
    Parameters: ground_truth: np array of shape (n, 3) (xyz position and orientation quaternion object)
    """
    #TODO: Loop env episodes so that visualization lasts. Possibly add object movement animation
    env.p.setGravity(0, 0, 0)
    n = ground_truth.shape[0] #Number of points
    iter = 0
    for i in range(50):
        #arg_dict["max_episode_steps"]
        #env.task_objects["actual_state"].setGravity()
        for e in range(20000):
            if e%300 == 0:
                if iter >= n:
                    break
                env.task_objects["actual_state"].set_position(ground_truth[iter, :])
                env.task_objects["goal_state"].set_position(noisy_data[iter])
                rot = rotations[iter]
                noisy_rot = noisy_rotations[iter]
                q = np.concatenate((rot.imaginary, [rot.scalar])).flatten()
                noisy_q = np.concatenate((noisy_rot.imaginary, [noisy_rot.scalar])).flatten()
                print("iter:", iter, "e:", e, "n:", n)
                env.task_objects["actual_state"].set_orientation(q)
                env.task_objects["goal_state"].set_orientation(noisy_q)
                iter += 1

            action = [0, 0, 0]
            obs, reward, done, info = env.step(action)
            time.sleep(0.001)
        #print(env.task_objects)



def visualize_trajectory(ground_truth, noisy_data):
    if np.linalg.norm(ground_truth) > 0.1:
        for i in range(ground_truth.shape[0] - 1):
            p.addUserDebugLine(noisy_data[i, :], noisy_data[i+1, :])
            p.addUserDebugLine(ground_truth[i, :], ground_truth[i+1, :])


if __name__ == "__main__":
    params = get_input()
    ground_truth, noisy_data, rotations, noisy_rotations = create_trajectory_points(params, 40)
    arg_dict = get_arg_dict()
    env = visualize_env(arg_dict)
    env.reset()
    visualize_trajectory(ground_truth, noisy_data)
    #visualize_errors(ground_truth, noisy_data)
    #visualize_rot_errors(rotations, noisy_rotations)
    vis_anim(ground_truth, noisy_data, rotations, noisy_rotations)


