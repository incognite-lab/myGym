from ast import arg
import gym
from myGym import envs
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env
import numpy as np
import time
from numpy import matrix
import pybullet as p
import pybullet_data
import pkg_resources
import random

import json, commentjson
from stable_baselines.bench import Monitor
from pyquaternion import Quaternion
import sys
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from myGym.envs.particle_filter import ParticleFilterGH




def quat_to_euler(quat):
    "Converts Quaternion object to Euler angles"
    q = np.concatenate((quat.imaginary, [quat.scalar])).flatten()
    rot_euler = np.array(p.getEulerFromQuaternion(q))  # Conversion to Euler angles
    return rot_euler


def euler_to_quat(euler_angles):
    """Converts Euler angles to Quaternion object"""
    q = np.array(p.getQuaternionFromEuler(euler_angles))
    quat = Quaternion(imaginary=q[:3], real=q[3])
    return quat


def polynomial(coeffs):
    """Return function which calculates polynomial of given coefficients at x"""
    return lambda x: sum(a*x**i for i, a in enumerate(coeffs))


def add_noise(traj, sigma):
    """Function that adds Gaussian noise of given sigma std to a trajectory"""
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
    phi = np.arctan2(normal[1],normal[0]) #azimuth angle, in [-pi, pi]
    theta = np.arctan2(np.sqrt(normal[0]**2 + normal[1]**2) ,normal[2]) #zenith angle, in [0,pi]
    t = np.linspace(0, 2*np.pi, n)
    x = center[0] - radius * (np.cos(t)*np.sin(phi) + np.sin(t)*np.cos(theta)*np.cos(phi))
    y = center[1] + radius * (np.cos(t)*np.cos(phi) - np.sin(t)*np.cos(theta)*np.sin(phi))
    z = center[2] + radius * np.sin(t)*np.sin(theta)
    trajectory[:, 0] = x
    trajectory[:, 1] = y
    trajectory[:, 2] = z
    return trajectory


def trajectory_length(trajectory):
    """
    Compute the total length of the trajectory
    """
    last_idx = trajectory.shape[0]
    diffs = trajectory[:last_idx - 1, :] - trajectory[1:last_idx]
    dists = np.linalg.norm(diffs, axis = 1)
    return np.sum(dists)


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
    return trajectory


def add_rotation_noise(rotations, sigma_q):
    """
    Add noise to rotational trajectory (series of quaternions)
    """
    noisy_rotations = []
    for i in range(len(rotations)):
        rot = rotations[i]
        #q = np.concatenate((rot.imaginary, [rot.scalar])).flatten()
        #rot_euler = np.array(p.getEulerFromQuaternion(q)) #Conversion to Euler angles
        #print("Adding noise to one angle of size:", np.random.randn()*sigma_q)
        #print("Adding noise to one angle of size:", np.random.randn() * sigma_q)
        #print("Adding noise to one angle of size:", np.random.randn() * sigma_q)
        #rot_euler[0] += np.random.randn()*sigma_q
        #rot_euler[1] += np.random.randn()*sigma_q
        #rot_euler[2] += np.random.randn()*sigma_q
        #q_noisy = np.array(p.getQuaternionFromEuler(rot_euler)) #Conversion back to quaternion
        #quat_noisy = Quaternion(imaginary = q_noisy[:3], real = q_noisy[3])
        w, x, y, z = np.random.randn()*sigma_q, np.random.randn()*sigma_q, np.random.randn()*sigma_q, np.random.randn()*sigma_q
        noise_q = Quaternion(w, x, y, z)
        quat_noisy = rot + noise_q
        noisy_rotations.append(quat_noisy)
    return noisy_rotations


def visualize_estimate(particle_filter):
    """Initializes the visualization of estimate position sphere."""
    position = particle_filter.estimate
    shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents = [0.02, 0.02, 0.02], rgbaColor = [0.2, 0.2, 0.2, 0.5])
    particle_id = p.createMultiBody(baseVisualShapeIndex = shape_id, basePosition = position)
    return particle_id


def visualize_trajectory(ground_truth, noisy_data):
    """Visualizes the trajectory of the ground truth and noisy data."""
    trajectory_ids = []
    noise_ids = []


    if np.linalg.norm(ground_truth) > 0.1:
        for i in range(ground_truth.shape[0] - 1):
            noise_id = p.addUserDebugLine(noisy_data[i, :], noisy_data[i+1, :])
            id = p.addUserDebugLine(ground_truth[i, :], ground_truth[i+1, :])
            trajectory_ids.append(id)
            noise_ids.append(noise_id)
    return trajectory_ids, noise_ids


def visualize_particles(particle_filter):
    """Initializes the visualization of position particles."""
    positions = particle_filter.particles[:, :3].tolist()
    #p.addUserDebugPoints(pointPositions = positions, pointColorsRGB=[20, 20, 20])
    particle_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.001)
    particles_id = p.createMultiBody(baseVisualShapeIndex=particle_id, batchPositions=positions)
    return particles_id


def move_particles(ids, positions):
    """Moves particle batch to a new position."""
    for i in range(len(ids)):
        p.resetBasePositionAndOrientation(ids[i], positions[i, :3], [1, 0, 0, 0])


def move_estimate(id, position, rotation):
    """Moves estimate sphere in visual environment"""
    p.resetBasePositionAndOrientation(id, position, [rotation[3], rotation[0], rotation[1], rotation[2]])


def visualize_env(arg_dict):
    """
    Initialization of gym_env visualization. Returns gym_env object.
    """
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
    env = gym.make(arg_dict["env_name"], **env_arguments)
    env.render("human")
    return env




def get_arg_dict():
    """
    Retreive necessary arguments from config test_filter.json
    """
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config, "r") as f:
        arg_dict = commentjson.load(f)
    return arg_dict



def load_rotations(filename):
    """
    Loads rotational trajectory from np file and converts into a sequence (list) of Quaternions
    """
    rot_file = np.load(filename)
    rotations = []
    for i in range(rot_file.shape[0]):
        quat = Quaternion(rot_file[i,:])
        rotations.append(quat)
    return rotations


def load_trajectory(type):
    fn_gt = "./dataset/visualizer_trajectories/" + type +".npy"
    fn_nd = "./dataset/visualizer_trajectories/" + type + "_noise.npy"
    fn_r = "./dataset/visualizer_trajectories/" + type + "_rot.npy"
    fn_nr = "./dataset/visualizer_trajectories/" + type + "_rot_noise.npy"
    ground_truth = np.load(fn_gt)
    noisy_data = np.load(fn_nd)
    rotations = np.load(fn_r)
    noisy_rotations = np.load(fn_nr)
    return ground_truth, rotations, noisy_data, noisy_rotations




