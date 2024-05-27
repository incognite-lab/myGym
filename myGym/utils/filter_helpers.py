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

import pandas as pd

import json, commentjson
from stable_baselines.bench import Monitor
from pyquaternion import Quaternion
import sys
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from myGym.envs.particle_filter import ParticleFilterGH

from decimal import Decimal, InvalidOperation




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


def create_circle(radius, center, normal, n, dt_std = None):
    """
    Create circular trajectory of n points
    """
    trajectory = np.zeros((n, 3))
    phi = np.arctan2(normal[1],normal[0]) #azimuth angle, in [-pi, pi]
    theta = np.arctan2(np.sqrt(normal[0]**2 + normal[1]**2) ,normal[2]) #zenith angle, in [0,pi]
    if dt_std is None:
        t = np.linspace(0, 2*np.pi, n)
    else:

        t_vec_base = np.linspace(0, 2*np.pi, n)
        t_noise = np.random.normal(0, dt_std, n)

        t = t_vec_base + t_noise

        if t[0] < 0:
            t -= t[0]
    x = center[0] - radius * (np.cos(t)*np.sin(phi) + np.sin(t)*np.cos(theta)*np.cos(phi))
    y = center[1] + radius * (np.cos(t)*np.cos(phi) - np.sin(t)*np.cos(theta)*np.sin(phi))
    z = center[2] + radius * np.sin(t)*np.sin(theta)
    if dt_std is None:
        t_vec = np.linspace(0, 0.2*(n-1), n)
    else:
        t_vec = (t/(2*np.pi))* 0.2*n
    trajectory[:, 0] = x
    trajectory[:, 1] = y
    trajectory[:, 2] = z
    return trajectory, t_vec


def trajectory_length(trajectory):
    """
    Compute the total length of the trajectory
    """
    last_idx = trajectory.shape[0]
    diffs = trajectory[:last_idx - 1, :] - trajectory[1:last_idx]
    dists = np.linalg.norm(diffs, axis = 1)
    return np.sum(dists)


def fit_polynomial(points, n, dt_std = None):
    """
    Make a polynomial of degree len(points) -1. Resulting trajectory will have n points
    """
    trajectory = np.zeros((n, 3))
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    deg = points.shape[0] -1
    t_max = (n-1)*0.2# n* dt
    t = np.linspace(0, t_max, deg + 1) #3 seconds for full trajectory traversal
    fitx = polynomial(np.flip(np.polyfit(t, x, deg)))
    fity = polynomial(np.flip(np.polyfit(t, y, deg)))
    fitz = polynomial(np.flip(np.polyfit(t, z, deg)))
    if dt_std is None:
        time = np.linspace(0, t_max, n)
    else:
        t_vec_base = np.linspace(0, t_max, n)
        t_noise = np.random.normal(0, dt_std, n)
        time = t_vec_base + t_noise
    trajectory[:, 0] = fitx(time)
    trajectory[:, 1] = fity(time)
    trajectory[:, 2] = fitz(time)
    return trajectory, time


def add_rotation_noise(rotations, sigma_q):
    """
    Add noise to rotational trajectory (series of quaternions)
    """
    noisy_rotations = []
    for i in range(len(rotations)):
        rot = rotations[i]
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


def calculate_t_vec_multimeasurements(t_vec, dt):
    n = len(t_vec)
    multi_measurements = 0
    zero_measurements = 0
    for i in range(n - 1):
        j = i
        iter_measurements = 0
        while t_vec[j] < dt * (i+1):
            iter_measurements += 1
            j += 1
        if iter_measurements > 1:
            multi_measurements += 1
        if iter_measurements == 0:
            zero_measurements += 1
    single_measurements = n - multi_measurements - zero_measurements
    return single_measurements, multi_measurements, zero_measurements


def visualize_gt(position):
    shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.0175, 0.0175, 0.0175], rgbaColor=[0.1, 0.8, 0.1, 0.8])
    gt_id = p.createMultiBody(baseVisualShapeIndex = shape_id, basePosition = position)
    return gt_id


def visualize_meas(position):
    shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0.75, 0.75, 0.1, 0.6])
    gt_id = p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position)
    return gt_id


def visualize_timestep_cube(position):
    shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.008, 0.008, 0.008], rgbaColor=[0.1, 0.1, 0.1, 0.8])
    cube_id = p.createMultiBody(baseVisualShapeIndex=shape_id, basePosition=position)
    return cube_id

def visualize_trajectory(ground_truth, noisy_data):
    """Visualizes the trajectory of the ground truth and noisy data."""
    trajectory_ids = []
    noise_ids = []
    if np.linalg.norm(ground_truth) > 0.1:
        for i in range(ground_truth.shape[0] - 1):
            if not np.allclose(noisy_data[i +1], [88., 88., 88.]):
                noise_id = p.addUserDebugLine(noisy_data[i, :], noisy_data[i+1, :])
                id = p.addUserDebugLine(ground_truth[i, :], ground_truth[i+1, :])
                trajectory_ids.append(id)
                noise_ids.append(noise_id)
    return trajectory_ids, noise_ids


def visualize_trajectory_timesteps(ground_truth, noisy_data):
    """
    Visualize trajectory including cubes to show timesteps
    """
    trajectory_ids = []
    noise_ids = []
    cube_ids = []
   #print("noisy_data:", noisy_data)
    last_noise_index = 0
    if np.linalg.norm(ground_truth) > 0.1:
        for i in range(ground_truth.shape[0] - 1):
            id = p.addUserDebugLine(ground_truth[i, :], ground_truth[i + 1, :])
            cube_id = visualize_timestep_cube(ground_truth[i, :])
            trajectory_ids.append(id)
            if i == 0 and np.allclose(noisy_data[i, :], [88., 88., 88.]):
                last_noise_index = i+1
                continue
            if np.allclose(noisy_data[i+1, :], [88., 88., 88.]):
                continue
            # if not np.allclose(noisy_data[last_noise_index, :], [88., 88., 88.]):
            #     if not np.allclose(noisy_data[i +1, :], [88., 88., 88.]):
            noise_id = p.addUserDebugLine(noisy_data[last_noise_index, :], noisy_data[i + 1, :])
            noise_ids.append(noise_id)
            cube_ids.append(cube_id)
            last_noise_index = i + 1

    return trajectory_ids, noise_ids, cube_ids


def visualize_particles(particle_filter, vis_option):
    """Initializes the visualization of position particles."""
    pos = particle_filter.particles[:, :3]
    if vis_option == "1":
        radius = 0.001
        positions = pos.tolist()
    elif vis_option == "2":
        positions = pos[np.random.randint(pos.shape[0], size = 75), :]
        radius = 0.0015
    else:
        positions = pos[np.random.randint(pos.shape[0], size = 1), :]
        radius = 0.0001
    #p.addUserDebugPoints(pointPositions = positions, pointColorsRGB=[20, 20, 20])
    particle_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius)
    particles_id = p.createMultiBody(baseVisualShapeIndex=particle_id, batchPositions=positions)
    return particles_id


def move_particles(ids, positions):
    """Moves particle batch to a new position."""
    for i in range(len(ids)):
        p.resetBasePositionAndOrientation(ids[i], positions[i, :3], [1, 0, 0, 0])


def move_object(id, position, rotation):
    """Moves object sphere in visual environment"""
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


def noise_and_sort(gts, rs, nds, nrs, t_vecs):
    X_fin, y_fin, X_rot_fin, y_rot_fin = [], [], [], []
    for j in range(len(gts)):
        Xc, yc, X_rotc, y_rotc = nds[j], gts[j], nrs[j], rs[j]
        #print("gt", j, ":", gts[j])
        t_vec = t_vecs[j]
        dft = pd.DataFrame(t_vec)
        dft.to_csv("./testing tables/t_vec.csv", index=False)
        Xc, yc, X_rotc, y_rotc = append_t_vec_and_class(Xc, yc, X_rotc, y_rotc, t_vec, j)
        X_fin.append(Xc)
        y_fin.append(yc)
        X_rot_fin.append(X_rotc)
        y_rot_fin.append(y_rotc)


    X = np.vstack((tuple(X_fin)))
    X_rot = np.vstack((tuple(X_rot_fin)))
    y_rot = np.vstack((tuple(y_rot_fin)))
    y = np.vstack((tuple(y_fin)))
    df = pd.DataFrame(y)
    df.to_csv("./testing tables/y_fin.csv", index=False)
    X = X[X[:, 3].argsort()]
    y = y[y[:, 3].argsort()]
    X_rot = X_rot[X_rot[:, 4].argsort()]
    y_rot = y_rot[y_rot[:, 4].argsort()]
    return X, y, X_rot, y_rot

def append_t_vec_and_class(X, y, X_rot, y_rot, t_vec, class_):
    #print("X_rot:", X_rot)
    #print("t_vec", t_vec)
    t_vec = np.reshape(t_vec, (X.shape[0], 1))
    #print("reshaped t_vec", t_vec)
    class_vec = np.full_like(t_vec, class_)
    X = np.append(X, t_vec, axis=1)
    y = np.append(y, t_vec, axis=1)
    X_rot = np.append(X_rot, t_vec, axis=1)
    y_rot = np.append(y_rot, t_vec, axis=1)
    X = np.append(X, class_vec, axis=1)
    y = np.append(y, class_vec, axis=1)
    X_rot = np.append(X_rot, class_vec, axis=1)
    y_rot = np.append(y_rot, class_vec, axis=1)
    return X, y, X_rot, y_rot

def convert_into_final_filter(X, y, X_rot, y_rot):
    print("started converting")
    num_objects = int(X.shape[1]/4)
    Xc, yc, X_rotc, y_rotc = create_regular_timestamps(X, num_objects)
    for i in range(num_objects):
        X_cut= X[:, i*4: (1+i)*4]
        y_cut = y[:, i*4: (1+i)*4]
        X_rot_cut = X_rot[:, i*5:(i+1)*5]
        y_rot_cut = y_rot[:, i*5:(i+1)*5]

        indexes = np.full((X.shape[0], 1), i)
        X_current = np.append(X_cut, indexes, axis =1)
        y_current = np.append(y_cut, indexes, axis =1)
        X_rot_current = np.append(X_rot_cut, indexes, axis =1)
        y_rot_current = np.append(y_rot_cut, indexes, axis =1)
        Xc = np.vstack((Xc, X_current))
        yc = np.vstack((yc, y_current))
        X_rotc = np.vstack((X_rotc, X_rot_current))
        y_rotc = np.vstack((y_rotc, y_rot_current))
    Xc = Xc[Xc[:, 3].argsort()]
    yc = yc[yc[:, 3].argsort()]
    X_rotc = X_rotc[X_rotc[:, 4].argsort()]
    y_rotc = y_rotc[y_rotc[:, 4].argsort()]
    print("ended converting")
    return Xc, yc, X_rotc, y_rotc


def create_regular_timestamps(X_orig, num_objects):
    timestamps = []
    for i in range(0, X_orig.shape[0], 1):
        t = 0.2*(i+1)
        timestamps.append(t)
    timestamps = np.array(timestamps)
    timestamps = np.expand_dims(timestamps, axis=1)
    X, y, X_rot, y_rot = None, None, None, None
    for i in range(num_objects):
        X_tmp = np.full((X_orig.shape[0], 3), 88)
        y_tmp = np.full((X_orig.shape[0], 3), 88)
        X_rot_tmp = np.full((X_orig.shape[0], 4), 88)
        y_rot_tmp = np.full((X_orig.shape[0], 4), 88)

        # Adding column with timestamp
        X_tmp = np.append(X_tmp, timestamps, axis=1)
        y_tmp = np.append(y_tmp, timestamps, axis=1)
        X_rot_tmp = np.append(X_rot_tmp, timestamps, axis=1)
        y_rot_tmp = np.append(y_rot_tmp, timestamps, axis=1)

        # Adding column determining object index
        indexes = np.full((X_orig.shape[0], 1), i)
        X_tmp = np.append(X_tmp, indexes, axis=1)
        y_tmp = np.append(y_tmp, indexes, axis=1)
        X_rot_tmp = np.append(X_rot_tmp, indexes, axis=1)
        y_rot_tmp = np.append(y_rot_tmp, indexes, axis=1)
        if X is None:
            X = X_tmp
            y = y_tmp
            X_rot = X_rot_tmp
            y_rot = y_rot_tmp
        else:
            X = np.vstack((X, X_tmp))
            y = np.vstack((y, y_tmp))
            X_rot = np.vstack((X_rot, X_rot_tmp))
            y_rot = np.vstack((y_rot, y_rot_tmp))
    print("timestamps shapes:", X.shape, y.shape, X_rot.shape, y_rot.shape)
    return X, y, X_rot, y_rot



def convert_X_y_into_vis_data(X, y, X_rot, y_rot, num_trajectories):
    """
        Works for MOT - converts arrays X and y with multiple trajectories into a list of trajectories
    """
    trajectories = num_trajectories*[np.zeros(3)]
    rot_trajectories = num_trajectories*[np.zeros(4)]
    noisy_trajectories = num_trajectories*[np.zeros(3)]
    noisy_rot_trajectories = num_trajectories*[np.zeros(4)]
    print("num_trajectories:", num_trajectories)
    for i in range(X.shape[0]):
        traj_idx = int(y[i, 4]) #Class info
        if np.allclose(trajectories[traj_idx], [0, 0, 0]):
            trajectories[traj_idx] = y[i, :3]
            rot_trajectories[traj_idx] = y_rot[i, :4]
            noisy_trajectories[traj_idx] = X[i, :3]
            noisy_rot_trajectories[traj_idx] = X_rot[i, :4]
        else:
            trajectories[traj_idx] = np.vstack((trajectories[traj_idx], y[i, :3]))
            rot_trajectories[traj_idx] = np.vstack((rot_trajectories[traj_idx],y_rot[i, :4]))
            noisy_trajectories[traj_idx] = np.vstack((noisy_trajectories[traj_idx], X[i, :3]))
            noisy_rot_trajectories[traj_idx] = np.vstack((noisy_rot_trajectories[traj_idx], X_rot[i, :4]))
    return trajectories, rot_trajectories, noisy_trajectories, noisy_rot_trajectories


def convert_X_y_into_vis_data_old(X, y, X_rot = None, y_rot= None):
    """
    Works for MOT - converts arrays X and y with multiple trajectories into a list of trajectories
    """
    trajectories, rot_trajectories, noisy_trajectories, noisy_rot_trajectories = [], [], [], []
    num_trajectories = int(y.shape[1]/4)
    print("trajectory amount:", num_trajectories)
    for i in range(num_trajectories):
        trajectories.append(y[:, 4*i:4*i+3])
        rot_trajectories.append(y_rot[:, 5*i:5*i+4])
        noisy_trajectories.append(X[:, 4*i:4*i+3])
        noisy_rot_trajectories.append(X_rot[:, 5*i:5*i+4])
    return trajectories, rot_trajectories, noisy_trajectories, noisy_rot_trajectories


def determine_traj_amount(gt):
    classes = set()
    for i in range(gt.shape[0]):
        class_ = gt[i, 4]
        classes.add(class_)
    return len(classes)


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


def visualize_multiple_trajectories(gts, nds):
    ids, noise_ids, cube_ids = [], [], []
    for i in range(len(gts)):
        id, noise_id, cube_id = visualize_trajectory_timesteps(gts[i], nds[i])
        ids.extend(id)
        noise_ids.extend(noise_id)
        cube_ids.extend(cube_id)
    return ids, noise_ids, cube_ids


def load_trajectory(type):
    fn_gt = "./dataset/visualizer_trajectories/" + type +".npy"
    fn_nd = "./dataset/visualizer_trajectories/" + type + "_noise.npy"
    fn_r = "./dataset/visualizer_trajectories/" + type + "_rot.npy"
    fn_nr = "./dataset/visualizer_trajectories/" + type + "_rot_noise.npy"
    ground_truth = np.delete(np.load(fn_gt),-1, 0)
    noisy_data = np.delete(np.load(fn_nd), -1, 0)
    rotations = np.delete(np.load(fn_r), -1 , 0)
    noisy_rotations = np.delete(np.load(fn_nr), -1, 0)
    return ground_truth, rotations, noisy_data, noisy_rotations


def load_MOT_scenario():
    """Loads predefined MOT set of trajectories"""
    y = np.load("./dataset/MOT/ground_truth.npy")
    X = np.load("./dataset/MOT/noisy_data.npy")
    y_rot = np.load("./dataset/MOT/rotations.npy")
    X_rot = np.load("./dataset/MOT/noisy_rotations.npy")
    return y, X, y_rot, X_rot



