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
from myGym.envs.particle_filter import ParticleFilter3D

#Global variable
velocity = 0.5

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
        trajectory_dict["stdv"] = input("Enter position noise standard deviation in meters: ")
        trajectory_dict["q_start"] = input("Enter starting rotation in axis-angle representation (format 'x y z theta')")
        trajectory_dict["q_end"] = input("Enter ending rotation in axis-angle representation [rad] (format 'x y z theta')")
        trajectory_dict["q_stdv"] = input("Enter rotation noise standard deviation in radians")
    else:
        type = int(input("Enter trajectory type (1 for line, 2 for circle, 3 for spatial spline: )"))
        trajectory_dict = get_parameters_from_config(type)

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


def trajectory_length(trajectory):
    """Compute the total length of the trajectory"""
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
    #print("Trajectory = ", trajectory)
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
        quat_noisy = Quaternion(imaginary = q_noisy[:3], real = q_noisy[3])
        noisy_rotations.append(quat_noisy)
    return noisy_rotations


def create_trajectory_points(parameters, n):
    #TODO: From retreived trajectory parameters, create set of n 6D points on the trajectory along with noisy data.
    ground_truth = np.zeros((n, 3))
    noisy_data = np.zeros((n, 3))
    new_n = 0
    if parameters["type"] == None:
        ground_truth, noisy_data, rotations, noisy_rotations = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    elif parameters["type"] == "line":
        #Ground truth position points line
        start = np.array(list(map(float, parameters["start"].split())))
        end = np.array(list(map(float, parameters["end"].split())))
        ground_truth[:, :3] = np.linspace(start, end, n)
        #trajectory_length(ground_truth) > n*0.1*velocity:
        new_n = int(trajectory_length(ground_truth)/(velocity * 0.1))
        ground_truth = np.linspace(start, end, new_n)

    elif parameters["type"] == "circle":
        # Ground truth position points circle
        radius = float(parameters["radius"])
        center = np.array(list(map(float, parameters["center"].split())))
        norm = np.array(list(map(float, parameters["plane"].split())))
        ground_truth[:, :3] = create_circle(radius, center, norm, n)
        #if trajectory_length(ground_truth) > n*0.1*velocity:
        new_n = int(trajectory_length(ground_truth)/(velocity * 0.1))
        ground_truth = create_circle(radius, center, norm, new_n)

    elif parameters["type"] == "spline":
        points_amount = len(parameters["points"])
        point_list = np.zeros((points_amount, 3))
        for i in range(points_amount):
            point_list[i, :] = np.array(list(map(float, parameters["points"][i].split())))
        ground_truth[:, :3] = fit_polynomial(point_list, n)
        #if trajectory_length(ground_truth) > n*0.1*velocity:
        new_n = int(trajectory_length(ground_truth)/(velocity * 0.1))
        ground_truth = fit_polynomial(point_list, new_n)

    else:
        ground_truth = np.zeros((n, 3))



    if parameters["type"] is not None:
        # Ground truth rotation quaternions
        #print("type:", parameters["type"])
        q1 = np.array(list(map(float, parameters["q_start"].split())))
        q2 = np.array(list(map(float, parameters["q_end"].split())))
        q_start = Quaternion(axis=q1[:3], angle=np.deg2rad(q1[3]))
        q_end = Quaternion(axis=q2[:3], angle=np.deg2rad(q2[3]))
        rotations = []
        for rot in Quaternion.intermediates(q_start, q_end, new_n):
            rotations.append(rot)
        #print("ROTATIONS:", rotations)
        sigma_q = np.deg2rad(float(parameters["q_stdv"]))
        sigma = float(parameters["stdv"])
        #Adding noise to position and rotation
        noisy_data = add_noise(ground_truth, float(sigma))
        noisy_rotations = add_rotation_noise(rotations, sigma_q)


    return ground_truth, noisy_data, rotations, noisy_rotations


def visualize_errors(ground_truth, noisy_data, estimates):
    """
    visualization of position errors
    """
    diffs_meas = []
    diffs_est = []
    for i in range(ground_truth.shape[0]):
        diffs_meas.append(np.linalg.norm(ground_truth[i, :] - noisy_data[i, :]))
        diffs_est.append(np.linalg.norm(ground_truth[i, :] - estimates[i]))
    diffs_meas = np.array(diffs_meas)
    plot_comparison(diffs_meas, diffs_est, "Position error evaluation")


def plot_comparison(diffs_meas, diffs_est, title):
    diffs_meas = np.array(diffs_meas)
    k = np.arange(len(diffs_meas))
    measured_avg = np.average(diffs_meas)
    estimated_avg = np.average(diffs_est)

    # Plotting results
    fig, ax = plt.subplots()
    ax.text(.01, .95, 'Measurement error average: ' + str(round(measured_avg, 5)), transform=ax.transAxes)
    plt.text(.01, .9, 'Estimate error average: ' + str(round(estimated_avg, 5)), transform=ax.transAxes)
    ax.plot(k, diffs_est, label="Estimate error")
    plt.plot(k, diffs_meas, label="Measurement error")
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_rot_errors(rotations, noisy_rotations, estimates):
    """
    Visualization of rotation errors
    """
    diffs_meas = []
    diffs_est = []
    #print("Ground truth rotations:", rotations)
    #print("Estimated rotations:", estimates)
    #for i in range(10, len(rotations), 1):
        #print("timestep", i ,": truth rotation:", np.rad2deg(quat_to_euler(rotations[i])), "|", "quaternion:", rotations[i])
        #print()
    for i in range(len(rotations)):
        diffs_meas.append(Quaternion.absolute_distance(rotations[i], noisy_rotations[i]))
        diffs_est.append(Quaternion.absolute_distance(rotations[i], euler_to_quat(estimates[i])))
    plot_comparison(diffs_meas, diffs_est, "Rotation error evaluation")


def polynomial(coeffs):
    """Return function which calculates polynomial of given coefficients at x"""
    return lambda x: sum(a*x**i for i, a in enumerate(coeffs))


def move_particles(ids, positions):
    """Removes particle batch from visualization"""
    for i in range(len(ids)):
        p.resetBasePositionAndOrientation(ids[i], positions[i, :], [1, 0, 0, 0])


def move_estimate(id, position):
    """Moves estimate sphere in visual environment"""
    p.resetBasePositionAndOrientation(id, position, [1, 0, 0, 0])


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


def filter_without_animation(noisy_data, noisy_rotations):
    """Similar to vis_anim but without any visualization"""
    position_filter = ParticleFilter3D(20000, 0.02, 0.02, g= 0.7, h = 0.4)
    rotation_filter = ParticleFilter3D(20000, np.deg2rad(1), np.deg2rad(4), g= 0.8, h = 0.8)
    n = noisy_data.shape[0] #Number of points
    for iter in range(n):
        measurement = noisy_data[iter, :]
        rot_meas = quat_to_euler(noisy_rotations[iter])
        if iter == 0:
            position_filter.apply_first_measurement(measurement)
            position_filter.state_estimate()
            rotation_filter.apply_first_measurement(rot_meas)
            rotation_filter.state_estimate()
            continue
        rotation_filter.filter_step(rot_meas)
        position_filter.filter_step(measurement)

    return position_filter, rotation_filter


def vis_anim(ground_truth, noisy_data, rotations, noisy_rotations):
    """
    Parameters: ground_truth: np array of shape (n, 3)
    """
    env.p.setGravity(0, 0, 0)
    n = ground_truth.shape[0] #Number of points
    iter = 0
    position_filter = ParticleFilter3D(1500, 0.02, 0.025, g = 0.7, h= 0.4)
    last_particle_batch = None
    estimate_id = None
    print("Goal_id:", env.task_objects["goal_state"].uid)
    print("Actual state id:", env.task_objects["actual_state"].uid)
    measurements = []
    for i in range(50):
        #arg_dict["max_episode_steps"]
        #env.task_objects["actual_state"].setGravity()
        for e in range(20000):
            if e%2 == 0:
                if iter >= n:
                    break
                env.task_objects["actual_state"].set_position(ground_truth[iter, :])
                env.task_objects["goal_state"].set_position(noisy_data[iter, :])

                if len(measurements) == 0:
                    position_filter.apply_first_measurement(noisy_data[iter, :])
                    last_particle_batch = visualize_particles(position_filter)
                    measurements.append(position_filter.particles)
                    estimate_id = visualize_estimate(position_filter)
                    continue
                #1) Predict movement
                position_filter.predict()
                move_particles(last_particle_batch, position_filter.particles)
                time.sleep(0.5)
                #2) Update based on measurement
                position_filter.update(noisy_data[iter, :])
                move_particles(last_particle_batch, position_filter.particles)
                time.sleep(0.5)
                #3) Compute estimate
                position_filter.state_estimate()
                move_particles(last_particle_batch, position_filter.particles)
                time.sleep(0.5)
                move_estimate(estimate_id, position_filter.estimate)
                time.sleep(0.5)
                #4) Resample particles
                position_filter.resample()
                move_particles(last_particle_batch, position_filter.particles)
                time.sleep(0.5)

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

    return position_filter
        #print(env.task_objects)


def visualize_estimate(particle_filter):
    position = particle_filter.estimate
    shape_id = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02, rgbaColor = [0.2, 0.2, 0.2, 0.5])
    particle_id = p.createMultiBody(baseVisualShapeIndex = shape_id, basePosition = position)
    return particle_id


def visualize_trajectory(ground_truth, noisy_data):
    if np.linalg.norm(ground_truth) > 0.1:
        for i in range(ground_truth.shape[0] - 1):
            p.addUserDebugLine(noisy_data[i, :], noisy_data[i+1, :])
            p.addUserDebugLine(ground_truth[i, :], ground_truth[i+1, :])


def visualize_particles(particle_filter):
    positions = particle_filter.particles.tolist()
    #p.addUserDebugPoints(pointPositions = positions, pointColorsRGB=[20, 20, 20])
    particle_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.001)

    #for i in range(particle_filter.particles.shape[0]):
        #position = particle_filter.particles[i, :]
        #p.createMultiBody(baseVisualShapeIndex = particle_id, basePosition = position)
    particles_id = p.createMultiBody(baseVisualShapeIndex=particle_id, batchPositions=positions)
    return particles_id


if __name__ == "__main__":
    params = get_input()
    ground_truth, noisy_data, rotations, noisy_rotations = create_trajectory_points(params, 20)
    arg_dict = get_arg_dict()
    #env = visualize_env(arg_dict)

    #env.reset()
    #visualize_trajectory(ground_truth, noisy_data)
    #time.sleep(15)
    #visualize_errors(ground_truth, noisy_data)
    #visualize_rot_errors(rotations, noisy_rotations)
    resulting_pos_filter, resulting_rot_filter = filter_without_animation(noisy_data, noisy_rotations)

    #resulting_pos_filter = vis_anim(ground_truth, noisy_data, rotations, noisy_rotations)
    visualize_errors(ground_truth, noisy_data, resulting_pos_filter.estimates)
    visualize_rot_errors(rotations, noisy_rotations, resulting_rot_filter.estimates)
    #sys.exit()


