"""
This script serves the purpose of visualizing generated (noisy) trajectory of object along with its
filtered result. Filtering is done through the Kalman and particle filtering algorithm.
"""

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
from myGym.envs.particle_filter import ParticleFilter3D
from myGym.utils.filter_helpers import *


#Global variables
velocity = 0.5 #Average object velocity in m/s
dt = 0.1 #Legth of one timestep in seconds

"""

Input retreiver and other necessary initialization functions:

"""


def get_parameters_from_config(type):
    """
    Load trajectory_config.json and create correct trajectory dict based on user input.
    """
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
    """Method to retreive user input. Returns dictionary with trajectory parameters."""
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
    trajectory_dict["vis"] = input("Do you want to visualize the filter process? (1 - yes, 2 - no) ")
    if trajectory_dict["vis"] == '1':
        trajectory_dict["pause"] = input("Enter the length of each filter iteration (seconds): ")
    return trajectory_dict

"""

Trajectory creator:

"""

def create_trajectory_points(parameters):
    """From retreived trajectory parameters, create set of n points on the trajectory along with noisy data.
    Also create a list of rotations (quaternion objects) and noisy rotations."""
    ground_truth = np.zeros((20, 3)) #20 is an arbitrary value just for initialization
    noisy_data = np.zeros((20, 3))#Will be replaced based on trajectory length and average velocity
    new_n = 0
    if parameters["type"] == None:
        #Empty input
        ground_truth, noisy_data, rotations, noisy_rotations = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    elif parameters["type"] == "line":
        #Ground truth position points line
        start = np.array(list(map(float, parameters["start"].split())))
        end = np.array(list(map(float, parameters["end"].split())))
        len = np.linalg.norm(end - start)
        new_n = int(len/(velocity * dt)) #0.1 is dt
        ground_truth = np.linspace(start, end, new_n)

    elif parameters["type"] == "circle":
        # Ground truth position points circle
        radius = float(parameters["radius"])
        center = np.array(list(map(float, parameters["center"].split())))
        norm = np.array(list(map(float, parameters["plane"].split())))
        len = 2*np.pi*radius
        new_n = int(len/(velocity * dt)) #0.1 is dt
        ground_truth = create_circle(radius, center, norm, new_n)

    elif parameters["type"] == "spline":
        #Ground truth position points spline
        points_amount = len(parameters["points"])
        point_list = np.zeros((points_amount, 3))
        for i in range(points_amount):
            point_list[i, :] = np.array(list(map(float, parameters["points"][i].split())))
        ground_truth[:, :3] = fit_polynomial(point_list, 40)#Initialization of spline with arbitrary amount of points
        new_n = int(trajectory_length(ground_truth)/(velocity * dt))#
        ground_truth = fit_polynomial(point_list, new_n)

    else:
        ground_truth = np.zeros((n, 3))


    if parameters["type"] is not None:
        # Ground truth rotation quaternions
        q1 = np.array(list(map(float, parameters["q_start"].split())))
        q2 = np.array(list(map(float, parameters["q_end"].split())))
        q_start = Quaternion(axis=q1[:3], angle=np.deg2rad(q1[3]))
        q_end = Quaternion(axis=q2[:3], angle=np.deg2rad(q2[3]))
        rotations = []
        for rot in Quaternion.intermediates(q_start, q_end, new_n):
            rotations.append(rot)
        sigma_q = np.deg2rad(float(parameters["q_stdv"]))
        sigma = float(parameters["stdv"])

        #Adding noise to position and rotation
        noisy_data = add_noise(ground_truth, float(sigma))
        noisy_rotations = add_rotation_noise(rotations, sigma_q)


    return ground_truth, noisy_data, rotations, noisy_rotations


"""

Evaluation plot functions:

"""

def visualize_errors(ground_truth, noisy_data, estimates):
    """
    Visualization of position errors in a plot.
    """
    diffs_meas = []
    diffs_est = []
    print("Lengths: ")
    print("Gnd truth: ", ground_truth.shape[0], "estimates:", len(estimates))
    for i in range(ground_truth.shape[0]):
        diffs_meas.append(np.linalg.norm(ground_truth[i, :] - noisy_data[i, :]))
        diffs_est.append(np.linalg.norm(ground_truth[i, :] - estimates[i]))
    diffs_meas = np.array(diffs_meas)
    plot_comparison(diffs_meas, diffs_est, "Position error evaluation")


def plot_comparison(diffs_meas, diffs_est, title):
    """Plots how the filter performed. Compares the error (difference between ground truth) of measured and estimated
    data."""
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
    for i in range(len(rotations)):

        diffs_meas.append(Quaternion.absolute_distance(rotations[i], noisy_rotations[i]))
        diffs_est.append(Quaternion.absolute_distance(rotations[i], euler_to_quat(estimates[i])))
    plot_comparison(diffs_meas, diffs_est, "Rotation error evaluation")


"""

Filtering process and its visualization functions:

"""

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


def vis_anim(ground_truth, noisy_data, rotations, noisy_rotations, pause_length):
    """
    Parameters: ground_truth: np array of shape (n, 3)
    """
    env.p.setGravity(0, 0, 0)
    n = ground_truth.shape[0] #Number of points
    iter = 0
    position_filter = ParticleFilter3D(1500, 0.02, 0.025, g = 0.7, h= 0.4)
    rotation_filter = ParticleFilter3D(1500, np.deg2rad(1), np.deg2rad(4), g= 0.8, h = 0.8)
    particle_batch = None
    estimate_id = None

    for i in range(n):
        measurement = noisy_data[i, :]
        rot_meas = quat_to_euler(noisy_rotations[i])
        env.task_objects["actual_state"].set_position(ground_truth[i, :])
        env.task_objects["goal_state"].set_position(measurement)

        if i == 0:
            #Initialize particle filter based on first measurement
            position_filter.apply_first_measurement(measurement)
            particle_batch = visualize_particles(position_filter)
            estimate_id = visualize_estimate(position_filter)
            rotation_filter.apply_first_measurement(rot_meas)
            rotation_filter.state_estimate()
            continue

        #1) Predict movement
        position_filter.predict()
        move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)

        #2) Update based on measurement
        position_filter.update(measurement)
        move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)

        #3) Compute estimate
        position_filter.state_estimate()
        move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)
        move_estimate(estimate_id, position_filter.estimate)
        time.sleep(pause_length)

        #4) Resample particles
        position_filter.resample()
        move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)

        #5) Rotation filter complete step (visualization of rotation particles doesn't make sense)
        rotation_filter.filter_step(rot_meas)
        rot = rotations[i]
        noisy_rot = noisy_rotations[i]
        q = np.concatenate((rot.imaginary, [rot.scalar])).flatten()
        noisy_q = np.concatenate((noisy_rot.imaginary, [noisy_rot.scalar])).flatten()
        est_rot = euler_to_quat(rotation_filter.estimate)
        est_q = np.concatenate((est_rot.imaginary, [est_rot.scalar])).flatten()
        p.resetBasePositionAndOrientation(estimate_id, position_filter.estimate, est_q)
        env.task_objects["actual_state"].set_orientation(q)
        env.task_objects["goal_state"].set_orientation(noisy_q)

        action = [0, 0, 0]
        obs, reward, done, info = env.step(action) #Necessary step for animation to continue
        time.sleep(0.001)
    return position_filter


if __name__ == "__main__":
    params = get_input()
    ground_truth, noisy_data, rotations, noisy_rotations = create_trajectory_points(params)
    arg_dict = get_arg_dict()
    if params["vis"] == "1":
        env = visualize_env(arg_dict)
        env.reset()
        visualize_trajectory(ground_truth, noisy_data)
        resulting_pos_filter = vis_anim(ground_truth, noisy_data, rotations, noisy_rotations, float(params["pause"]))
        visualize_errors(ground_truth, noisy_data, resulting_pos_filter.estimates)
    else:
        resulting_pos_filter, resulting_rot_filter = filter_without_animation(noisy_data, noisy_rotations)
        visualize_errors(ground_truth, noisy_data, resulting_pos_filter.estimates)
        visualize_rot_errors(rotations, noisy_rotations, resulting_rot_filter.estimates)



