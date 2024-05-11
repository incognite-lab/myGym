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
from myGym.envs.particle_filter import *
from myGym.utils.filter_helpers import *

from myGym.envs.trajectory_generator import *

#Global variables
velocity = 0.2 #Average object velocity in m/s
dt = 0.2 #Legth of one timestep in seconds

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
    else:
        traj = args["positions"][2]
        trajectory_dict["type"] = "line_acc"
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
            trajectory_dict["type"] = "line_acc"

        trajectory_dict["stdv"] = input("Enter position noise standard deviation in meters: ")
        trajectory_dict["q_start"] = input("Enter starting rotation in axis-angle representation (format 'x y z theta')")
        trajectory_dict["q_end"] = input("Enter ending rotation in axis-angle representation [rad] (format 'x y z theta')")
        trajectory_dict["q_stdv"] = input("Enter rotation noise standard deviation in radians")

    else:
        type = int(input("Enter trajectory type (1 for line, 2 for circle, 3 for spatial spline: )"))
        trajectory_dict = get_parameters_from_config(type)
    trajectory_dict["filter_type"] = input("Enter the type of particle filter (1 for g-h, 2 for 6D, 3 for 3D with Kalman, 4 for simple Kalman")
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
    ground_truth = np.zeros((40, 3)) #40 is an arbitrary value just for initialization
    noisy_data = np.zeros((40, 3))#Will be replaced based on trajectory length and average velocity
    new_n = 0
    if parameters["type"] == None:
        #Empty input
        ground_truth, noisy_data, rotations, noisy_rotations = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    elif parameters["type"] == "line":
        #Ground truth position points line
        start = np.array(list(map(float, parameters["start"].split())))
        end = np.array(list(map(float, parameters["end"].split())))
        length = np.linalg.norm(end - start)
        new_n = int(length/(velocity * dt)) #0.1 is dt
        ground_truth = np.linspace(start, end, new_n)

    elif parameters["type"] == "circle":
        #Ground truth position points circle
        radius = float(parameters["radius"])
        center = np.array(list(map(float, parameters["center"].split())))
        norm = np.array(list(map(float, parameters["plane"].split())))
        length = 2*np.pi*radius
        new_n = int(length/(velocity * dt)) #0.1 is dt
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

def visualize_errors(ground_truth, noisy_data, estimates, filename):
    """
    Visualization of position errors in a plot.
    """
    diffs_meas = []
    diffs_est = []
    for i in range(ground_truth.shape[0]):
        diffs_meas.append(np.linalg.norm(ground_truth[i, :] - noisy_data[i, :])**2)
        diffs_est.append(np.linalg.norm(ground_truth[i, :] - estimates[i])**2)
    diffs_meas = np.array(diffs_meas)
    plot_comparison(diffs_meas, diffs_est, "Position error evaluation", filename)


def plot_comparison(diffs_meas, diffs_est, title, filename):
    """Plots how the filter performed. Compares the error (difference between ground truth) of measured and estimated
    data."""
    diffs_meas = np.array(diffs_meas)
    k = np.arange(len(diffs_meas))
    measured_avg = np.average(diffs_meas)/3
    estimated_avg = np.average(diffs_est)/3

    # Plotting results
    fig, ax = plt.subplots()
    ax.text(.01, .95, 'Measurement MSE: ' + str(round(measured_avg, 5)), transform=ax.transAxes)
    plt.text(.01, .9, 'Estimate MSE: ' + str(round(estimated_avg, 5)), transform=ax.transAxes)
    ax.plot(k, diffs_est, label="Estimate SE")
    plt.plot(k, diffs_meas, label="Measurement SE")
    plt.title(title)
    plt.legend()
    #plt.savefig('/home/frederik/school/projekt+BP/figures/' + filename + ".png")
    plt.show()


def visualize_rot_errors(rotations, noisy_rotations, estimates, filename):
    """
    Visualization of rotation errors
    """
    diffs_meas = []
    diffs_est = []
    for i in range(len(rotations)):
        diffs_meas.append(Quaternion.absolute_distance(rotations[i], noisy_rotations[i])**2)
        diffs_est.append(Quaternion.absolute_distance(rotations[i], Quaternion(estimates[i]))**2)
    plot_rot_marginals(rotations, noisy_rotations, estimates)
    plot_comparison(diffs_meas, diffs_est, "Rotation error evaluation", filename)



def plot_rot_marginals(rotations, noisy_rotations, estimates):
    """Creates 4 plots for each Quaternion element comparing ground truth, measurements and estimates"""

    k = np.arange(len(rotations))
    fig, axs = plt.subplots(2, 2)
    gt = np.zeros((len(rotations), 4))
    meas = np.zeros((len(rotations), 4))
    est = np.zeros((len(rotations), 4))
    for i in range(len(rotations)):
        gt[i, :] = rotations[i].elements
        meas[i, :] = noisy_rotations[i].elements
        est[i, :] = estimates[i]
    multiplot_func(gt, meas, est, axs, k)
    #plt.plot(k, diffs_meas, label="Measurement SE")
    #plt.title(title)
    plt.legend()
    plt.show()


def multiplot_func(gt, meas, est, axs, k):
    axs[0, 0].plot(k, gt[:, 0], label="Ground truth w")
    axs[0, 0].plot(k, meas[:, 0], label="Measurement w")
    axs[0, 0].plot(k, est[:, 0], label="Estimated w")
    axs[0, 1].plot(k, gt[:, 1], label="Ground truth x")
    axs[0, 1].plot(k, meas[:, 1], label="Measurement x")
    axs[0, 1].plot(k, est[:, 1], label="Estimated x")
    axs[1, 0].plot(k, gt[:, 2], label="Ground truth y")
    axs[1, 0].plot(k, meas[:, 2], label="Measurement y")
    axs[1, 0].plot(k, est[:, 2], label="Estimated y")
    axs[1, 1].plot(k, gt[:, 3], label="Ground truth z")
    axs[1, 1].plot(k, meas[:, 3], label="Measurement z")
    axs[1, 1].plot(k, est[:, 3], label="Estimated z")


def create_paramstring(params, filter):
    """Creates a name for saved figure based on tested trajectory and filter parameters."""
    filename = ""
    filename += params["type"] + "_"
    if params["filter_type"] == "1":
        #g-h filter
        filename += "gh_" #filter type
        filename += "g:" + str(filter.g) + "_" #filter g value
        filename += "h:" + str(filter.h) + "_" #filter h value
    elif params["filter_type"] == "2":
        #6D filter
        filename += "6D_" #filter type
        filename += "v-std:" + str(filter.vel_std) + "_"
    #TODO: Cases for Kalman and 3D+ Kalman filters
    #General parameters
    filename += "std_p:" + str(filter.process_std) + "_" #Process standard deviation of noise
    filename += "std_m:" + str(filter.measurement_std) + "_" #Measurement standard deviation of noise
    filename += "N:" + str(filter.num_particles) + "_" #Amount of particles
    return filename

"""

Filtering process and its visualization functions:

"""

def initialize_filter(type):
    """Filter initialization based on user input. Parameters can be changed here manually"""
    position_filter, rotation_filter = None, None
    if type == "1": #Particle GH
        position_filter = ParticleFilterGH(2500, 0.02, 0.02, g= 0.7, h = 0.4)
        rotation_filter = ParticleFilterGHRot(2500, np.deg2rad(1), np.deg2rad(4), g= 0.8, h = 0.8)
    elif type == "2": #Particle 3D
        position_filter = ParticleFilter6D(600, 0.02, 0.1, 0.02)
        rotation_filter = ParticleFilter6DRot(2500, 0.02, np.deg2rad(1), np.deg2rad(4))
    elif type == "3":
        position_filter = ParticleFilterWithKalman(2500, 0.02, 0.02, Q= 0.01/dt)
        rotation_filter = ParticleFilterWithKalmanRot(2500, np.deg2rad(1), np.deg2rad(4))
    elif type == "4": #Kalman
        x = np.array([0., 0., 0., 0., 0., 0.]) # [x, dx/dt, y, dy/dt, z, dz/dt]
        xr = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        #P = np.diag([100., 25., 64, 0.1, 0.1, 0.1])
        Q = 0.08
        R = 0.02
        var_angle = np.deg2rad(90)
        var_angle_vel = np.deg2rad(20)
        Pr= np.diag([var_angle, var_angle_vel, var_angle, var_angle_vel, var_angle, var_angle_vel, var_angle, var_angle_vel])
        Qr = np.deg2rad(4)
        Rr = np.deg2rad(4)
        position_filter = myKalmanFilter(x, P=P, Q=Q, R=R)
        rotation_filter = myKalmanFilterRot(xr, P=Pr, Q=Qr, R=Rr)
    else:
        print("Entered wrong input type")
        sys.exit()
    return position_filter, rotation_filter


def filter_without_animation(noisy_data, noisy_rotations, type):
    """Similar to vis_anim but without any visualization"""
    position_filter, rotation_filter = initialize_filter(type)
    n = noisy_data.shape[0] #Number of points
    for iter in range(n):
        measurement = noisy_data[iter, :]
        rot_meas = noisy_rotations[iter]
        if iter == 0:
            position_filter.apply_first_measurement(measurement)
            position_filter.state_estimate()
            rotation_filter.apply_first_measurement(rot_meas.elements)
            rotation_filter.state_estimate()
            continue
        rotation_filter.filter_step(rot_meas.elements)
        position_filter.filter_step(measurement)
    return position_filter, rotation_filter


def vis_anim(ground_truth, noisy_data, rotations, noisy_rotations, pause_length, type):
    """
    Parameters: ground_truth: np array of shape (n, 3)
    """
    env.p.setGravity(0, 0, 0)
    n = ground_truth.shape[0] #Number of points
    #position_filter = ParticleFilter6D(1500, 0.00, 0.2, 0.025)
    #rotation_filter = ParticleFilter6D(1500, np.deg2rad(0), np.deg2rad(0.5), np.deg2rad(4))
    position_filter, rotation_filter = initialize_filter(type)
    particle_batch = None
    estimate_id = None

    for i in range(n):
        measurement = noisy_data[i, :]
        rot_meas = noisy_rotations[i]
        env.task_objects["actual_state"].set_position(ground_truth[i, :])
        env.task_objects["goal_state"].set_position(measurement)

        if i == 0:
            #Initialize particle filter based on first measurement
            position_filter.apply_first_measurement(measurement)
            position_filter.state_estimate()
            if type != "4":
                particle_batch = visualize_particles(position_filter)
            estimate_id = visualize_estimate(position_filter)
            rotation_filter.apply_first_measurement(rot_meas.elements)
            rotation_filter.state_estimate()
            continue

        #1) Predict movement
        position_filter.predict()
        if type != "4":
            move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)

        #2) Update based on measurement
        position_filter.update(measurement)
        if type != "4":
            move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)

        do_resample = True
        #3) Compute estimate
        if type != "4":
            if position_filter.neff() < position_filter.num_particles/30:
                position_filter.resample()
                position_filter.reapply_measurement(measurement)
                do_resample = False
        position_filter.state_estimate()
        if type != "4":
            move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)
        move_estimate(estimate_id, position_filter.estimate, rotation_filter.estimate)
        time.sleep(pause_length)

        #4) Resample particles
        if type != "4":
            if position_filter.neff() < position_filter.num_particles/2 and do_resample:
                position_filter.resample()
            move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)

        #5) Rotation filter complete step (visualization of rotation particles doesn't make sense)
        rot = rotations[i]
        noisy_rot = noisy_rotations[i]
        q = [rot.elements[3], rot.elements[0], rot.elements[1], rot.elements[2]]
        noisy_q = [noisy_rot.elements[3], noisy_rot.elements[0], noisy_rot.elements[1], noisy_rot.elements[2]]
        #est_rot = euler_to_quat(rotation_filter.estimate)
        #est_q = np.concatenate((est_rot.imaginary, [est_rot.scalar])).flatten()
        #p.resetBasePositionAndOrientation(estimate_id, position_filter.estimate, est_q)
        env.task_objects["actual_state"].set_orientation(q)
        env.task_objects["goal_state"].set_orientation(noisy_q)
        rotation_filter.filter_step(noisy_rot.elements)

        action = [0, 0, 0]
        obs, reward, done, info = env.step(action) #Necessary step for animation to continue
        time.sleep(0.001)
    return position_filter, rotation_filter


if __name__ == "__main__":

    params = get_input()
    #ground_truth, noisy_data, rotations, noisy_rotations = create_trajectory_points(params)
    if params["type"] == "line":
        generator = LineGenerator(0.0225, 3, 0.125, [(-2.5, 2.5), (-2.5, 2.5), (0, 4)], accelerate=False)
    elif params["type"] == "circle":
        generator = CircleGenerator(0.0225)
    elif params["type"] == "spline":
        generator = SplineGenerator(0.0225, 4, 0.35)
    else:
        generator = LineGenerator(0.0225, 3, 0.15, [(-2, 2), (-2, 2), (0, 4)], accelerate=True)
    #ground_truth, rotations = generator2.generate_1_circle()
    #ground_truth = generator2.generate_1_circle()
    #ground_truth, rotations = generator.generate_1_trajectory()

    #generator.save_trajectories([ground_truth],[rotations])
    #generator.save_1_trajectory(ground_truth)
    #generator3.generate_and_save_n_trajectories(5)
    #ground_truth = np.load("./dataset/circles/positions/circle2.npy")
    # rotations = load_rotations("./dataset/circles/rotations/rot2.npy")
    #
    # noisy_data = np.load("./dataset/circles/positions/circle_noise2.npy")
    # noisy_rotations = load_rotations("./dataset/circles/rotations/rot_noise2.npy")
    #noisy_rotations = rotations
    arg_dict = get_arg_dict()
    #params = {"vis": "1"}
    if params["vis"] == "3":
        env = visualize_env(arg_dict)
        env.reset()
        saved_trajectory_index = 4
        for i in range(20):
            ground_truth, rotations, noisy_data, noisy_rotations = generator.generate_1_trajectory()
            ids, noise_ids = visualize_trajectory(ground_truth, noisy_data)
            test = input("Press 1 for saving trajectory, 0 for not saving")
            if test == "1":
                print("This trajectory was saved")
                saved_trajectory_index += 1
                generator.save_1_trajectory(ground_truth, rotations, noisy_data, noisy_rotations, i= str(saved_trajectory_index))
            else:
                print("This trajectory was not saved")
            for j in range(len(ids)):
                p.removeUserDebugItem(ids[j])
                p.removeUserDebugItem(noise_ids[j])

    if params["vis"] == "1":
        env = visualize_env(arg_dict)
        env.reset()
        ids = visualize_trajectory(ground_truth, noisy_data)
        #time.sleep(10)
        #sys.exit()
        resulting_pos_filter, resulting_rot_filter = vis_anim(ground_truth, noisy_data, rotations, noisy_rotations, float(params["pause"]), type =params["filter_type"])
        filenameP = None
        visualize_errors(ground_truth, noisy_data, resulting_pos_filter.estimates, filenameP)
        filenameR =None
        visualize_rot_errors(rotations, noisy_rotations, resulting_rot_filter.estimates, filenameR)
    else:
        resulting_pos_filter, resulting_rot_filter = filter_without_animation(noisy_data, noisy_rotations, type = params["filter_type"])
        filenameP = None
        filenameR = None
        visualize_errors(ground_truth, noisy_data, resulting_pos_filter.estimates, filenameP)
        visualize_rot_errors(rotations, noisy_rotations, resulting_rot_filter.estimates, filenameR)



