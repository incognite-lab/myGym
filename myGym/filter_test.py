"""
This script serves the purpose of visualizing generated (noisy) trajectory of object along with its
filtered result. Filtering is done through the Kalman and particle filtering algorithm.
"""

from ast import arg
import gym
from myGym import envs
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env

import pandas as pd
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
import matplotlib
matplotlib.use('TkAgg')


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
    first_option = int(input("Enter 1 for filtering, 2 for MOT"))
    input_option = int(input("Enter 1 to generate trajectories, 2 to visualize filtration process"))
    trajectory_dict = dict()
    if input_option == 1:
        if first_option == 1:
            type = int(input("Enter trajectory type (1 for lines, 2 for circle, 3 for spline, 4 for lines with acceleration: )"))

            if type == 1:
                trajectory_dict["type"] = "lines"

            elif type == 2:
                trajectory_dict["type"] = "circle"

            elif type == 3:
                trajectory_dict["type"] = "spline"

            elif type == 4:
                trajectory_dict["type"] = "lines_acc"

        elif first_option == 2:
            trajectory_dict["type"] = "multiple"
            trajectory_dict["traj_amount"] = int(input("Enter the amount of generated trajectories: "))
        trajectory_dict["action"] = '1'
    else:
        trajectory_dict = dict()
        if first_option == 1:
            type = int(
                input("Enter trajectory type (1 for lines, 2 for circle, 3 for spline, 4 for lines with acceleration: )"))
            if type == 1:
                trajectory_dict["type"] = "lines"
            elif type == 2:
                trajectory_dict["type"] = "circle"
            elif type == 3:
                trajectory_dict["type"] = "spline"
            elif type == 4:
                trajectory_dict["type"] = "lines_acc"
            option = int(input("Enter 1 to generate a trajectory, 2 to load a trajectory from saved trajectories"))
            if option == 1:
                trajectory_dict["action"] = '3'
            else:
                trajectory_dict["action"] = '2'
        else:
            trajectory_dict["type"] = "multiple"
            option = int(input("Enter 1 to generate a trajectory, 2 to load a trajectory from saved trajectories"))
            if option == 1:
                trajectory_dict["action"] = '3'
                trajectory_dict["traj_amount"] = int(input("Enter the amount of generated trajectories: "))
            else:
                trajectory_dict["action"] = '2'

        trajectory_dict["filter_type"] = input("Enter the type of particle filter (1 for g-h, 2 for VelocityPF, 3 for PFK, 4 for simple Kalman")
    if trajectory_dict["action"] != '1':
        trajectory_dict["pause"] = input("Enter the length of each filter phase (predict, update, resample) in seconds: ")
        if trajectory_dict["filter_type"] != '4':
            trajectory_dict["particle_vis_option"] = input("Enter visualization option of particles (1 for all particles, 2 for just a few, 3 for none)")

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

def visualize_errors(ground_truth, noisy_data, estimates):
    """
    Visualization of position errors in a plot.
    """
    diffs_meas = []
    diffs_est = []
    print("squared errors:")
    for i in range(ground_truth.shape[0]):
        meas_error = np.linalg.norm(ground_truth[i, :] - noisy_data[i]) ** 2
        est_error  = np.linalg.norm(ground_truth[i, :] - estimates[i]) ** 2
        print("K:", i, "Error:", est_error)
        print("K;", i, "measurement_error", meas_error)
        diffs_meas.append(meas_error)
        diffs_est.append(np.linalg.norm(ground_truth[i, :] - estimates[i])**2)
    diffs_meas = np.array(diffs_meas)
    plot_comparison(diffs_meas, diffs_est, "Position error evaluation")


def plot_comparison(diffs_meas, diffs_est, title):
    """Plots how the filter performed. Compares the error (difference between ground truth) of measured and estimated
    data."""
    diffs_meas = np.array(diffs_meas)
    k = np.arange(len(diffs_meas))
    measured_avg = np.average(diffs_meas)/3
    estimated_avg = np.average(diffs_est)/3


    # Plotting results
    fig, ax = plt.subplots()
    print("visualizing errors:")
    ax.text(.01, .95, 'Measurement MSE: ' + str(round(measured_avg, 5)), transform=ax.transAxes)
    plt.text(.01, .9, 'Estimate MSE: ' + str(round(estimated_avg, 5)), transform=ax.transAxes)
    ax.plot(k, diffs_est, label="Estimate SE")
    plt.plot(k, diffs_meas, label="Measurement SE")
    plt.title(title)
    plt.legend()
    #plt.savefig('/home/frederik/school/projekt+BP/figures/' + filename + ".png")
    plt.show()


def visualize_rot_errors(rotations, noisy_rotations, estimates):
    """
    Visualization of rotation errors
    """
    diffs_meas = []
    diffs_est = []
    for i in range(len(rotations)):
        diffs_meas.append(Quaternion.absolute_distance(rotations[i], noisy_rotations[i])**2)
        diffs_est.append(Quaternion.absolute_distance(rotations[i], Quaternion(estimates[i]))**2)
    plot_rot_marginals(rotations, noisy_rotations, estimates)
    plot_comparison(diffs_meas, diffs_est, "Rotation error evaluation")



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
        position_filter = ParticleFilterGH(600, 0.02, 0.02, g= 0.7, h = 0.4)
        rotation_filter = ParticleFilterGHRot(600, np.deg2rad(1), np.deg2rad(4), g= 0.8, h = 0.8)
    elif type == "2": #Particle 3D
        position_filter = ParticleFilterVelocity(600, 0.0225942, 0.38, 0.0106, res_g=0.3629)
        rotation_filter = ParticleFilterVelocityRot(600, 0.02, np.deg2rad(1), np.deg2rad(4))
    elif type == "3":#PFK
        position_filter = ParticleFilterWithKalman(1500, 0.00860, 0.03373, Q= 0.05076, R = 0.07527, std_a=0.31746)
        rotation_filter = ParticleFilterWithKalmanRot(1500, 0.1582, 0.4987, Q= 0.76, R =2.62, factor_a = 0.868)
    elif type == "4": #Kalman
        x = np.array([0., 0., 0., 0., 0., 0.]) # [x, dx/dt, y, dy/dt, z, dz/dt]
        xr = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        #P = np.diag([100., 25., 64, 0.1, 0.1, 0.1])
        Q = 0.05
        R = 0.015
        var_angle = np.deg2rad(90)
        var_angle_vel = np.deg2rad(20)
        Pr= np.diag([var_angle, var_angle_vel, var_angle, var_angle_vel, var_angle, var_angle_vel, var_angle, var_angle_vel])
        Qr = np.deg2rad(4)
        Rr = np.deg2rad(4)
        position_filter = myKalmanFilter(x, P=P, Q=Q, R=R, Q_scale_factor=499, eps_max =0.12)
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
            rotation_filter.apply_first_measurement(rot_meas)
            rotation_filter.state_estimate()
            continue
        rotation_filter.filter_step(rot_meas)
        position_filter.filter_step(measurement)
    return position_filter, rotation_filter


def vis_anim_MOT_proof_of_concept(y, X, lens, X_rot, pause_length, type, num_objects):
    env.p.setGravity(0, 0, 0)
    p_length = 0.002
    n = max(lens)
    print(lens)
    Cdiag = np.diag([1]*num_objects)
    Coffdiag = np.full((num_objects, num_objects), 0.000)
    C = Cdiag + Coffdiag
    print(C)
    particle_batches = [None] * num_objects
    estimates_ids = [None] * num_objects
    measurements_ids = [None] * num_objects
    actual_state_ids = [None] * num_objects
    pos_filters = []
    last_X_index = 0
    rot = [1, 0, 0, 0]
    ground_truths = []
    noisy_data = []
    for i in range(num_objects):
        position_filter, rotation_filter = initialize_filter(type)
        pos_filters.append(position_filter)
        ground_truths.append(np.zeros((lens[i], 3)))
        noisy_data.append(np.zeros((lens[i], 3)))
    for k in range(n):
        print("doing filter iteration:", k)
        measurements_step_k = []
        num_objects_at_step_k = 0
        index = 0
        meas_classes = []
        while X[last_X_index + index, 3] == k*0.2:
            num_objects_at_step_k += 1
            index += 1
            if last_X_index + index >= X.shape[0]:
                break
        for j in range(num_objects_at_step_k):
            print("num_objects_at_step_k:", num_objects_at_step_k)
            #Collecting current step measurements
            gt_val = y[last_X_index + j, :3]
            meas = X[last_X_index + j, :3]
            meas_class = int(X[last_X_index+ j, 4])
            measurements_step_k.append(meas)
            meas_classes.append(meas_class)
            # min_dist = np.inf

            pos_filter = pos_filters[meas_class] # Choosing the particular filter based on 100% accurate data association

            ground_truths[meas_class][k, :3] = gt_val
            noisy_data[meas_class][k, :3] = meas
            if k == 0:

                # Initialize visualization of estimate and actual state
                actual_state_ids[meas_class] = visualize_gt(gt_val)
                measurements_ids[meas_class] = visualize_meas(meas)

                # Applying first filter measurement and estimating state
                pos_filter.apply_first_measurement(meas)
                pos_filter.state_estimate()
                pos_filter.object_class = meas_class

                # Visualizing particles and estimate
                particle_batch = visualize_particles(pos_filter, params["particle_vis_option"])
                particle_batches[meas_class] = particle_batch
                estimate_id = visualize_estimate(pos_filter)
                estimates_ids[meas_class] = estimate_id
                continue

            # pos_filter = None
            # for filter in pos_filters:
            #     distance = np.linalg.norm(filter.estimate - meas)
            #     if distance < min_dist:
            #         min_dist = distance
            #         pos_filter = filter

            assert pos_filter.object_class == meas_class
            move_object(actual_state_ids[meas_class], gt_val, rot)
            move_object(measurements_ids[meas_class], meas, rot)
            time.sleep(p_length)

            # 1) Predict movement
            pos_filter.predict()
            move_particles(particle_batches[meas_class], pos_filter.particles)
            time.sleep(p_length)

            # 2) Update based on measurement - ALL filters
            #pos_filter.update(meas)

            for l in range(len(pos_filters)):
                weightsum = pos_filters[l].update_MOT(meas, meas_class)
                print("updated filter", pos_filters[l].object_class, "with measurement", meas_class)
                print("average weight contribution", weightsum/pos_filters[l].num_particles)
            time.sleep(p_length)


            # 3) Compute estimate

            # pos_filter.state_estimate()
            # move_object(estimates_ids[meas_class], pos_filter.estimate, rot)
            # time.sleep(p_length)
            # print("dt of pos filter:", pos_filter.dt)

            # 4) Resample particles
            # if pos_filter.neff() < pos_filter.num_particles/2:
            #     pos_filter.resample()
            # move_particles(particle_batch, pos_filter.particles)
        for l in range(len(pos_filters)):
            filter = pos_filters[l]
            filter.state_estimate_MOT()
            move_object(estimates_ids[filter.object_class], filter.estimate, rot)
            time.sleep(p_length)
            if filter.neff() < filter.num_particles / 2:
                filter.resample()
            filter.reset_weights()
            move_particles(particle_batches[filter.object_class], filter.particles)
            time.sleep(p_length)


        last_X_index += num_objects_at_step_k
    return pos_filters, ground_truths, noisy_data

def vis_anim_MOT(y, X, y_rot, X_rot, pause_length, type, num_objects):
    env.p.setGravity(0, 0, 0)
    n = X.shape[0]
    particle_batches = [None]*num_objects
    estimates_ids = [None]*num_objects
    measurements_ids = [None]*num_objects
    actual_state_ids = [None]*num_objects
    pos_filters = []
    rot_filters = []
    initialized_filters = set()

    for i in range(num_objects):
        position_filter, rotation_filter = initialize_filter(type)
        pos_filters.append(position_filter)
        rot_filters.append(rotation_filter)

    for k in range(n):
        measurement = X[k, :3]
        rot_meas = X_rot[k, :4]
        meas_time = X[k, 3]
        meas_class = int(X[k, 4])
        gt_val = y[k, :3]
        rot = y_rot[k, : 4]
        pos_filter = pos_filters[meas_class] #Choosing the particular filter based on 100% accurate data association
        rot_filter = rot_filters[meas_class]
        #If filter of given class has not been initialized yet
        if meas_class not in initialized_filters:
            #Initialize visualization of estimate and actual state
            actual_state_ids[meas_class] = visualize_gt(gt_val)
            measurements_ids[meas_class] = visualize_meas(measurement)

            #Applying first filter measurement and estimating state
            pos_filter.apply_first_measurement(measurement)
            pos_filter.state_estimate_MOT(meas_time)

            #Visualizing particles and estimate
            particle_batch = visualize_particles(pos_filter, params["particle_vis_option"])
            particle_batches[meas_class] = particle_batch
            estimate_id = visualize_estimate(pos_filter)

            #Performing rotation filter first measurement
            rot_filter.apply_first_measurement(rot_meas)
            rot_filter.state_estimate_MOT(meas_time)
            #Saving initialized data to list and set
            estimates_ids[meas_class] = estimate_id
            initialized_filters.add(meas_class)
            continue

        #If there is no measurement value at given timestep (at times 0.2, 0.4, 0.6...)
        if np.allclose(measurement, [88, 88, 88]):
            #Perform filter prediction
            pos_filter.predict_MOT(meas_time)
            rot_filter.predict_MOT(meas_time)
            #Estimate state to compare with ground truth
            pos_filter.state_estimate_MOT(meas_time)
            rot_filter.state_estimate_MOT(meas_time)
            #Move particles in visualization
            move_particles(particle_batches[meas_class], pos_filter.particles)
            print("estimate:", pos_filter.estimate)
            print("ground_truth:", gt_val)
            move_object(estimates_ids[meas_class], pos_filter.estimate, rot_filter.estimate)
            continue
        #New step of filter -> need to move visualization of ground truth and measurement
        move_object(actual_state_ids[meas_class], gt_val, rot)
        move_object(measurements_ids[meas_class], measurement, rot_meas)

        # 1) Predict movement
        pos_filter.predict_MOT(meas_time)
        move_particles(particle_batches[meas_class], pos_filter.particles)
        time.sleep(pause_length)

        # 2) Update based on measurement
        pos_filter.update(measurement)
        time.sleep(pause_length)
        do_resample = True

        #3) Compute estimate
        if pos_filter.neff() < pos_filter.num_particles/50:
            pos_filter.resample()
            pos_filter.reapply_measurement(measurement)
        pos_filter.state_estimate_MOT(meas_time)
        print("estimate:", pos_filter.estimate)
        print("ground_truth:", gt_val)
        print("estimate_id:", estimates_ids[meas_class])
        move_object(estimates_ids[meas_class], pos_filter.estimate, rot_filter.estimate)
        time.sleep(pause_length)

        #4) Resample particles
        if pos_filter.neff() < pos_filter.num_particles / 2 and do_resample:
            pos_filter.resample()
        move_particles(particle_batches[meas_class], pos_filter.particles)
        time.sleep(pause_length)

        #5) Compute rot filter step
        rot_filter.filter_step(rot_meas)
        action = [0, 0, 0]
        obs, reward, done, info = env.step(action)  # Necessary step for animation to continue
        time.sleep(0.001)
    return pos_filters, rot_filters



def vis_anim_MOT_old(y, X, y_rot, X_rot, pause_length, type):
    """
    Visualization of MOT filtering
    """
    env.p.setGravity(0, 0, 0)
    n = X.shape[0]
    num_objects = int(y.shape[1]/3)
    particle_batches = []
    estimates_ids = []
    measurements_ids = []
    actual_state_ids = []
    pos_filters = []
    rot_filters = []
    gts, rs, nds, nrs = convert_X_y_into_vis_data(X, y, X_rot, y_rot)

    for i in range(num_objects):
        position_filter, rotation_filter = initialize_filter(type)
        pos_filters.append(position_filter)
        rot_filters.append(rotation_filter)

    for k in range(n):
        #Iterate through measurement steps k
        for j in range(num_objects):
            #Select filter based on trajectory
            position_filter = pos_filters[j]
            rotation_filter = rot_filters[j]
            #Iterate through every object and its trajectory on scene
            measurement = nds[j][k, :]
            if np.allclose(measurement, [88, 88, 88]):
                continue
            rot_meas = nrs[j][k, :]
            rot = rs[j][k, :]
            gt_val = gts[j][k, :]
            if k == 0:
                #Initialize PF based on first measurement
                actual_state_ids.append(visualize_gt(gt_val))
                measurements_ids.append(visualize_meas(measurement))

                position_filter.apply_first_measurement(measurement)
                position_filter.state_estimate()
                if type != "4":
                    particle_batch = visualize_particles(position_filter, params["particle_vis_option"])
                    particle_batches.append(particle_batch)
                estimate_id = visualize_estimate(position_filter)
                rotation_filter.apply_first_measurement(rot_meas)
                rotation_filter.state_estimate()
                estimates_ids.append(estimate_id)
                continue

            move_object(actual_state_ids[j], gt_val, rot)
            move_object(measurements_ids[j], measurement, rot_meas)

            #1) Predict movement
            position_filter.predict()
            if type != "4":
                move_particles(particle_batches[j], position_filter.particles)
            time.sleep(pause_length)

            #2) Update based on measurement
            position_filter.update(measurement)
            if type != "4":
                move_particles(particle_batches[j], position_filter.particles)
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
                move_particles(particle_batches[j], position_filter.particles)
            move_object(estimates_ids[j], position_filter.estimate, rotation_filter.estimate)
            time.sleep(pause_length)

            #4) Resample particles
            if type != "4":
                if position_filter.neff() < position_filter.num_particles/2 and do_resample:
                    position_filter.resample()
                move_particles(particle_batches[j], position_filter.particles)
            time.sleep(pause_length)

            #5) Compute rotation_filter step
            rotation_filter.filter_step(rot_meas)

            action = [0, 0, 0]
            obs, reward, done, info = env.step(action)  # Necessary step for animation to continue
            time.sleep(0.001)
    return pos_filters, rot_filters





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
                particle_batch = visualize_particles(position_filter, params["particle_vis_option"])
            estimate_id = visualize_estimate(position_filter)
            rotation_filter.apply_first_measurement(rot_meas)
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
        # if type != "4":
        #     if position_filter.neff() < position_filter.num_particles/30:
        #         position_filter.resample()
        #         position_filter.reapply_measurement(measurement)
        #         do_resample = False
        position_filter.state_estimate()
        if type != "4":
            move_particles(particle_batch, position_filter.particles)
        time.sleep(pause_length)
        move_object(estimate_id, position_filter.estimate, rotation_filter.estimate)
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
        q = [rot[3], rot[0], rot[1], rot[2]]
        noisy_q = [noisy_rot[3], noisy_rot[0], noisy_rot[1], noisy_rot[2]]
        #est_rot = euler_to_quat(rotation_filter.estimate)
        #est_q = np.concatenate((est_rot.imaginary, [est_rot.scalar])).flatten()
        #p.resetBasePositionAndOrientation(estimate_id, position_filter.estimate, est_q)
        env.task_objects["actual_state"].set_orientation(q)
        env.task_objects["goal_state"].set_orientation(noisy_q)
        rotation_filter.filter_step(noisy_rot)

        action = [0, 0, 0]
        obs, reward, done, info = env.step(action) #Necessary step for animation to continue
        time.sleep(0.001)
    return position_filter, rotation_filter


if __name__ == "__main__":
    params = get_input()
    #ground_truth, noisy_data, rotations, noisy_rotations = create_trajectory_points(params)
    if params["type"] == "lines":
        generator = LineGenerator(0.0175, 3, 0.15, [(-2.5, 2.5), (-2.5, 2.5), (0, 4)], accelerate=False)
    elif params["type"] == "circle":
        generator = CircleGenerator(0.0175)
    elif params["type"] == "spline":
        generator = SplineGenerator(0.0175, 4, 0.35)
    elif params["type"] == "lines_acc":
        generator = LineGenerator(0.0175, 3, 0.15, [(-2, 2), (-2, 2), (0, 4)], accelerate=True)
    else:
        #Multiple trajectories
        if params["action"] != "2":
            with open("./configs/MOT_trajectory_parameters.json") as json_file:
                traj_params = json.load(json_file)
            generator = MultipleTrajectoryGenerator(params["traj_amount"], traj_params, dt_std = None) #CONSTAND dt_std

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
    if params["action"] == "1":
        env = visualize_env(arg_dict)
        env.reset()
        saved_trajectory_index = 0
        for i in range(20):
            if params["type"] != "multiple":
                ground_truth, rotations, noisy_data, noisy_rotations, t_vec = generator.generate_1_trajectory()
                ids, noise_ids, cube_ids = visualize_trajectory_timesteps(ground_truth, noisy_data)
                save = input("Press 1 for saving trajectory, 0 for not saving")
                if save == "1":
                    print("This trajectory was saved")
                    saved_trajectory_index += 1
                    generator.save_1_trajectory(ground_truth, rotations, noisy_data, noisy_rotations,
                                                i=str(saved_trajectory_index))
                else:
                    print("This trajectory was not saved")
                p.removeAllUserDebugItems()
                for j in range(len(cube_ids)):
                    p.removeBody(cube_ids[j])
            else:

                X, y, X_rot, y_rot = generator.generate_1_scenario()
                gts, rs, nds, nrs = convert_X_y_into_vis_data(X, y, X_rot, y_rot, params["traj_amount"])
                ids, noise_ids, cube_ids = visualize_multiple_trajectories(gts, nds)
                save = input("Press 1 for saving trajectory, 0 for not saving")
                if save == "1":
                    print("This trajectory was saved")
                    saved_trajectory_index += 1
                    generator.save_1_trajectory(X,y, X_rot, y_rot)
                else:
                    print("This trajectory was not saved")
                p.removeAllUserDebugItems()
                for j in range(len(cube_ids)):
                    p.removeBody(cube_ids[j])

    elif params["action"] == "2":
        if params["type"] != "multiple":
            ground_truth, rotations, noisy_data, noisy_rotations = load_trajectory(params["type"])
            #print("PRINTING SHAPES:", ground_truth.shape)
            env = visualize_env(arg_dict)
            env.reset()
            ids = visualize_trajectory(ground_truth, noisy_data)
            resulting_pos_filter, resulting_rot_filter = filter_without_animation(noisy_data, noisy_rotations, type =params["filter_type"])
            visualize_errors(ground_truth, noisy_data, resulting_pos_filter.estimates)
        else:
            y, X, y_rot, X_rot = load_MOT_scenario()
            traj_amount = determine_traj_amount(y)
            gts, rs, nds, nrs = convert_X_y_into_vis_data(X, y, X_rot, y_rot, traj_amount)
            df = pd.DataFrame(y)
            # df.to_csv("./testing tables/last_day/ground_truth.csv")
            # df2 = pd.DataFrame(X)
            # df.to_csv("./testing tables/last_day/noisy_data.csv")
            # df3 = pd.DataFrame(X_rot)
            # df.to_csv("./testing tables/last_day/rotations.csv")
            # df4 = pd.DataFrame(y_rot)
            # df.to_csv("./testing tables/last_day/noisy_rotations.csv")
            env = visualize_env(arg_dict)
            env.reset()
            ids, noise_ids, cube_ids = visualize_multiple_trajectories(gts, nds)
            lens = []
            for traj in gts:
                lens.append(traj.shape[0])

            resulting_pos_filters, ground_truths, noisy_data = vis_anim_MOT_proof_of_concept(y, X, lens, X_rot,
                                    type = params["filter_type"], pause_length = float(params["pause"]), num_objects = 3)
            for i in range(len(resulting_pos_filters)):
                visualize_errors(ground_truths[i], noisy_data[i], resulting_pos_filters[i].estimates)

    else:
        env = visualize_env(arg_dict)
        env.reset()
        saved_trajectory_index = 4
        if params["type"] != "multiple":
            ground_truth, rotations, noisy_data, noisy_rotations, t_vec = generator.generate_1_trajectory()
        else:
            X, y, X_rot, y_rot = generator.generate_1_scenario()
            gts, rs, nds, nrs = convert_X_y_into_vis_data(X, y, X_rot, y_rot, params["traj_amount"])
        for i in range(50):
            if params["type"] != "multiple":
                ids, noise_ids = visualize_trajectory(ground_truth, noisy_data)
            else:
                ids, noise_ids, cube_ids = visualize_multiple_trajectories(gts, nds)
            test = input("Press 1 for using trajectory, 0 for generating a new one")
            if test == "1":
                print("This trajectory will be used for filtration")
                saved_trajectory_index += 1
                # df = pd.DataFrame(y)
                # df.to_csv("./testing tables/last_day/ground_truth.csv")
                # df2 = pd.DataFrame(X)
                # df.to_csv("./testing tables/last_day/noisy_data.csv")
                # df3 = pd.DataFrame(X_rot)
                # df.to_csv("./testing tables/last_day/rotations.csv")
                # df4 = pd.DataFrame(y_rot)
                # df.to_csv("./testing tables/last_day/noisy_rotations.csv")
                break
            else:
                print("Generating new trajectory")
                if params["type"] != "multiple":
                    ground_truth, rotations, noisy_data, noisy_rotations, t_vec = generator.generate_1_trajectory()
                else:
                    X, y, X_rot, y_rot = generator.generate_1_scenario()
                    gts, rs, nds, nrs = convert_X_y_into_vis_data(X, y, X_rot, y_rot, params["traj_amount"])


                for j in range(len(ids)):
                    p.removeUserDebugItem(ids[j])
                    p.removeUserDebugItem(noise_ids[j])

        if params["type"] != "multiple":
            resulting_pos_filter, resulting_rot_filter = vis_anim(ground_truth, noisy_data, rotations, noisy_rotations,
                                                              float(params["pause"]), type=params["filter_type"])
        else:
            traj_amount = determine_traj_amount(y)
            gts, rs, nds, nrs = convert_X_y_into_vis_data(X, y, X_rot, y_rot, traj_amount)

            #ids, noise_ids, cube_ids = visualize_multiple_trajectories(gts, nds)
            lens = []
            for traj in gts:
                lens.append(traj.shape[0])
            print("lens:", lens)
            resulting_pos_filters, ground_truths, noisy_data = vis_anim_MOT_proof_of_concept(y, X, lens, X_rot,
                                                                                             type=params["filter_type"],
                                                                                             pause_length=float(
                                                                                                 params["pause"]),
                                                                                             num_objects=traj_amount)
            for i in range(len(resulting_pos_filters)):
                visualize_errors(ground_truths[i], noisy_data[i], resulting_pos_filters[i].estimates)







