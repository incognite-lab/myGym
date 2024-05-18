from ast import arg
import gym
from myGym import envs
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error

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

from scipy.stats import uniform
import matplotlib.pyplot as plt
from myGym.envs.particle_filter import *
from myGym.utils.filter_helpers import *

from myGym.envs.trajectory_generator import *

import pandas as pd

from myGym.filter_test import visualize_errors, plot_comparison


class primitiveEstimator(BaseEstimator):

    def __init__(self, param1 = 0, param2 = 10):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y=None):
        self.y_ = y
        self.x_ = X
        return self

    def predict(self, X):
        self.x_ = X + self.param2
        return self.x_


class myEstimator(BaseEstimator):

    def fit(self, X, y=None):
        self.y_ = y
        self.x_ = X
        return self

    def predict(self, X):
        """
        Entire filtration algorithm present here. X is noisy data representing the measurements.
        """
        noisy_data = X
        n = X.shape[0]
        new_trajectory = False
        self.x_ = np.zeros_like(X)

        measurement_dim = X.shape[1]
        for i in range(n):
            measurement = noisy_data[i]
            if np.allclose(measurement, np.array([99]*measurement_dim)):
                #New trajectory, resetting filter
                self.initialize_filter()
                self.x_[i] = measurement
                new_trajectory = True
            elif new_trajectory:
                #print("New trajectory, iter:", i)

                self.filter.apply_first_measurement(measurement)
                self.filter.state_estimate()
                self.x_[i] = self.filter.estimate
                #print("State estimate:", self.filter.estimate)
                new_trajectory = False
                continue
            else:
                self.filter.filter_step(measurement)
                self.x_[i] = self.filter.estimate
        #visualize_errors(self.y_, X, self.x_)
        return self.x_


    def initialize_filter(self):
        self.filter = ParticleFilterVelocity(100, 0.02, 0.1,
                                       0.03, res_g=0.5)



class EstimatorVelocity(myEstimator):
    def __init__(self, num_particles = 600, process_std = 0.02, vel_std = 0.1, measurement_std = 0.02, res_g = 0.5):
        self.vel_std = vel_std
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.res_g = res_g


    def initialize_filter(self):
        self.filter = ParticleFilterVelocity(self.num_particles, self.process_std, self.vel_std,
                                       self.measurement_std, res_g=self.res_g)





class EstimatorVelocityRot(myEstimator):
    def __init__(self, num_particles = 600, process_std = 0.02, vel_std = 0.1, measurement_std = 0.02, res_g =0.5,
                 rotflip_const = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.res_g = res_g
        self.vel_std = vel_std
        self.rotflip_const = rotflip_const


    def initialize_filter(self):
        self.filter = ParticleFilterVelocityRot(
            self.num_particles, self.process_std, self.vel_std, self.measurement_std, res_g=self.res_g,
            rotflip_const=self.rotflip_const)





class EstimatorGH(myEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, g = 0.5, h = 0.5):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.g = g
        self.h = h



    def initialize_filter(self):
        self.filter = ParticleFilterGH(self.num_particles, self.process_std,
                                                self.measurement_std, vel_std=0.01, g=self.g, h=self.h)



class EstimatorGHRot(myEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, g = 0.5, h = 0.5, rotflip_const = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.g = g
        self.h = h
        self.rotflip_const = rotflip_const


    def initialize_filter(self):
        self.filter = ParticleFilterGHRot(self.num_particles, self.process_std,
                                          self.measurement_std, vel_std=0.01, g=self.g, h=self.h)



class EstimatorPFKalman(myEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, Q=None, R=None, std_a = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.Q = Q
        self.R = R
        self.std_a = std_a


    def initialize_filter(self):
        self.filter = ParticleFilterWithKalman(self.num_particles, self.process_std,
                                                        self.measurement_std, Q=self.Q, R=self.R, std_a=self.std_a)

class EstimatorPFKalmanRot(myEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, Q =None, R = None, std_a = 0.1, rotflip_const = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.rotflip_const = rotflip_const
        self.Q = Q
        self.R = R
        self.std_a = std_a


    def initialize_filter(self):
        self.filter = ParticleFilterWithKalmanRot(self.num_particles, self.process_std,
                                                  self.measurement_std, Q=self.Q, R=self.R, factor_a=self.std_a,
                                                  rotflip_const=self.rotflip_const)



class EstimatorKalman(myEstimator):
    def __init__(self, Q = 0.08, R = 0.02, eps_max = 0.18, Q_scale_factor = 500):
        self.Q = Q
        self.R = R
        self.eps_max = eps_max
        self.Q_scale_factor = Q_scale_factor
        self.initial_x = np.array([0., 0., 0., 0., 0., 0.])
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


    def initialize_filter(self):
        self.filter = myKalmanFilter(self.initial_x, P=self.P, R=self.R, Q=self.Q,
                                              Q_scale_factor=self.Q_scale_factor, eps_max=self.eps_max)



class EstimatorKalmanRot(myEstimator):
    def __init__(self, Q = 0.08, R = 0.02, eps_max = 0.18, Q_scale_factor = 500, rotflip_const = 0.1):
        self.Q = Q
        self.R = R
        self.eps_max = eps_max
        self.Q_scale_factor = Q_scale_factor
        self.initial_x = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.rotflip_const = rotflip_const

    def initialize_filter(self):
        self.filter = myKalmanFilterRot(self.initial_x, self.P, self.R, self.Q,
                                                 self.rotflip_const, self.Q_scale_factor)


def filter_gridsearch(filters, parameters, trajectory_types, search_type, iterations):
    """
    All-encompassing testing function for using gridsearch
    """

    filter_estimator_dict = {
        "myKalmanFilter" : "EstimatorKalman",
        "myKalmanFilterRot": "EstimatorKalmanRot",
        "ParticleFilterVelocity": "EstimatorVelocity",
        "ParticleFilterGH" : "EstimatorGH",
        "ParticleFilterVelocityRot" : "EstimatorVelocityRot",
        "ParticleFilterGHRot" : "EstimatorGHRot",
        "ParticleFilterWithKalman" : "EstimatorPFKalman",
        "ParticleFilterWithKalmanRot" : "EstimatorPFKalmanRot",
    }
    for filter in filters:
        evaluate_and_save(filter, parameters, filter_estimator_dict, trajectory_types, search_type, iterations)


def load_params(paramstring):
    if type(paramstrintg) == dict:
        return paramstring
    param_grid = dict()
    with open("/home/alblfred/myGym/myGym/configs/" + paramstring + ".json") as json_file:
        data = json.load(json_file)
        for key in data:
            param_grid[key] = data[key]
    return param_grid


def evaluate_and_save(filter, parameters, est_dict, trajectory_types, search_type, iterations):
    """
    Function that performs gridsearch on one estimator with parameters corresponding to the filter type.
    Results are saved in a .npy file.
    """
    X, y = load_trajectories(trajectory_types)
    param_grid = load_params(parameters)
    cv = [(slice(None), slice(None))]
    estimator = est_dict[filter]
    if search_type == "GridSearch":
        search = GridSearchCV(eval(estimator + "()"), param_grid=param_grid[filter],
                               scoring="neg_mean_squared_error", cv=cv, verbose = 1)
    else:
        if filter == "myKalmanFilter" or filter == "myKalmanFilterRot":
            exclude_list = []
        else:
            exclude_list = ["num_particles"]
        search = RandomizedSearchCV(eval(estimator + "()"), param_conversion_to_RS(param_grid[filter], exclude_list),
                                    scoring = "neg_mean_squared_error",
                                    cv =cv, verbose = 1, n_iter = iterations)
    search.fit(X, y)
    create_and_save_dataframe(eval(estimator + "()"), search, param_grid[filter], trajectory_types)


def param_conversion_to_RS(param_grid, exclude_list):
    """Converts the given parameter grid into uniform distributions from the lowest amount to the highest"""
    new_param_grid = dict()
    for key in param_grid:
        value = param_grid[key]
        if key in exclude_list:
            new_param_grid[key] = value
            continue
        if len(value) > 1:
            first_val = value[0]
            last_val = value[-1]
            new_param_grid[key] = uniform(first_val, last_val - first_val)
        else:
            new_param_grid[key] = value
    return new_param_grid


def load_trajectories(types):
    """
    Creates X and y vectors that will get passed into gridsearch fit function.
    Vectors are made from dataset based on the input types.
    """
    filenames_base = []
    filenames_base_noise = []
    for type in types:
        if type =="lines":
            filenames_base.append("./dataset/lines/positions/line")
            filenames_base_noise.append( "./dataset/lines/positions/line_noise")
        elif type == "lines_acc":
            filenames_base.append("./dataset/lines_acc/positions/line")
            filenames_base_noise.append("./dataset/lines_acc/positions/line_noise")
        elif type == "circles":
            filenames_base.append("./dataset/circles/positions/circle")
            filenames_base_noise.append( "./dataset/circles/positions/circle_noise")
        elif type == "splines":
            filenames_base.append("./dataset/splines/positions/spline")
            filenames_base_noise.append( "./dataset/splines/positions/spline_noise")
        elif type == "lines_rot":
            filenames_base.append("./dataset/lines/rotations/rot")
            filenames_base_noise.append("./dataset/lines/rotations/rot_noise")
        elif type == "circles_rot":
            filenames_base.append("./dataset/circles/rotations/rot")
            filenames_base_noise.append("./dataset/circles/rotations/rot_noise")
        elif type == "splines_rot":
            filenames_base.append("./dataset/splines/rotations/rot")
            filenames_base_noise.append("./dataset/splines/rotations/rot_noise")
        elif type == "lines_acc_rot":
            filenames_base.append("./dataset/lines_acc/rotations/rot")
            filenames_base_noise.append("./dataset/lines_acc/rotations/rot_noise")
        else:
            print("Entered wrong trajectory types")
            sys.exit()
    #Constant velocity lines
    for i in range(len(filenames_base)):
        filename_base = filenames_base[i]
        filename_base_noise = filenames_base_noise[i]
        if "rotations" in filename_base:
            X = np.array([99, 99, 99, 99])
            y = np.array([99, 99, 99, 99])
        else:
            X = np.array([99, 99, 99])
            y = np.array([99, 99, 99])
        for i in range(1, 7, 1):
            filename = filename_base + str(i) + ".npy"
            filename_noise = filename_base_noise + str(i) + ".npy"
            gt = np.load(filename)
            meas = np.load(filename_noise)
            X = np.vstack((X, meas))
            y = np.vstack((y, gt))
    return X, y


def load_params_from_csv(filename):
    """
    Used to replace mean score time with mean fit time
    """
    df = pd.read_csv(filename, sep='\t', header=None)
    data = df.values[:11, 1:]
    print(data)
    param_list = []
    for row in range(1, data.shape[0], 1):
        params = dict()
        for i in range(data.shape[1]):
            params[data[0][i]] = [data[row][i]]
        param_list.append(params)
    return param_list


def create_and_save_dataframe(Estimator, grid_search, param_grid, trajectory_types):
    filter_name = Estimator.__class__.__name__
    dataframe_list = []
    rename_list = []
    trajectory_types_string = ""
    for type in trajectory_types:
        trajectory_types_string = trajectory_types_string + type + "_"
    for key in param_grid:
        rename_list.append(key)
        dataframe_list.append("param_" + key)
    dataframe_list.append("mean_test_score")
    dataframe_list.append("mean_fit_time")
    dataframe = pd.DataFrame(grid_search.cv_results_)[dataframe_list]
    rename_dict = {}
    for i in range(len(rename_list)):
        rename_dict[dataframe_list[i]] = rename_list[i]
    dataframe = dataframe.rename(columns = rename_dict)
    dataframe = dataframe.sort_values(by=['mean_test_score'], ascending = False, kind = "quicksort")
    dataframe.to_csv("/home/alblfred/myGym/myGym/results/a_fittime_" + filter_name + "_" + trajectory_types_string + ".csv", sep='\t')
    print("Saved one results table:", filter_name)


if __name__ == "__main__":
    # X, y = load_trajectories(["lines_rot", "lines_acc_rot"])
    #
    # param_grid = dict()
    # with open("/home/student/Desktop/myGym/myGym/configs/testing_parameters.json") as json_file:
    #     data = json.load(json_file)
    #     for key in data:
    #         param_grid[key] = data[key]
    #
    #
    # cv = [(slice(None), slice(None))]
    #scorer = make_scorer(scoring, greater_is_better = True)
    # grid_search = GridSearchCV(EstimatorVelocityRot(), param_grid = param_grid["ParticleVelocityFilter"], scoring = "neg_mean_squared_error", cv = cv)
    # print(grid_search.fit(X, y))
    # #print(grid_search.predict(X))
    # print(grid_search.cv_results_)
    #dataframe = pd.DataFrame(grid_search.cv_results_)[[ "param_Q", "param_R", "param_eps_max", "param_Q_scale_factor",  "mean_test_score"]]
    #dataframe = dataframe.rename(columns = {"param_num_particles": "Numberof particles", "param_process_std": "Process std"})
    #dataframe = pd.read_csv("/home/frederik/Prace/Brigada2023/mygym/myGym/out1.csv", sep = '\t')
    # #print("dataframe:", dataframe)
    # #dataframe = dataframe.sort_values(by="mean_test_score")
    # #dataframe.to_csv("outKalman.csv", sep='\t')


    # create_and_save_dataframe(EstimatorVelocityRot(), grid_search, param_grid["ParticleVelocityFilter"])
    filters = ["myKalmanFilter", "ParticleFilterGH", "ParticleFilterVelocity",  "ParticleFilterWithKalman"]
    filters_rot = ["myKalmanFilterRot", "ParticleFilterGHRot", "ParticleFilterVelocityRot", "ParticleFilterWithKalmanRot"]
    trajectory_types1 = ["lines", "lines_acc"]
    trajectory_types2 = ["circles", "splines"]
    trajectory_types1_rot = ["lines_rot", "lines_acc_rot"]
    trajectory_types2_rot = ["circles_rot", "splines_rot"]
    # filter_gridsearch(filters, "parameters", trajectory_types1, "GridSearch", 20)
    # filter_gridsearch(filters, "parameters", trajectory_types2, "GridSearch", 20)
    # filter_gridsearch(filters_rot, "parameters", trajectory_types1_rot, "GridSearch", 20)
    # filter_gridsearch(filters_rot, "parameters", trajectory_types2_rot, "GridSearch", 20)

    for file_name in glob.glob("./results/" + '*.csv'):
        x = pd.read_csv(file_name, low_memory=False)
    dicts = load_params_from_csv("./results/EstimatorGH_circles_splines_.csv")
    for params in dicts:
        filter_gridsearch(filters, params, trajectory_types1, "GridSearch", 20)
        filter_gridsearch(filters, params, trajectory_types2, "GridSearch", 20)
        filter_gridsearch(filters_rot, params, trajectory_types1_rot, "GridSearch", 20)
        filter_gridsearch(filters_rot, params, trajectory_types2_rot, "GridSearch", 20)
    # X, y = load_trajectories(["lines"])
    # param_grid = {"param1": [10, 20], "param2": [10, 20]}
    # cv = [(slice(None), slice(None))]
    # random_search = RandomizedSearchCV(primitiveEstimator(), param_grid, scoring = "neg_mean_squared_error",
    #                                 cv =cv, verbose = 1, n_iter = 4)
    # random_search.fit(X,X)
    # dataframe = pd.DataFrame(random_search.cv_results_)[["param_param1", "param_param2", "mean_test_score", "mean_score_time"]]
    # dataframe = dataframe.sort_values(by=['mean_test_score'], ascending = False, kind = "quicksort")
    # dataframe.to_csv("./results/test_estimator.csv", sep='\t')
