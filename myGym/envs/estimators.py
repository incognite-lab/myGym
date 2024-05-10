from ast import arg
import gym
from myGym import envs
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
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
import matplotlib.pyplot as plt
from myGym.envs.particle_filter import *
from myGym.utils.filter_helpers import *

from myGym.envs.trajectory_generator import *

import pandas as pd




class Estimator6D(BaseEstimator):
    def __init__(self, num_particles = 600, process_std = 0.02, vel_std = 0.1, measurement_std = 0.02, res_g = 0.5):
        self.vel_std = vel_std
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.res_g = res_g


    def fit(self, X, y=None):
        self.x_ = X
        self.y_ = y
        return self


    def predict(self, X):
        noisy_positions = X
        #self.x_ = noisy_positions

        self.position_filter = ParticleFilter6D(self.num_particles, self.process_std, self.vel_std,
                                                self.measurement_std, res_g=self.res_g)
        n = noisy_positions.shape[0]
        for i in range(n):
            measurement = noisy_positions[i]
            if i == 0:
                self.position_filter.apply_first_measurement(measurement)
                self.position_filter.state_estimate()
                continue
            self.position_filter.filter_step(measurement)
        self.position_filter.convert_estimates()
        self.x_ = self.position_filter.estimates

        #self.x_ = X
        return self.x_





class Estimator6DRot(BaseEstimator):
    def __init__(self, num_particles = 600, process_std = 0.02, vel_std = 0.1, measurement_std = 0.02, res_g =0.5,
                 rotflip_const = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.res_g = res_g
        self.vel_std = vel_std
        self.rotflip_const = rotflip_const


    def fit(self, X, y):
        #X, y = check_X_y(X, y)
        self.x_ = X
        self.y_ = y
        return self


    def predict(self, X):
        noisy_rots = X
        self.rotation_filter = ParticleFilter6DRot(
            self.num_particles, self.process_std, self.vel_std, self.measurement_std, res_g=self.res_g,
            rotflip_const=self.rotflip_const)
        n = noisy_rots.shape[0]
        for i in range(n):
            measurement = noisy_rots[i]
            if i == 0:
                self.rotation_filter.apply_first_measurement(measurement)
                self.rotation_filter.state_estimate()
                continue
            self.rotation_filter.filter_step(measurement)
        self.rotation_filter.convert_estimates()
        self.x_ = self.rotation_filter.estimates
        return self.x_





class EstimatorGH(BaseEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, g = 0.5, h = 0.5):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.g = g
        self.h = h


    def fit(self, X, y):
        self.x_ = X
        self.y_ = y
        return self


    def predict(self, X):
        #X, y = check_X_y(X, y)
        noisy_positions = X
        self.position_filter = ParticleFilterGH(self.num_particles, self.process_std,
                                                self.measurement_std, vel_std=0.01, g=self.g, h=self.h)
        n = noisy_positions.shape[0]
        for i in range(n):
            measurement = noisy_positions[i]
            if i == 0:
                self.position_filter.apply_first_measurement(measurement)
                self.position_filter.state_estimate()
                continue
            self.position_filter.filter_step(measurement)
        self.position_filter.convert_estimates()
        self.x_ = self.position_filter.estimates
        return self.x_



class EstimatorGHRot(BaseEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, g = 0.5, h = 0.5, rotflip_const = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.g = g
        self.h = h
        self.rotflip_const = rotflip_const

    def fit(self, X, y):
        self.x_ = X
        self.y_ = y
        return self


    def predict(self, X):
        noisy_rots = X
        self.rotation_filter = ParticleFilterGHRot(self.num_particles, self.process_std,
                                                self.measurement_std, vel_std=0.01, g=self.g, h=self.h)
        n = noisy_rots.shape[0]
        for i in range(n):
            measurement = noisy_rots[i]
            if i == 0:
                self.rotation_filter.apply_first_measurement(measurement)
                self.rotation_filter.state_estimate()
                continue
            self.rotation_filter.filter_step(measurement)
        self.rotation_filter.convert_estimates()
        self.x_ = self.rotation_filter.estimates
        return self.x_



class EstimatorPFKalman(BaseEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, Q=None, R=None, std_a = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.Q = Q
        self.R = R
        self.std_a = std_a

    def fit(self, X, y):
        self.x_ = X
        self.y_ = y
        return self

    def predict(self, X):
        #X, y = check_X_y(X, y)
        noisy_positions = X
        self.position_filter = ParticleFilterWithKalman(self.num_particles, self.process_std,
                                                self.measurement_std, Q=self.Q, R=self.R, factor_a=self.std_a)
        n = noisy_positions.shape[0]
        for i in range(n):
            measurement = noisy_positions[i]
            if i == 0:
                self.position_filter.apply_first_measurement(measurement)
                self.position_filter.state_estimate()
                continue
            self.position_filter.filter_step(measurement)
        self.position_filter.convert_estimates()
        self.x_ = self.position_filter.estimates
        return self.x_


class EstimatorPFKalmanRot(BaseEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, Q =None, R = None, std_a = 0.1, rotflip_const = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.rotflip_const = rotflip_const
        self.Q = Q
        self.R = R
        self.std_a = std_a


    def fit(self, X, y):
        self.x_ = X
        self.y_ = y
        return self


    def predict(self, X):
        #X, y = check_X_y(X, y)
        noisy_rots = X
        self.rotation_filter = ParticleFilterWithKalmanRot(self.num_particles, self.process_std,
                                                self.measurement_std, Q = self.Q, R = self.R, factor_a = self.std_a, rotflip_const = self.rotflip_const)
        n = noisy_rots.shape[0]
        for i in range(n):
            measurement = noisy_rots[i]
            if i == 0:
                self.rotation_filter.apply_first_measurement(measurement)
                self.rotation_filter.state_estimate()
                continue
            self.rotation_filter.filter_step(measurement)
        self.rotation_filter.convert_estimates()
        self.x_ = self.rotation_filter.estimates
        return self.x_



class EstimatorKalman(BaseEstimator):
    def __init__(self, Q = 0.08, R = 0.02, eps_max = 0.18, Q_scale_factor = 500):
        self.Q = Q
        self.R = R
        self.eps_max = eps_max
        self.Q_scale_factor = Q_scale_factor
        self.initial_x = np.array([0, 0, 0, 0, 0, 0])
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


    def fit(self, X, y):
        self.x_ = X
        self.y_ = y
        return self


    def predict(self, X):
        #X, y = check_X_y(X, y)
        noisy_positions = X
        self.position_filter = myKalmanFilter(self.initial_x, P = self.P, R = self.R, Q = self.Q, Q_scale_factor=self.Q_scale_factor, eps_max = self.eps_max)
        n = noisy_positions.shape[0]
        for i in range(n):
            measurement = noisy_positions[i]
            if i == 0:
                self.position_filter.apply_first_measurement(measurement)
                self.position_filter.state_estimate()
                continue
            self.position_filter.filter_step(measurement)
        self.position_filter.convert_estimates()
        self.x_ = self.position_filter.estimates
        print("Estimates:")
        print(self.x_)
        return self.x_


class EstimatorKalmanRot(BaseEstimator):
    def __init__(self, Q = 0.08, R = 0.02, eps_max = 0.18, Q_scale_factor = 500, rotflip_const = 0.1):
        self.Q = Q
        self.R = R
        self.eps_max = eps_max
        self.Q_scale_factor = Q_scale_factor
        self.initial_x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.rotflip_const = rotflip_const


    def fit(self, X, y):
        self.x_ = X
        self.y_ = y
        return self


    def predict(self, X):
        noisy_rotations = X
        self.rotation_filter = myKalmanFilterRot(self.initial_x, self.P, self.R, self.Q, self.Q_scale_factor, self.rotflip_const)
        n = noisy_rotations.shape[0]
        for i in range(n):
            measurement = noisy_rotations[i]
            if i == 0:
                self.rotation_filter.apply_first_measurement(measurement)
                self.rotation_filter.state_estimate()
                continue
            self.rotation_filter.filter_step(measurement)
        self.rotation_filter.convert_estimates()
        self.x_ = self.rotation_filter.estimates
        return self.x_


if __name__ == "__main__":
    gt = np.load("./dataset/circles/positions/circle2.npy")
    meas = np.load("./dataset/circles/positions/circle_noise2.npy")
    X= meas
    """
    param_grid = dict()
    with open("/home/frederik/Prace/Brigada2023/mygym/myGym/configs/parameters.json") as json_file:
        data = json.load(json_file)

        print("Type:", type(data))
        param_grid = data["ParticleFilter6D"] 
    """
    param_grid = {
        "Q" : [0.08, 0.04],
        "R" : [0.02, 0.15],
        "eps_max" : [0.18],
        "Q_scale_factor": [500, 1000],
    }
    #print(param_grid)
    cv = [(slice(None), slice(None))]
    #scorer = make_scorer(scoring, greater_is_better = True)
    grid_search = GridSearchCV(EstimatorKalman(), param_grid = param_grid, scoring = "neg_mean_squared_error", cv = cv)
    print(grid_search.fit(X, gt))
    print(grid_search.predict(X))
    print(grid_search.cv_results_)
    dataframe = pd.DataFrame(grid_search.cv_results_)[[ "param_Q", "param_R", "param_eps_max", "param_Q_scale_factor",  "mean_test_score"]]
    dataframe = dataframe.rename(columns = {"param_num_particles": "Numberof particles", "param_process_std": "Process std"})
    #dataframe = pd.read_csv("/home/frederik/Prace/Brigada2023/mygym/myGym/out1.csv", sep = '\t')
    #print("dataframe:", dataframe)
    dataframe = dataframe.sort_values(by="mean_test_score")
    dataframe.to_csv("outKalman.csv", sep='\t')
