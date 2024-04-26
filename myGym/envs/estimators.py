from ast import arg
import gym
from myGym import envs
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

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



def scoring(y_filter, y_gt):
    #TODO: Implement MSE and runtime calculation
    squared_error = np.square(np.linalg.norm(y_gt - y_filter, axis = 1))
    MSE = np.mean(squared_error)
    return MSE



class Estimator6D(BaseEstimator):
    def __init__(self, num_particles = 600, process_std = 0.02, vel_std = 0.1, measurement_std = 0.02, res_g = 0.5):
        self.vel_std = vel_std
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.res_g = res_g


    def fit(self, X, y):
        check_X_y(X, y)
        return self



    def predict(self, X, Y):
        noisy_positions = X
        self.x_ = noisy_positions
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
        self.y_ = self.position_filter.estimates
        return self





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
        X, y = check_X_y(X, y)
        noisy_rots = X
        self.x_ = noisy_rots
        self.rotation_filter = ParticleFilter6DRot(
            self.num_particles, self.process_std, self.vel_std, self.measurement_std, res_g=self.res_g,
            rotflip_const=self.rotflip_const)
        n = noisy_data.shape[0]
        for i in range(n):
            measurement = noisy_positions[i]
            if i == 0:
                self.rotation_filter.apply_first_measurement(measurement)
                self.rotation_filter.state_estimate()
                continue
            self.rotation_filter.filter_step(position_filter.elements)
        self.rotation_filter.convert_estimates()
        self.y_ = rotation_filter.estimates
        return self


    def predict(self, X):
        check_is_fitted(self)





class EstimatorGH(BaseEstimator):
    def __init__(self, num_particles=600, process_std=0.02, measurement_std=0.02, g = 0.5, h = 0.5):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.g = g
        self.h = h

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        noisy_rots = X
        self.x_ = noisy_rots
        self.position_filter = ParticleFilterGH(self.num_particles, self.process_std,
                                                self.measurement_std, vel_std = 0.01, g = self.g, h = self.h)
        n = noisy_data.shape[0]
        for i in range(n):
            measurement = noisy_positions[i]
            if i == 0:
                self.rotation_filter.apply_first_measurement(measurement)
                self.rotation_filter.state_estimate()
                continue
            self.rotation_filter.filter_step(position_filter.elements)
        self.rotation_filter.convert_estimates()
        self.y_ = rotation_filter.estimates
        return self




if __name__ == "__main__":
    gt = np.load("./dataset/splines/positions/spline2.npy")
    meas = np.load("./dataset/splines/positions/spline_noise2.npy")
    X= [gt, meas]
    #estimator = Estimator6D()
    #estimator.fit(X, y)
    param_grid = {
        "num_particles" : [600],
        "process_std" : [0.02],
        "measurement_std" : [0.02],
        "vel_std" : [0.1, 0.2],
        "res_g" : [0.5],
    }
    scorer = make_scorer(scoring, greater_is_better = True)
    grid_search = GridSearchCV(Estimator6D(), param_grid = param_grid, scoring = scorer)
    print(grid_search.fit(X))
    print(grid_search.cv_results_)

