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


class myEstimator(BaseEstimator):

    def fit(self, X, y=None):
        self.x_ = X
        for i in range(len(y)):
            if i == 0:
                self.y_ = y
            else:
                self.y_ = np.vstack((self.y_, y[i]))
        return self

    def predict(self, X):
        for t in range(len(X)):
            noisy_data = X[t]
            self.filter = self.initialize_filter()
            n = noisy_data.shape[0]

            for i in range(n):
                measurement = noisy_data[i]
                if i == 0:
                    self.filter.apply_first_measurement(measurement)
                    self.filter.state_estimate()
                    continue
                self.filter.filter_step(measurement)
            self.filter.convert_estimates()
            estimates = self.filter.estimates
            if t == 0:
                self.x_ = estimates
            else:
                self.x_ = np.vstack((self.x_, estimates))
        return self.x_


    def initialize_filter(self):
        self.filter = ParticleFilter6D(100, 0.02, 0.1,
                                       0.03, res_g=0.5)



class Estimator6D(myEstimator):
    def __init__(self, num_particles = 600, process_std = 0.02, vel_std = 0.1, measurement_std = 0.02, res_g = 0.5):
        self.vel_std = vel_std
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.res_g = res_g


    def initialize_filter(self):
        self.filter = ParticleFilter6D(self.num_particles, self.process_std, self.vel_std,
                                       self.measurement_std, res_g=self.res_g)





class Estimator6DRot(myEstimator):
    def __init__(self, num_particles = 600, process_std = 0.02, vel_std = 0.1, measurement_std = 0.02, res_g =0.5,
                 rotflip_const = 0.1):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.res_g = res_g
        self.vel_std = vel_std
        self.rotflip_const = rotflip_const


    def initialize_filter(self):
        self.filter = ParticleFilter6DRot(
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
        self.position_filter = ParticleFilterGH(self.num_particles, self.process_std,
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
                                                        self.measurement_std, Q=self.Q, R=self.R, factor_a=self.std_a)

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
        self.initial_x = np.array([0, 0, 0, 0, 0, 0])
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
        self.initial_x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.rotflip_const = rotflip_const

    def initialize_filter(self):
        self.filter = myKalmanFilterRot(self.initial_x, self.P, self.R, self.Q, self.Q_scale_factor,
                                                 self.rotflip_const)



def save_results(Estimator, dataframe):
    filter = Estimator.__class__.__name__
    if filter == "EstimatorKalman":
        dataframe.to_csv("results_Kalman_csv", sep='\t')


def create_dataframe(Estimator, grid_search, param_grid):
    filter_name = Estimator.__class__.__name__
    dataframe_list = []
    rename_list = []
    for key in param_grid:
        rename_list.append(key)
        dataframe_list.append("param_" + key)
    dataframe = pd.DataFrame(grid_search.cv_results_)[dataframe_list]
    rename_dict = {}
    for i in range(len(rename_list)):
        rename_dict[dataframe_list[i]] = rename_list[i]
    dataframe = dataframe.rename(collumns = rename_dict)
    dataframe = dataframe.sort_values(by=['mean_test_score'])
    dataframe.to_csv("results_" + filter_name, sep='\t')



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
