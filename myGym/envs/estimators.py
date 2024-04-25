from ast import arg
import gym
from myGym import envs
from myGym.train import get_parser, get_arguments, configure_implemented_combos, configure_env

from sklearn.base import BaseEstimator
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



def scoring(estimator, X, y):
    #TODO: Implement MSE and runtime calculation
    pass




class Estimator6D(BaseEstimator):
    def __init__(self,num_particles = 600, process_std = 0.02, vel_std = 0.1, measurement_std = 0.02):
        self.vel_std = vel_std
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std


    def fit(self, X, y):
        positions = y
        noisy_positions = X
        self.x_ = noisy_positions
        self.position_filter = ParticleFilter6D(self.num_particles, self.process_std, self.vel_std, self.measurement_std)

        n = noisy_data.shape[0]
        for i in range(n):
            measurement = noisy_positions[i]
            if i == 0:
                position_filter.apply_first_measurement(measurement)
                position_filter.state_estimate()
                continue
            position_filter.filter_step(position_filter.elements)
        position_filter.convert_estimates()
        self.y_ = position_filter.estimates
        return self




if __name__ == "__main__":
    print("hooray")