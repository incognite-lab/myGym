"""Module for particle filter to estimate 6D object pose in mygym environment"""

import numpy as np
import filterpy
from filterpy.monte_carlo import residual_resample, stratified_resample, systematic_resample
import scipy


class ParticleFilter3D(object):
    """Custom Particle filter class for estimating position or orientation of 3D objects from noisy measurements"""
    def __init__(self, num_particles, process_std = 5, measurement_std = 5, workspace_bounds = None, dt = 0.1, g = 0.5, h = 0.5):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.estimate = np.zeros(3) #Actual state estimate - this is our tracked value
        self.estimate_var = 20 #Large value in the beginning
        if workspace_bounds is None:
            self.workspace_bounds = [(-5,5),(0.3, 5), (0, 8)] #Bounds for movement in meters
        else:
            self.workspace_bounds = workspace_bounds
        self.vel = np.zeros(3) #process velocity, i.e. position or angular velocity determined externally from observed data
        self.vel_std = 0.01 #Unknown velocity in the beginning
        self.particles = self.create_uniform_particles()
        self.dt = dt
        self.weights = np.ones(self.num_particles)/self.num_particles #Uniform weights initialization
        self.estimates = []
        self.estimate_vars = []
        self.g = g
        self.h = h
        self.a = np.zeros(3)


    def create_uniform_particles(self):
        """Create uniform particles in workspace bounds"""
        particles = np.empty((self.num_particles, 3))
        N = self.num_particles
        x_range = self.workspace_bounds[0]
        y_range = self.workspace_bounds[1]
        z_range = self.workspace_bounds[2]
        particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = np.random.uniform(z_range[0], z_range[1], size=N)
        return particles


    def apply_first_measurement(self, measurement):
        """Determine initial values based on first measurement"""
        self.estimate = measurement
        initial_uncertainty_factor = 5  #multiply std by this factor to account for initial uncertainty to better address
        #system nonlinearities
        self.particles = self.create_gaussian_particles(self.measurement_std*initial_uncertainty_factor)


    def create_gaussian_particles(self, std):
        """Create normally distributed particles around estimate with given std"""
        N = self.num_particles
        particles = np.empty((N, 3))
        for i in range(3):
            particles[:, i] = self.estimate[i] + (np.random.randn(N) * std)
        return particles


    def predict(self):
        """Move particles based on estimated velocity"""
        n = self.num_particles
        vel = self.vel + np.random.randn(3)*self.vel_std #Adding velocity noise
        self.particles[:, :3] += vel*self.dt  #Predict + process noise
        self.particles[:, 0] += np.random.randn(n)* self.process_std
        self.particles[:, 1] += np.random.randn(n) * self.process_std
        self.particles[:, 2] += np.random.randn(n) * self.process_std


    def update(self, z):
        """Update particle weights based on the measurement likelihood of each particle"""
        distance_pos = np.linalg.norm(self.particles[:, 0:3] - z[0:3], axis = 1)
        #print(distance_pos)
        self.weights *= scipy.stats.norm(0, self.measurement_std).pdf(distance_pos) #Multiply weights based on the
        #particle's Gaussian probability
        self.weights += 1.e-300 #Avoid round-off to zero
        self.weights = self.weights / np.sum(self.weights) #Normalization
        #Update vel and acceleration
        #self.vel = (1 -self.g) * self.vel + ((z - self.estimate)/self.dt) * self.g#velocity change


    def state_estimate(self):
        """Update estimated state and std based on weighted average of particles"""
        last_estimate = self.estimate
        self.estimate = np.average(self.particles, weights = self.weights, axis = 0) #Weighted average of particles
        var = np.average(((self.particles - self.estimate)**2), weights = self.weights, axis = 0)
        self.estimate_var = var
        self.estimates.append(self.estimate)
        self.estimate_vars.append(self.estimate_var)
        last_vel = self.vel
        self.vel = last_vel * self.g + (1-self.g)*(self.estimate - last_estimate)/self.dt
        self.a = self.h * self.a + (1-self.h)*(self.vel - last_vel)/self.dt
        self.vel += self.a * self.dt


    def resample(self):
        """Resample particles based on their contribution to estimate. Algorithm used is stratified resample"""
        #print("weights", self.weights)
        indexes = filterpy.monte_carlo.systematic_resample(self.weights)
        #print("indexes", indexes)

        self.particles[:] = self.particles[indexes]
        self.weights.fill(1/self.num_particles)


    def filter_step(self, z):
        """One full step of the filter"""
        self.predict()
        self.update(z)
        self.state_estimate()
        self.resample()



