"""Module for particle filter to estimate 6D object pose in mygym environment"""

import numpy as np
import filterpy
from filterpy.monte_carlo import residual_resample, stratified_resample, systematic_resample
import scipy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class MOTFilter:
    """
    Wrapper class that combines multiple filters and their actions for MOT
    """
    def __init__(self, object_classes, filter_type, filter_params, classes_mapping):
        self.object_classes = object_classes
        self.pos_filters = [] #array for position filters
        self.rot_filters = [] #array for rotation filters
        self.initialized_classes = []
        self.filter_type = filter_type
        self.filter_params = filter_params
        self.classes_mapping = classes_mapping
        self.weights = None
        self.particles = None


    def initialize_1_filter(self, measurement_class):
        #TODO: initialize filter with measurement class and save it to self.initialized_classes
        pos_filter, rot_filter = self.filter_init()
        pos_filter.set_object_class(measurement_class)
        rot_filter.set_object_class(measurement_class)
        self.initialized_classes.append(measurement_class)
        self.pos_filters.append(pos_filter)
        self.rot_filters.append(rot_filter)

    def predict(self, time):
        """Predict with all initialized_filters."""
        for i in range(len(self.pos_filters)):
            self.pos_filters[i].predict_MOT(time)
            self.rot_filters[i].predict_MOT(time)


    def map_class_weights(self, measured_class):
        """
        From measured object class, retrieve weight related to probability of given filter class being in accordance
        with the measured class
        """
        class_vector = np.zeros(self.classes_mapping.shape[0])
        class_vector[measured_class - 1] = 1
        class_weights = self.classes_mapping@class_vector
        return class_weights


    def update(self, z):
        """
        JPDA update considering all initialized_filters
        """
        all_particle_set = self.get_particle_set()
        distance_pos = np.linalg.norm(all_particle_set[:, 0:3] - z[0:3], axis=1)
        weights = np.ones_like(distance_pos)/len(distance_pos) #initialize equal weights
        weights *= scipy.stats.norm(0, self.measurement_std).pdf(distance_pos)
        weights += 1.e-300  # Avoid round-off to zero
        weights = weights / np.sum(weights)  # Normalization
        measured_class = z[3]
        class_weights = self.map_class_weights(measured_class)
        last_weight_index = 0
        for i in range(len(self.pos_filters)):
            pos_filter = self.pos_filters[i]
            class_weight = class_weights[pos_filter.object_class - 1]
            pos_filter.weights = weights[:pos_filter.num_particles]
            pos_filter.weights *= class_weight
            weights[last_weight_index:last_weight_index + pos_filter.num_particles] = pos_filter.weights
            last_weight_index += pos_filter.num_particles
        self.weights = weights
        self.particles = all_particle_set


    def local_updates(self):
        #TODO: maybe perform local updates for all filters?
        pass

    def state_estimate(self):
        estimate = np.average(self.particles, weights=self.weights, axis=0)
        return estimate


    def resample(self):
        indexes = filterpy.monte_carlo.systematic_resample(self.weights)
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1 / self.self.particles.shape[0])
        self.load_global_particles_into_filters()


    def load_global_particles_into_filters(self):
        last_index = 0
        for i in range(len(self.pos_filters)):
            pos_filter = self.pos_filters[i]
            pos_filter.particles = self.particles[last_index:last_index + pos_filter.num_particles]
            last_index += pos_filter.num_particles




    def get_particle_set(self):
        """
        Combine positional particles from all initialized_filters into one array
        """
        particles_list = []
        for i in range(len(self.pos_filters)):
            particles_list.append(self.pos_filters[i].particles[:, :3])
        particles = tuple(particles_list)
        particle_set = np.vstack(particles)
        return particle_set

    def filter_init(self):
        pos_filter = None
        rot_filter = None
        if self.filter_type == 'ParticleFilterGH':
            pos_filter = ParticleFilterGH(self.filter_params["num_particles"], self.filter_params["process_std"],
                                      self.filter_params["measurement_std"],
                                      g = self.filter_params["g"], h = self.filter_params["h"])
            rot_filter = ParticleFilterGHRot(self.filter_params["num_particles"], self.filter_params["process_std"],
                                      self.filter_params["measurement_std"],
                                      g = self.filter_params["g"], h = self.filter_params["h"],
                                             rotflip_const=self.filter_params["rotflip_const"])
        elif self.filter_type == 'ParticleFilterVelocity':
            pos_filter = ParticleFilterVelocity(self.filter_params["num_particles"], self.filter_params["process_std"],
                                            self.filter_params["vel_std"], self.filter_params["measurement_std"],
                                            res_g = self.filter_params["res_g"])
            rot_filter = ParticleFilterVelocityRot(self.filter_params["num_particles"], self.filter_params["process_std"],
                                            self.filter_params["vel_std"], self.filter_params["measurement_std"],
                                            res_g = self.filter_params["res_g"], rotflip_const="rotflip_const")
        elif self.filter_type == 'ParticleFilterWithKalman':
            pos_filter = ParticleFilterWithKalman(self.filter_params["num_particles"], self.filter_params["process_std"],
                                              self.filter_params["measurement_std"], Q =self.filter_params["Q"],
                                              R = self.filter_params["R"], std_a = self.filter_params["std_a"])
            rot_filter = ParticleFilterWithKalmanRot(self.filter_params["num_particles"], self.filter_params["process_std"],
                                              self.filter_params["measurement_std"], Q =self.filter_params["Q"],
                                              R = self.filter_params["R"], factor_a = self.filter_params["std_a"],
                                                     rotflip_const=self.filter_params["rotflip_const"])
        return pos_filter, rot_filter





class myKalmanFilter():
    """Returns a Kalman filter which implements constant velocity model"""
    def __init__(self, x, P, R, Q= 0.08, dt=0.2, Q_scale_factor = 1000, eps_max = 0.18):
        self.filter = KalmanFilter(dim_x=x.shape[0], dim_z=3)
        self.filter.x = np.array(x)
        self.filter.F = np.array([[1, dt, 0, 0, 0 ,0 ],
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, dt, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, dt],
                                  [0, 0, 0, 0, 0, 1]])
        self.filter.H = np.array([[1, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0]])  # Measurement function
        self.filter.R *= R  # measurement uncertainty
        if np.isscalar(P):
            self.filter.P *= P  # covariance matrix
        else:
            self.filter.P[:] = P  # [:] makes deep copy
        if np.isscalar(Q):
            Q1dim = Q_discrete_white_noise(dim=2, dt=dt, var=Q) #Discrete white noise for one dimension
            self.filter.Q = np.zeros((6, 6))
            self.filter.Q[:2, :2] = Q1dim
            self.filter.Q[2:4, 2:4] = Q1dim
            self.filter.Q[4:6, 4:6] = Q1dim
        else:
            self.filter.Q[:] = Q
        self.estimate = np.zeros(3)
        self.estimate_vars = []
        self.estimates = []
        # For adaptive filtering below:
        self.eps_max = eps_max #CONSTANT
        self.Q_scale_factor = Q_scale_factor #CONSTANT
        self.count = 0


    def get_est(self):
        state = np.array([self.filter.x[0], self.filter.x[2], self.filter.x[4]])
        return state


    def apply_first_measurement(self, measurement):
        self.filter.x[0] = measurement[0]
        self.filter.x[2] = measurement[1]
        self.filter.x[4] = measurement[2]


    def predict(self):
        self.filter.predict()


    def update(self, z):
        self.filter.update(z)

    def state_estimate(self):
        self.estimate = self.get_est()
        self.estimates.append(self.estimate)
        self.estimate_vars.append(self.filter.P)

    def filter_step(self, z):
        self.predict()
        self.update(z)
        self.state_estimate()
        #Adaptive filtering
        y, S = self.filter.y, self.filter.S
        eps = y.T @ np.linalg.inv(S) @ y

        if eps > self.eps_max:
            #print("scaling Q, count: ", self.count)
            self.filter.Q *= self.Q_scale_factor
            self.count += 1
        elif self.count >0:
            self.filter.Q /= self.Q_scale_factor
            self.count -=1

    def convert_estimates(self):
        """Converts list of estimates into numpy estimate array"""
        estimates = self.estimates
        self.estimates = np.zeros((len(estimates), 3))
        for i in range(len(estimates)):
            self.estimates[i, :] = estimates[i]






class myKalmanFilterRot(myKalmanFilter):
    """To make computations in quaternion, filter needs to be have four spatial dimensions"""
    def __init__(self, x, P, R, Q= 0., dt=0.2, rotflip_const =0.1, eps_max = 0.18, Q_scale_factor=1000):
        self.filter = KalmanFilter(dim_x=x.shape[0], dim_z=4)
        self.filter.x = np.array(x)
        self.filter.F = np.array([[1, dt, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, dt, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, dt, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, dt],
                                  [0, 0, 0, 0, 0, 0, 0, 1]])
        self.filter.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0]])  # Measurement function
        self.filter.R *= R  # measurement uncertainty
        if np.isscalar(P):
            self.filter.P *= P  # covariance matrix
        else:
            self.filter.P[:] = P  # [:] makes deep copy
        if np.isscalar(Q):
            Q1dim = Q_discrete_white_noise(dim=2, dt=dt, var=Q)  # Discrete white noise for one dimension
            self.filter.Q = np.zeros((8, 8))
            self.filter.Q[:2, :2] = Q1dim
            self.filter.Q[2:4, 2:4] = Q1dim
            self.filter.Q[4:6, 4:6] = Q1dim
            self.filter.Q[6:8, 6:8] = Q1dim
        else:
            self.filter.Q[:] = Q
        self.estimate = np.zeros(4)
        self.estimate_vars = []
        self.estimates = []
        # For adaptive filtering below:
        self.eps_max = eps_max
        self.Q_scale_factor = Q_scale_factor
        self.count = 0
        self.rotation_flip_const = rotflip_const


    def get_est(self):
        state = np.array([self.filter.x[0], self.filter.x[2], self.filter.x[4], self.filter.x[6]])
        return state


    def flip_est(self):
        self.filter.x = -self.filter.x


    def apply_first_measurement(self, measurement):
        self.filter.x[0] = measurement[0]
        self.filter.x[2] = measurement[1]
        self.filter.x[4] = measurement[2]
        self.filter.x[6] = measurement[3]


    def update(self, z):

        norm = np.linalg.norm(z + self.get_est())
        #print("Norm: ", norm)
        if norm < self.rotation_flip_const:
            self.flip_est()
        self.filter.update(z)

    def convert_estimates(self):
        """Converts list of estimates into numpy estimate array"""
        estimates = self.estimates
        self.estimates = np.zeros((len(estimates), 4))
        for i in range(len(estimates)):
            self.estimates[i, :] = estimates[i]


class ParticleFilter(object):
    """Base particle filter parent class. Contains method and parameters used in every type of particle filter"""
    def __init__(self, num_particles, process_std, measurement_std, workspace_bounds = None, dt = 0.2):
        self.num_particles = num_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        if workspace_bounds is None:
            self.workspace_bounds = [(-5,5),(0.3, 5), (0, 8)] #Bounds for movement in meters (only for 3D filters)
        else:
            self.workspace_bounds = workspace_bounds
        self.weights = np.ones(self.num_particles) / self.num_particles  # Uniform weights initialization
        self.estimates = [] #List for estimates
        self.estimate_vars = [] #List for estimate variances
        self.dt = dt
        self.vel = np.random.normal([0, 0, 0], [self.process_std]*3, (1, 3))
        self.particles = None
        self.estimate = np.zeros(3)
        self.current_time = 0 #Keeps track of current filtering time, used in MOT
        self.last_predict_time = 0
        self.object_class = None
        self.weight_contributions = []



    def set_object_class(self, obj_class):
        self.object_class = obj_class

    def apply_first_measurement(self, measurement):
        """Determine initial values based on first measurement"""
        self.estimate = measurement
        initial_uncertainty_factor = 3  # multiply std by this factor to account for initial uncertainty to better address
        # system nonlinearities
        self.particles = self.create_gaussian_particles(self.measurement_std * initial_uncertainty_factor)


    def create_uniform_particles(self):
        """Create uniform particles in workspace bounds of a given dimension"""
        particles = np.empty((self.num_particles, 3))
        N = self.num_particles
        x_range = self.workspace_bounds[0]
        y_range = self.workspace_bounds[1]
        z_range = self.workspace_bounds[2]
        particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = np.random.uniform(z_range[0], z_range[1], size=N)
        return particles


    def create_gaussian_particles(self, std):
        """Create gaussian particles around the estimate"""
        N = self.num_particles
        particles = np.empty((N, 3))
        for i in range(3):
            particles[:, i] = self.estimate[i] + (np.random.randn(N) * std)
        return particles


    def predict(self):
        """Move particles based on estimated velocity"""
        n = self.num_particles
        vel = self.velocity_function() # Determining current velocity of particles for prediction
        self.particles[:, :3] += vel * self.dt  # Predict + process noise
        self.particles[:, 0] += np.random.randn(n) * self.process_std
        self.particles[:, 1] += np.random.randn(n) * self.process_std
        self.particles[:, 2] += np.random.randn(n) * self.process_std


    def velocity_function(self):
        """Calculate velocity according to the filter characteristics"""
        return self.vel


    def update(self, z):
        """Update particle weights based on the measurement likelihood of each particle"""
        distance_pos = np.linalg.norm(self.particles[:, 0:3] - z[0:3], axis=1)
        self.weights *= scipy.stats.norm(0, self.measurement_std).pdf(distance_pos)  # Multiply weights based on the
        # particle's Gaussian probability
        self.weights += 1.e-300  # Avoid round-off to zero
        self.weights = self.weights / np.sum(self.weights)  # Normalization


    def state_estimate(self):
        """Update estimated state and std based on weighted average of particles"""
        self.estimate = np.average(self.particles, weights = self.weights, axis = 0) #Weighted average of particles
        var = np.average(((self.particles - self.estimate)**2), weights = self.weights, axis = 0)
        self.estimates.append(self.estimate) #Store each estimate in a list
        self.estimate_vars.append(var) #Store each estimate variation in a list


    def index_resample_function(self):
        return filterpy.monte_carlo.systematic_resample(self.weights)


    def resample(self):
        """Resample particles based on their contribution to estimate. Algorithm used is systematic resample"""
        indexes = self.index_resample_function()
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1/self.num_particles)


    def filter_step(self, z):
        """One full step of the filter."""
        self.predict()
        self.update(z)
        if self.neff() < self.num_particles/30:
            self.resample()
            self.reapply_measurement(z)
            self.state_estimate()
        elif self.neff() < self.num_particles/2:
            self.state_estimate()
            self.resample()
        else:
            self.state_estimate()


    def neff(self):
        return 1. / np.sum(np.square(self.weights))


    def reapply_measurement(self, z):

        """When measurement is too far from prediction, move particles to the measurement"""
        self.estimate = z
        uncertainty_factor = 2  # multiply std by this factor to account for initial uncertainty to better address
        # system nonlinearities
        N = self.num_particles
        particles = np.empty((N, 3))
        for i in range(3):
            particles[:, i] = self.estimate[i] + (np.random.randn(N) * self.measurement_std * uncertainty_factor)  # Position of particles
        self.particles[:, :3] = particles


    def convert_estimates(self):
        """Converts list of estimates into numpy estimate array"""
        estimates = self.estimates
        self.estimates = np.zeros((len(estimates), self.particles.shape[1]))
        for i in range(len(estimates)):
            self.estimates[i, :] = estimates[i]


    def predict_MOT(self, time):
        """
        Predict part of MOT algorithm - collects all measurements from buffer and does predict based on all of them
        """
        self.dt = time -self.last_predict_time #Time elapsed since last filter predict/update
        self.predict()


    def update_MOT(self, measurement, meas_class):
        if self.object_class == meas_class:
            gamma = 1
        else:
            gamma = 0.05
        distance_pos = np.linalg.norm(self.particles[:, 0:3] - measurement[0:3], axis=1)
        weights_contrib = scipy.stats.norm(0, self.measurement_std).pdf(distance_pos) * gamma
        self.weights += scipy.stats.norm(0, self.measurement_std).pdf(distance_pos) * gamma  # Multiply weights based on the
        # particle's Gaussian probability
        self.weights += 1.e-300  # Avoid round-off to zero
        return np.sum(weights_contrib)

    def state_estimate_MOT(self):
        # self.dt = time - self.last_predict_time
        # if self.dt < 0.0001:
        #     self.dt = 0.01
        # print("performing state estimation at time", time, "last_predict_time", self.last_predict_time, "that makes dt", self.dt)
        # self.last_predict_time = time
        #self.weights = self.weights / np.sum(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        self.state_estimate()

    def MOT_step_finish(self, time):
        self.weights = self.weights / np.sum(self.weights)
        #self.state_estimate_MOT()

    def filter_step_MOT(self, measurement_buffer):
        pass


    def reset_weights(self):
        zero_hypothesis_weight = scipy.stats.norm(0, self.measurement_std).pdf(self.measurement_std * 4)
        #print("zero_hypothesis_weight", zero_hypothesis_weight)

        self.weights.fill(1/self.num_particles)




class ParticleFilterVelocity(ParticleFilter):
    """
    Particle filter which includes particle velocities for better prediction.
    Each particle has its velocity, which changes only based on noise. Those particles, that move correctly in
    the direction of the velocity get to be resampled. This simulates acceleration.
    """
    def __init__(self, num_particles, process_std, vel_std, measurement_std, workspace_bounds = None, dt = 0.2, res_g = 0.5):
        super().__init__(num_particles, process_std, measurement_std, workspace_bounds, dt)
        self.vel_std = vel_std
        self.vel_bounds = [(-1, 1), (-1, 1), (-1, 1)] #Max velocity of a particle for uniform particles
        self.particles = self.create_uniform_particles()
        # Below there are attributes used for computing the moving average of measured acceleration
        #self.last_measured_distances = np.zeros((self.moving_avg_length+1, 3))
        self.last_measured_position = None
        self.vel_estimate = np.zeros(3)
        self.process_std_const = process_std
        self.vel_std_const = vel_std
        self.prev_residual = np.zeros(3)
        self.res_g = res_g



    def compute_residual(self, measurement):
        """
        Compute the residual (difference between predicted position and measured position) from particles
        This is done for adaptive filtering - increase uncertainty in prediction when the track seems to be changing.
        """
        predicted_state = np.average(self.particles[:, :3], weights=self.weights, axis=0)
        new_residual = measurement - predicted_state
        residual = self.res_g*new_residual + (1 - self.res_g)*self.prev_residual
        self.prev_residual = residual
        return residual


    def apply_first_measurement(self, measurement):
        """Determine initial values based on first measurement"""
        self.estimate = measurement
        initial_uncertainty_factor = 3  #Multiply std by this factor to account for initial uncertainty to better address
        #System nonlinearities
        self.last_measured_position = self.estimate
        self.particles = self.create_gaussian_particles(self.measurement_std * initial_uncertainty_factor)


    def create_uniform_particles(self):
        """Create uniform particles in workspace bounds of a given dimension including velocity"""
        particles = np.empty((self.num_particles, 6))
        N = self.num_particles
        for i in range(3):
            bounds = self.workspace_bounds[i]
            particles[:, i] = np.random.uniform(*bounds, size = N)
        for i in range(3):
            bounds = self.vel_bounds[i]
            particles[:, i+ 3] = np.random.uniform(*bounds, size = N)
        return particles



    def create_gaussian_particles(self, std):
        """Create gaussian particles in workspace bounds of a given dimension"""
        N = self.num_particles
        particles = np.empty((N, 6))
        #print(self.estimate)
        for i in range(3):
            particles[:, i] = self.estimate[i] + (np.random.randn(N) * std) #Position of particles
        particles[:, 3:] = (np.random.randn(N, 3) * std) #Velocities of particles initialized to Gaussian noise around zero
        return particles


    def velocity_function(self):
        """Add noise to current velocity"""
        current_velocities = self.particles[:, 3:]
        noise = np.random.randn(*current_velocities.shape)*self.vel_std
        self.particles[:, 3:] = current_velocities + noise
        return current_velocities + noise


    def index_resample_function(self):
        return filterpy.monte_carlo.systematic_resample(self.weights)


    def state_estimate(self):
        """Update estimated state and std based on weighted average of particles"""
        self.estimate = np.average(self.particles[:, :3], weights = self.weights, axis = 0) #Weighted average of particles
        var = np.average(((self.particles[:, :3] - self.estimate)**2), weights = self.weights, axis = 0)
        #self.vel_estimate = np.average(self.particles[:, 3:], weights = self.weights, axis = 0)
        self.estimates.append(self.estimate) #Store each estimate in a list
        self.estimate_vars.append(var) #Store each estimate variation in a list


    def update(self, z):
        """Update step of the particle filter."""
        residual = np.linalg.norm(self.compute_residual(z))
        super().update(z)
        self.process_std = self.process_std_const*residual*3 #Constant
        self.vel_std = 1000*self.vel_std_const*(residual**2) #Constant


    def convert_estimates(self):
        """Converts list of estimates into numpy estimate array"""
        estimates = self.estimates
        self.estimates = np.zeros((len(estimates), 3))
        for i in range(len(estimates)):
            self.estimates[i, :] = estimates[i]





class ParticleFilterVelocityRot(ParticleFilter):
    """Idea is the same as ParticleFilter6D but with additional dimension for rotation"""
    def __init__(self, num_particles, process_std, vel_std, measurement_std, workspace_bounds = None, dt = 0.2, res_g = 0.5, rotflip_const = 0.1):
        super().__init__(num_particles, process_std, measurement_std, workspace_bounds, dt)
        self.vel_std = vel_std
        self.vel_bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)] #Max velocity of a particle for uniform particles
        self.particles = None
        self.estimate = np.zeros(4)
        self.last_measured_position = None
        self.vel_estimate = np.zeros(4)
        self.process_std_const = process_std
        self.vel_std_const = vel_std
        self.last_residual = 0
        self.rotation_flip_const = rotflip_const #CONSTANT
        self.res_g = res_g



    def apply_first_measurement(self, measurement):
        """Determine initial values based on first measurement"""
        self.estimate = measurement
        initial_uncertainty_factor = 3  #Multiply std by this factor to account for initial uncertainty to better address
        #System nonlinearities
        self.last_measured_position = self.estimate
        self.particles = self.create_gaussian_particles(self.measurement_std * initial_uncertainty_factor)


    def compute_residual(self, measurement):
        """
        Compute the residual (difference between predicted position and measured position) from particles
        This is done for adaptive filtering - increase uncertainty in prediction when the track seems to be changing.
        """
        predicted_state = np.average(self.particles[:, :4], weights=self.weights, axis=0)

        residual = self.res_g* self.last_residual + (1 - self.res_g)* (measurement - predicted_state) #CONSTANT
        self.last_residual = residual
        return residual


    def create_uniform_particles(self):
        """Create uniform particles in workspace bounds of a given dimension including velocity"""
        particles = np.empty((self.num_particles, 8))
        N = self.num_particles
        for i in range(4):
            bounds = self.workspace_bounds[i]
            particles[:, i] = np.random.uniform(*bounds, size=N)
        for i in range(4):
            bounds = self.vel_bounds[i]
            particles[:, i + 3] = np.random.uniform(*bounds, size=N)
        return particles

    def create_gaussian_particles(self, std, vel = [0, 0, 0, 0]):
        """Create gaussian particles in workspace bounds of a given dimension"""
        N = self.num_particles
        particles = np.empty((N, 8))
        for i in range(4):
            particles[:, i] = self.estimate[i] + (np.random.randn(N) * std)  # Position of particles
        particles[:, 4:] = vel + np.random.randn(N, 4) * std  # Velocities of particles initialized to Gaussian noise around zero
        return particles

    def velocity_function(self):
        """Add noise to current velocity"""
        current_velocities = self.particles[:, 4:]
        noise = np.random.randn(*current_velocities.shape)*self.vel_std
        self.particles[:, 4:] = current_velocities + noise
        return current_velocities + noise


    def state_estimate(self):
        """Update estimated state and std based on weighted average of particles"""
        self.estimate = np.average(self.particles[:, :4], weights = self.weights, axis = 0) #Weighted average of particles
        var = np.average(((self.particles[:, :4] - self.estimate)**2), weights = self.weights, axis = 0)
        #self.vel_estimate = np.average(self.particles[:, 3:], weights = self.weights, axis = 0)
        self.estimates.append(self.estimate) #Store each estimate in a list
        self.estimate_vars.append(var) #Store each estimate variation in a list


    def predict(self):
        """Move particles based on estimated velocity"""
        n = self.num_particles
        vel = self.velocity_function() # Determining current velocity of particles for prediction
        self.particles[:, :4] += vel * self.dt  # Predict + process noise
        self.particles[:, 0] += np.random.randn(n) * self.process_std
        self.particles[:, 1] += np.random.randn(n) * self.process_std
        self.particles[:, 2] += np.random.randn(n) * self.process_std
        self.particles[:, 3] += np.random.randn(n) * self.process_std


    def update(self, z):
        """Update particle weights based on the measurement likelihood of each particle"""
        norm = np.linalg.norm(self.estimate + z)
        #print("norm:", norm)
        if norm < self.rotation_flip_const:
            self.reapply_measurement(z)
        distance_pos = np.linalg.norm(self.particles[:, 0:4] - z[0:4], axis=1)
        self.weights *= scipy.stats.norm(0, self.measurement_std).pdf(distance_pos)  # Multiply weights based on the
        # particle's Gaussian probability
        self.weights += 1.e-300  # Avoid round-off to zero
        self.weights = self.weights / np.sum(self.weights)  # Normalization


    def filter_step(self, z):
        """One full step of the filter."""
        self.predict()
        self.update(z)
        self.state_estimate()
        if self.neff() < self.num_particles/2:
            self.resample()


    def reapply_measurement(self, z):
        """Determine initial values based on first measurement"""
        vel = self.estimate_vel()
        self.estimate = z
        initial_uncertainty_factor = 1.2  # Multiply std by this factor to account for initial uncertainty to better address
        # System nonlinearities
        self.last_measured_position = self.estimate
        self.particles = self.create_gaussian_particles(self.measurement_std * initial_uncertainty_factor, -vel)


    def estimate_vel(self):
        """Update estimated state and std based on weighted average of particles"""
        vel_estimate = np.average(self.particles[:, 4:], weights=self.weights, axis=0)
        # Weighted average of particles
        return vel_estimate


    def convert_estimates(self):
        """Converts list of estimates into numpy estimate array"""
        estimates = self.estimates
        self.estimates = np.zeros((len(estimates), 4))
        for i in range(len(estimates)):
            self.estimates[i, :] = estimates[i]



class ParticleFilterGH(ParticleFilter):
    """Custom Particle filter class for estimating position or orientation of 3D objects from noisy measurements"""
    def __init__(self, num_particles, process_std, measurement_std, vel_std = 0.01, workspace_bounds = None, dt = 0.2, g=0.5, h=0.5):
        super().__init__(num_particles, process_std, measurement_std, workspace_bounds, dt)
        self.estimate = np.zeros(3) #Actual state estimate - this is our tracked value
        self.estimate_var = 20 #Large value in the beginning
        self.vel = np.zeros(3) #process velocity, i.e. position or angular velocity determined externally from observed data
        self.vel_std = vel_std
        self.particles = self.create_uniform_particles()
        self.dt = dt
        self.weights = np.ones(self.num_particles)/self.num_particles #Uniform weights initialization
        self.estimates = []
        self.estimate_vars = []
        self.g = g
        self.h = h
        self.a = np.zeros(3)


    def create_uniform_particles(self):
        """Creates uniform particles in workspace"""
        return super().create_uniform_particles()


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
        self.weights *= scipy.stats.norm(0, self.measurement_std).pdf(distance_pos) #Multiply weights based on the
        #particle's Gaussian probability
        self.weights += 1.e-300 #Avoid round-off to zero
        self.weights = self.weights / np.sum(self.weights) #Normalization


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
        indexes = filterpy.monte_carlo.systematic_resample(self.weights)
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1/self.num_particles)



class ParticleFilterGHRot(ParticleFilter):

    def __init__(self, num_particles, process_std, measurement_std, workspace_bounds = None, vel_std = 0.01, dt = 0.2, g=0.5, h=0.5, rotflip_const = 0.1):
        super().__init__(num_particles, process_std, measurement_std, workspace_bounds, dt)
        self.estimate = np.zeros(4) #Actual state estimate - this is our tracked value
        self.estimate_var = 20 #Large value in the beginning
        self.vel = np.zeros(4) #process velocity, i.e. position or angular velocity determined externally from observed data
        self.vel_std = 0.01 #Unknown velocity in the beginning
        self.particles = self.create_uniform_particles()
        self.dt = dt
        self.weights = np.ones(self.num_particles)/self.num_particles #Uniform weights initialization
        self.estimates = []
        self.estimate_vars = []
        self.g = g
        self.h = h
        self.a = np.zeros(4)
        self.rotation_flip_const = rotflip_const
        self.print_data = False


    def create_gaussian_particles(self, std):
        """Create normally distributed particles around estimate with given std"""
        N = self.num_particles
        particles = np.empty((N, 4))
        for i in range(4):
            particles[:, i] = self.estimate[i] + (np.random.randn(N) * std)
        return particles


    def predict(self):
        """Move particles based on estimated velocity"""
        n = self.num_particles
        vel = self.vel + np.random.randn(4)*self.vel_std #Adding velocity noise
        self.particles[:, :4] += vel*self.dt  #Predict + process noise
        self.particles[:, 0] += np.random.randn(n)* self.process_std
        self.particles[:, 1] += np.random.randn(n) * self.process_std
        self.particles[:, 2] += np.random.randn(n) * self.process_std
        self.particles[:, 3] += np.random.randn(n) * self.process_std


    def update(self, z):
        """Update particle weights based on the measurement likelihood of each particle"""
        norm = np.linalg.norm(self.estimate + z)
        if norm < self.rotation_flip_const:
            self.reapply_measurement(z)
            self.print_data = True
        distance_pos = np.linalg.norm(self.particles[:, 0:4] - z[0:4], axis = 1)
        self.weights *= scipy.stats.norm(0, self.measurement_std).pdf(distance_pos) #Multiply weights based on the
        #particle's Gaussian likelihood
        self.weights += 1.e-300 #Avoid round-off to zero
        self.weights = self.weights / np.sum(self.weights) #Normalization


    def filter_step(self, z):
        """One full step of the filter."""
        self.predict()
        self.update(z)
        self.state_estimate()
        if self.neff() < self.num_particles/2:
            self.resample()


    def reapply_measurement(self, z):
        uncertainty_factor = 1.2
        self.estimate = z
        self.vel = 0
        self.a = 0
        self.particles = self.create_gaussian_particles(uncertainty_factor* self.measurement_std)


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
        if self.print_data:
            self.print_data = False

    def convert_estimates(self):
        """Converts list of estimates into numpy estimate array"""
        estimates = self.estimates
        self.estimates = np.zeros((len(estimates), 4))
        for i in range(len(estimates)):
            self.estimates[i, :] = estimates[i]


class ParticleFilterWithKalman(ParticleFilter):
    """
    Combination of particle and Kalman filters.
    Particle filter estimates position and Kalman filter estimates global velocity and acceleration
    of all particles.
    """

    def __init__(self, num_particles, process_std, measurement_std, Q = None, R = None,
                 workspace_bounds = None, dt = 0.2, std_a = 0.1):
        super().__init__(num_particles, process_std, measurement_std, workspace_bounds, dt)
        self.estimate = np.zeros(3)  # Actual state estimate - this is our tracked value
        self.estimate_var = 20  # Large value in the beginning
        self.vel = np.zeros(3)  # process velocity, i.e. position or angular velocity determined externally from observed data
        self.vel_std = 0.01  # Unknown velocity in the beginning
        self.particles = self.create_uniform_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles  # Uniform weights initialization
        self.estimates = []
        self.estimate_vars = []
        self.initialize_kalman(Q, R)
        self.dt = dt
        self.factor_a = std_a


    def initialize_kalman(self, Q, R):
        if Q is None:
            Q = 0.08 / self.dt
        if R is None:
            R = 0.02 / self.dt
        x = np.zeros(6)
        P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.kalman = KalmanFilter(dim_x = x.shape[0], dim_z = 3)
        self.kalman.x = x
        self.kalman.F = np.array([[1, self.dt, 0, 0, 0 ,0 ],
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, self.dt, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, self.dt],
                                  [0, 0, 0, 0, 0, 1]])

        self.kalman.H = np.array([[1, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0]])  # Measurement function
        self.kalman.R *= R  # measurement uncertainty

        if np.isscalar(P):
            self.kalman.P *= P  # covariance matrix
        else:
            self.kalman.P[:] = P  # [:] makes deep copy
        if np.isscalar(Q):
            Q1dim = Q_discrete_white_noise(dim=2, dt=self.dt, var=Q) #Discrete white noise for one dimension
            self.kalman.Q = np.zeros((6, 6))
            self.kalman.Q[:2, :2] = Q1dim
            self.kalman.Q[2:4, 2:4] = Q1dim
            self.kalman.Q[4:6, 4:6] = Q1dim
        else:
            self.kalman.Q[:] = Q


    def create_gaussian_particles(self, std):
        """Create normally distributed particles around estimate with given std"""
        N = self.num_particles
        particles = np.empty((N, 3))
        for i in range(3):
            particles[:, i] = self.estimate[i] + (np.random.randn(N) * std)
        return particles


    def predict(self):
        n = self.num_particles
        a = np.linalg.norm([self.kalman.x[1], self.kalman.x[3], self.kalman.x[5]])
        vel = self.vel #+ np.random.randn(3)
        time_factor = self.dt/0.2
        self.particles[:, :3] += vel * self.dt  # #CONSTANTS BELOW:
        self.particles[:, 0] += np.random.randn(n) * a * time_factor * self.factor_a + time_factor * self.process_std * np.random.randn(n)
        self.particles[:, 1] += np.random.randn(n) * a * time_factor * self.factor_a + time_factor* self.process_std * np.random.randn(n)
        self.particles[:, 2] += np.random.randn(n) * a * time_factor *self.factor_a + time_factor *self.process_std * np.random.randn(n)

        self.kalman.predict()


    def update(self, z):
        distance_pos = np.linalg.norm(self.particles[:, 0:3] - z[0:3], axis=1)
        self.weights *= scipy.stats.norm(0, self.measurement_std).pdf(distance_pos)  # Multiply weights based on the
        # particles' Gaussian probability
        self.weights += 1.e-300  # Avoid round-off to zero
        self.weights = self.weights / np.sum(self.weights)  # Normalization


    def state_estimate(self):
        last_estimate = self.estimate
        self.estimate = np.average(self.particles, weights=self.weights, axis=0)  # Weighted average of particles
        var = np.average(((self.particles - self.estimate) ** 2), weights=self.weights, axis=0)
        self.estimate_var = var
        self.estimates.append(self.estimate)
        self.estimate_vars.append(self.estimate_var)
        self.kalman.update((self.estimate - last_estimate) / self.dt)
        self.vel = self.get_velocity()





    def get_velocity(self):
        return np.array([self.kalman.x[0], self.kalman.x[2], self.kalman.x[4]])


    def index_resample_function(self):
        return filterpy.monte_carlo.stratified_resample(self.weights)




class ParticleFilterWithKalmanRot(ParticleFilter):

    def __init__(self, num_particles, process_std, measurement_std, Q = None, R = None,
                 workspace_bounds = None, dt = 0.2, factor_a = 0.1, rotflip_const = 0.1):
        super().__init__(num_particles, process_std, measurement_std, workspace_bounds, dt)
        self.estimate = np.zeros(4)  # Actual state estimate - this is our tracked value
        self.estimate_var = 20  # Large value in the beginning
        self.vel = np.zeros(4)  # process velocity, i.e. position or angular velocity determined externally from observed data
        self.vel_std = 0.01  # Unknown velocity in the beginning
        self.particles = self.create_uniform_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles  # Uniform weights initialization
        self.estimates = []
        self.estimate_vars = []
        self.initialize_kalman(Q, R)
        self.dt =dt
        self.rotation_flip_const = rotflip_const
        self.factor_a = factor_a


    def initialize_kalman(self, Q, R):
        if Q is None:
            Q = 0.08 / self.dt
        if R is None:
            R = 0.02 / self.dt
        x = np.zeros(8)
        P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.kalman = KalmanFilter(dim_x=x.shape[0], dim_z=4)
        self.kalman.x = x
        self.kalman.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, self.dt, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, self.dt, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, self.dt],
                                  [0, 0, 0, 0, 0, 0, 0, 1]])

        self.kalman.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0]])  # Measurement function
        self.kalman.R *= R  # measurement uncertainty

        if np.isscalar(P):
            self.kalman.P *= P  # covariance matrix
        else:
            self.kalman.P[:] = P  # [:] makes deep copy
        if np.isscalar(Q):
            Q1dim = Q_discrete_white_noise(dim=2, dt=self.dt, var=Q)  # Discrete white noise for one dimension
            self.kalman.Q = np.zeros((8, 8))
            self.kalman.Q[:2, :2] = Q1dim
            self.kalman.Q[2:4, 2:4] = Q1dim
            self.kalman.Q[4:6, 4:6] = Q1dim
            self.kalman.Q[6:8, 6:8] = Q1dim
        else:
            self.kalman.Q[:] = Q


    def create_gaussian_particles(self, std):
        """Create normally distributed particles around estimate with given std"""
        N = self.num_particles
        particles = np.empty((N, 4))
        for i in range(4):
            particles[:, i] = self.estimate[i] + (np.random.randn(N) * std)
        return particles


    def predict(self):
        n = self.num_particles
        a = np.linalg.norm([self.kalman.x[1], self.kalman.x[3], self.kalman.x[5], self.kalman.x[7]])
        vel = self.vel #+ np.random.randn(3)

        self.particles[:, :4] += vel * self.dt  # Predict + process noise
        self.particles[:, 0] += np.random.randn(n) * a*self.factor_a + self.process_std *np.random.randn(n)
        self.particles[:, 1] += np.random.randn(n) * a*self.factor_a + self.process_std *np.random.randn(n)
        self.particles[:, 2] += np.random.randn(n) * a*self.factor_a + self.process_std *np.random.randn(n)
        self.particles[:, 3] += np.random.randn(n) * a*self.factor_a + self.process_std * np.random.randn(n)
        self.kalman.predict()


    def update(self, z):
        norm = np.linalg.norm(z + self.estimate)
        if norm < self.rotation_flip_const:
            self.reapply_measurement(z)

        distance_pos = np.linalg.norm(self.particles[:, 0:4] - z[0:4], axis=1)
        self.weights *= scipy.stats.norm(0, self.measurement_std).pdf(distance_pos)  # Multiply weights based on the
        # particles' Gaussian probability
        self.weights += 1.e-300  # Avoid round-off to zero
        self.weights = self.weights / np.sum(self.weights)  # Normalization


    def state_estimate(self):
        last_estimate = self.estimate
        self.estimate = np.average(self.particles, weights=self.weights, axis=0)  # Weighted average of particles
        var = np.average(((self.particles - self.estimate) ** 2), weights=self.weights, axis=0)
        self.estimate_var = var
        self.estimates.append(self.estimate)
        self.estimate_vars.append(self.estimate_var)
        self.kalman.update((self.estimate - last_estimate) / self.dt)
        self.vel = self.get_velocity()


    def get_velocity(self):
        return np.array([self.kalman.x[0], self.kalman.x[2], self.kalman.x[4], self.kalman.x[6]])


    def index_resample_function(self):
        return filterpy.monte_carlo.stratified_resample(self.weights)


    def reapply_measurement(self, z):
        """If the quaternion flips (or changes abruptly), set estimate according to the measurement"""
        self.estimate = z
        uncertainty_factor = 1.2
        self.particles = self.create_gaussian_particles(uncertainty_factor* self.measurement_std)

    def filter_step(self, z):
        """One full step of the filter."""
        self.predict()
        self.update(z)
        self.state_estimate()
        if self.neff() < self.num_particles/2:
            self.resample()

    def convert_estimates(self):
        """Converts list of estimates into numpy estimate array"""
        estimates = self.estimates
        self.estimates = np.zeros((len(estimates), 4))
        for i in range(len(estimates)):
            self.estimates[i, :] = estimates[i]
