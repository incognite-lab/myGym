"""Module for particle filter to estimate 6D object pose in mygym environment"""

import numpy as np
import filterpy
from filterpy.monte_carlo import residual_resample, stratified_resample, systematic_resample
import scipy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class myKalmanFilter():
    """Returns a Kalman filter which implements constant velocity model"""
    def __init__(self, x, P, R, Q= 0., dt=0.2):

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
        self.process_std = Q #Values for filename
        self.measurement_std = R #Values for filename
        self.num_particles = 0 #Values for filename


    def get_est(self):
        state = np.array([self.filter.x[0], self.filter.x[2], self.filter.x[4]])
        #self.filter.x[0] = state[0]
        #self.filter.x[2] = state[1]
        #self.filter.x[4] = state[2]
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



class ParticleFilter(object):
    """Base particle filter parent class. Contains method and parameters used in every type of particle filter"""
    def __init__(self, num_particles, process_std, measurement_std, workspace_bounds = None, dt = 0.1):
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


    def apply_first_measurement(self, measurement):
        """Determine initial values based on first measurement"""
        self.estimate = measurement
        initial_uncertainty_factor = 5  # multiply std by this factor to account for initial uncertainty to better address
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
        """Create gaussian particles in workspace bounds of a given dimension"""
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
        self.state_estimate()
        self.resample()


class ParticleFilter6D(ParticleFilter):
    """
    Particle filter which includes particle velocities for better prediction.
    Each particle has its velocity, which changes only based on noise. Those particles, that move correctly in
    the direction of the velocity get to be resampled. This simulates acceleration.
    """
    def __init__(self, num_particles, process_std, vel_std, measurement_std, workspace_bounds = None, dt = 0.1):
        super().__init__(num_particles, process_std, measurement_std, workspace_bounds, dt)
        self.vel_std = vel_std
        self.vel_bounds = [(-1, 1), (-1, 1), (-1, 1)] #Max velocity of a particle for uniform particles
        self.particles = self.create_uniform_particles()
        # Below there are attributes used for computing the moving average of measured acceleration
        self.moving_avg_length = 10
        self.last_measured_distances = np.zeros((self.moving_avg_length+1, 3))
        self.last_measured_position = None
        self.vel_estimate = np.zeros(3)
        self.prev_vel_estimate = np.zeros(3)
        self.vel_std_const = 0.15
        self.a = 0#np.zeros(3)
        self.k = 0.8#coefficient of past acceleration memory


    def apply_first_measurement(self, measurement):
        """Determine initial values based on first measurement"""
        self.estimate = measurement
        initial_uncertainty_factor = 2  #Multiply std by this factor to account for initial uncertainty to better address
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


    def acceleration_ma(self):
        """Computes the moving average of object acceleration to determine a good value of vel_std
        (the higher the acceleration, the bigger the velocity change and therefore its uncertainty)
        """
        v = np.array(self.last_measured_distances)/self.dt
        if v.size > 1: #To determine acceleration we need at least two elements
            dv = np.diff(v, axis = 0)
            print("dv:", dv)
            a = np.linalg.norm(np.sum(dv, axis = 0) / (self.dt * self.moving_avg_length))
            return a
        else:
            return None


    def update(self, z):
        super().update(z)
        #2nd approach to filter acceleration
        self.vel_estimate = (z - self.last_measured_position)/self.dt
        self.a = self.k*self.a + (1 - self.k) *(self.vel_estimate - self.prev_vel_estimate)/self.dt
        self.vel_std = self.a * self.vel_std_const
        self.last_measured_position = z
        self.prev_vel_estimate = self.vel_estimate
        print("Acceleration:", np.linalg.norm(self.a), "vel_std:", np.linalg.norm(self.vel_std))
        #Bellow is the process of calculating moving average of acceleration to determine self.vel_std
        """
        if self.last_measured_position is not None:
            self.last_measured_distances = np.roll(self.last_measured_distances, -1, axis = 0)
            self.last_measured_distances[self.moving_avg_length, :] = (z - self.last_measured_position)#
            self.last_measured_position = z #Updating last measured position value
        else:
            self.last_measured_position = z
        ma = self.acceleration_ma()
        if ma is not None:
            self.vel_std = ma*self.vel_std_const
        print("vel_std=", self.vel_std)
        """





class ParticleFilterGH(ParticleFilter):
    """Custom Particle filter class for estimating position or orientation of 3D objects from noisy measurements"""
    def __init__(self, num_particles, process_std, measurement_std, workspace_bounds = None, dt = 0.1, g=0.5, h=0.5):
        super().__init__(num_particles, process_std, measurement_std, workspace_bounds, dt)
        self.estimate = np.zeros(3) #Actual state estimate - this is our tracked value
        self.estimate_var = 20 #Large value in the beginning
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
        #print(distance_pos)
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






