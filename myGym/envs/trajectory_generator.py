"""
This file serves as a dataset generator used to evaluate particle filter performance.
"""
import numpy as np
from pyquaternion import Quaternion
import random
from myGym.utils.filter_helpers import  fit_polynomial, create_circle, trajectory_length
import json


class MultipleTrajectoryGenerator:
    """Class that generates random multiple trajectories from lines, splines and circles."""
    def __init__(self, num_trajectories, trajectory_parameters, dt_std):
        self.trajectory_parameters = trajectory_parameters
        self.num_trajectories = num_trajectories
        self.dt_std = dt_std


    def save_1_trajectory(self, X, y, X_rot, y_rot):
        filename_exists = True
        i = 0
        while filename_exists:
            filename = "./dataset/MOT/ground_truth"+ str(i)+".npy"
            try:
                with open(filename, 'x') as file:
                    filename_exists = False
            except FileExistsError:
                i+=1
        np.save("./dataset/MOT/ground_truth" + str(i) + ".npy", y)
        np.save("./dataset/MOT/noisy_data" + str(i) + ".npy", X)
        np.save("./dataset/MOT/rotations" + str(i) + ".npy", y_rot)
        np.save("./dataset/MOT/noisy_rotations" + str(i) + ".npy", X_rot)


    def find_maximal_length(self, gt):
        max = 0
        for traj in gt:
            if traj.shape[0] > max:
                max = traj.shape[0]
        return max

    def generate_trajectories(self, generators):
        gt, r, nd, nr = [], [], [], []
        for i in range(self.num_trajectories):
            generator = random.choice(generators)
            ground_truth, rotations, noisy_data, noisy_rotations = generator.generate_1_trajectory()
            rotations = generator.convert_1_rot_trajectory(rotations)
            noisy_rotations = generator.convert_1_rot_trajectory(noisy_rotations)
            gt.append(ground_truth)
            r.append(rotations)
            nd.append(noisy_data)
            nr.append(noisy_rotations)
        return gt, r, nd, nr

    def initialize_X_and_y(self, gt):
        max = self.find_maximal_length(gt)
        X = np.array([99] * 3 * self.num_trajectories)
        y = np.array([99] * 3 * self.num_trajectories)
        X_rot = np.array([99] * 4 * self.num_trajectories)
        y_rot = np.array([99] * 4 * self.num_trajectories)

        X = np.vstack((X, np.full((max, 3*self.num_trajectories), 88.)))
        y = np.vstack((y, np.full((max, 3*self.num_trajectories), 88.)))
        X_rot = np.vstack((X_rot, np.full((max, 4*self.num_trajectories), 88.)))
        y_rot = np.vstack((y_rot, np.full((max, 4*self.num_trajectories), 88.)))
        return X, y, X_rot, y_rot

    def generate_1_scenario(self):
        """
        Function that generates multiple trajectories as one dataset
        """
        generators = []
        for type in self.trajectory_parameters:
            parameters =  self.trajectory_parameters[type]
            generator = self.initialize_generator(type, parameters)
            generators.append(generator)
        gt, r, nd, nr = self.generate_trajectories(generators)
        X, y, X_rot, y_rot = self.initialize_X_and_y(gt)
        for i in range(self.num_trajectories):
            y[:gt[i].shape[0], 3*i:3*(i+1)] = gt[i]
            X[:nd[i].shape[0], 3*i:3*(i+1)] = nd[i]
            X_rot[:r[i].shape[0], 4*i:4*(i+1)] = r[i]
            y_rot[:nr[i].shape[0]:, 4*i:4*(i+1)] = nr[i]
        return X, y, X_rot, y_rot

    def initialize_generator(self, type, params):
        if type == "lines":
            generator = LineGenerator(params["std"], params["exp_points"], params["distance_limit"],
                                      params["workspace_bounds"], accelerate = False)
        elif type == "circle":
            generator = CircleGenerator(params["std"],  r_min =params["r_min"], r_max = params["r_max"])
        elif type == "spline":
            generator = SplineGenerator(params["std"], params["exp_points"], params["distance_limit"])
        elif type == "lines_acc":
            generator = LineGenerator(params["std"], params["exp_points"], params["distance_limit"],
                                      params["workspace_bounds"], accelerate = True)
        else:
            print("entered wrong trajectory type:", type)
            sys.exit()
        return generator



class TrajectoryGenerator:

    def __init__(self, std, velocity = 0.2, dt = 0.2, workspace_bounds = None):
        self.std = std
        self.v = velocity
        self.dt = dt
        if workspace_bounds is None:
            self.workspace_bounds = [(-2, 2), (-2, 2), (0, 4)]
        else:
            self.workspace_bounds = workspace_bounds

        self.filename_basis = "./dataset/unknown"
        self.filename_basis_r = "./dataset/unknown_r"
        self.filename_basis_noise = "./dataset/unknown_noise"
        self.filename_basis_noise_r = "./dataset/unknown_noise_r"


    """
    Helper functions
    ||||||
    vvvvvv
    """

    def check_min_distance(self, new_point, keypoints):
        for point in keypoints:
            if np.linalg.norm(np.array(new_point) - np.array(point)) < self.distance_limit:
                return False
        return True


    def check_rotation_distance(self, q1, q2, trajectory_section, omega):
        """Checks whether or not two rotations too spatially close to each other are not too different."""
        s = trajectory_length(trajectory_section)
        v = self.v
        max_angle_diff = omega * s / v
        if Quaternion.absolute_distance(q1, q2) * 2 > max_angle_diff:
            return False
        return True


    """
    End helper functions
    """

    """
    Generator functions
    """

    def generate_n_trajectories(self, n):
        trajs = []
        rots = []
        noisy_trajs = []
        noisy_rots = []
        for i in range(n):
            traj, rot, noisy_traj, noisy_rot = self.generate_1_trajectory()
            trajs.append(traj)
            rots.append(rot)
            noisy_trajs.append(noisy_traj)
            noisy_rots.append(noisy_rot)
        return trajs, rots, noisy_trajs, noisy_rots


    def generate_1_trajectory(self):
        traj = np.random.uniform(-2, 2, (100, 3))
        noisy_traj = self.add_noise(traj, self.std)
        rotations = [Quaternion(0, 0, 0, 0)]*100
        noisy_rotations = self.add_rotation_noise(rotations)
        return traj, noisy_traj, rotations, noisy_rotations

    """
    End generator functions
    """

    """
    Saver functions
    """

    def save_trajectories(self, trajectories, rot_trajectories, noisy_trajectories, noisy_rotations):
        for i in range(len(trajectories)):
            self.save_1_trajectory(trajectories[i], rot_trajectories[i], noisy_trajectories[i], noisy_rotations[i], i)


    def save_1_trajectory(self, trajectory, rot_trajectory, noisy_trajectory, noisy_rot, i=""):
        """
        Saves 1 position and rotation trajectory.
        """
        filename_exists = True
        i = 0
        while filename_exists:
            filename = self.filename_basis + str(i) + ".npy"
            try:
                with open(filename, 'x') as file:
                    filename_exists = False
            except FileExistsError:
                i += 1
            filename = self.filename_basis + str(i)
            filename_r = self.filename_basis_r + str(i)
            filename_noise = self.filename_basis_noise + str(i)
            filename_noisy_r = self.filename_basis_noise_r + str(i)
            rot_traj = self.convert_1_rot_trajectory(rot_trajectory)
            noisy_rot_traj = self.convert_1_rot_trajectory(noisy_rot)
            np.save(filename, trajectory)
            np.save(filename_r, rot_traj)
            np.save(filename_noise, noisy_trajectory)
            np.save(filename_noisy_r, noisy_rot_traj)


    def convert_1_rot_trajectory(self, rotation_trajectory):
        """Transforms a list of quaternions (rotational trajectory) into a numpy array of
        quaternion elements"""
        n = len(rotation_trajectory)
        rot_array = np.zeros((n, 4))
        for i in range(n):
            rot = rotation_trajectory[i].elements
            rot_array[i, :] = rot
        return rot_array

    def generate_and_save_n_trajectories(self, n):
        trajectories, rotation_trajectories, noisy_trajectories, noisy_rotations = self.generate_n_trajectories(n)
        self.save_trajectories(trajectories, rotation_trajectories, noisy_trajectories, noisy_rotations)

    """
    End saver functions
    """


    def add_noise(self, traj, sigma):
        """Function that adds Gaussian noise of given sigma std to a trajectory"""
        noise_x, noise_y, noise_z = sigma * np.random.randn(len(traj)), sigma * np.random.randn(
            len(traj)), sigma * np.random.randn(len(traj))
        ret = np.copy(traj)
        ret[:, 0] += noise_x
        ret[:, 1] += noise_y
        ret[:, 2] += noise_z
        return ret


    def add_rotation_noise(self, rotations, sigma_q = np.sqrt(3) * 8):
        """
        Add noise to rotational trajectory (series of quaternions) by generating a noisy rotation in random direction
        and random angle generated from gaussian distribution of std size sigma_q.
        sigma_q: Standard deviation of angle difference (representing 4 degrees of stdev in every dimension)
        """
        noisy_rotations = []
        for i in range(len(rotations)):
            q_gt = rotations[i]
            q_dir = Quaternion.random()
            dq = q_gt.inverse * q_dir
            dir_axis = dq.axis #Axis of rotation between ground truth and random rotation
            noise_angle = np.abs(np.random.randn() * sigma_q) #Sample angle size for noisy rotation
            q_noise = Quaternion(axis=dir_axis, angle=np.deg2rad(noise_angle))
            noisy_quat = q_gt*q_noise
            noisy_rotations.append(noisy_quat)
        return noisy_rotations



class LineGenerator(TrajectoryGenerator):

    def __init__(self, std, exp_points, distance_limit, workspace_bounds, accelerate = False, dt = 0.2, v = 0.2):
        """
        Parameters:
            std : float (standard deviation of measured noise)
            exp_points : int (The expected value of trajectory points generated by a Poisson distributions)
            distance_limit : float (The minimal distance between two points)
            worskpace_bounds : list of tuples (Every point on the trajectory must be located inside of these x, y, z bounds)
            accelerate : vool (determine whether object moves with constant velocity or acceleration)
        """
        super().__init__(std, velocity = v, dt = dt, workspace_bounds = workspace_bounds)
        if accelerate == False:
            self.filename_basis = "./dataset/newly_generated/lines/positions/line"
            self.filename_basis_r = "./dataset/newly_generated/lines/rotations/rot"
            self.filename_basis_noise = "./dataset/newly_generated/lines/positions/line_noise"
            self.filename_basis_noise_r = "./dataset/newly_generated/lines/rotations/rot_noise"
        else:
            self.filename_basis = "./dataset/newly_generated/lines_acc/positions/line"
            self.filename_basis_r = "./dataset/newly_generated/lines_acc/rotations/rot"
            self.filename_basis_noise = "./dataset/newly_generated/lines_acc/positions/line_noise"
            self.filename_basis_noise_r = "./dataset/newly_generated/lines_acc/rotations/rot_noise"
        self.exp_points = exp_points
        self.distance_limit = distance_limit
        self.accelerate = accelerate



    def generate_1_trajectory(self):
        num_points = np.random.poisson(lam = self.exp_points)
        if num_points < 2: #Minimum of 2 points for a trajectory
            num_points = 2
        key_points = []
        for i in range(num_points):
            x_range, y_range, z_range = self.workspace_bounds
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = np.random.uniform(z_range[0], z_range[1])
            while not self.check_min_distance([x, y, z], key_points):
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                z = np.random.uniform(z_range[0], z_range[1])
            key_points.append([x, y, z])
        if self.accelerate:
            return self.create_trajectory_const_acc(key_points)
        else:
            return self.create_trajectory_const_vel(key_points)



    def create_trajectory_const_vel(self, keypoints):
        """
        Creates trajectory between keypoints assuming the object moves with a constant velocity.
        """
        v = 0.2
        lines = []
        rot_lines = []
        starting_rot = None
        for i in range(len(keypoints) - 1):
            start = np.array(keypoints[i])
            end = np.array(keypoints[i + 1])
            length = np.linalg.norm(end - start)
            n = int(length / (v*self.dt))
            line = np.linspace(start, end, n)
            rot_line = self.generate_rotations(line, np.deg2rad(15), starting_rot) #CONSTANTS
            rot_lines.extend(rot_line)
            starting_rot = rot_line[-1]
            lines.append(line)
        lines = tuple(lines)
        trajectory = np.vstack(lines)
        noisy_trajectory = self.add_noise(trajectory, self.std)
        noisy_rotations = self.add_rotation_noise(rot_lines)
        return trajectory, rot_lines, noisy_trajectory, noisy_rotations


    def create_trajectory_const_acc(self, keypoints):
        """
        Creates trajectory between keypoints assuming the object moves with a constant acceleration.
        """

        lines = []
        rot_lines = []
        starting_rot = None
        for i in range(len(keypoints) - 1):
            line = self.const_acc_single_line(np.array(keypoints[i]), np.array(keypoints[i + 1]))
            lines.append(line)
            #print("Line:", line)
            rot_line = self.generate_rotations(line, np.deg2rad(15), starting_rot)  # CONSTANTS
            rot_lines.extend(rot_line)
            starting_rot = rot_line[-1]
        lines = tuple(lines)
        trajectory = np.vstack(lines)
        noisy_trajectory = self.add_noise(trajectory, self.std)
        noisy_rotations = self.add_rotation_noise(rot_lines)
        return trajectory, rot_lines, noisy_trajectory, noisy_rotations


    def const_acc_single_line(self, start, end):
        """
        Creates single line simulating object moving with constant acceleration.
        """
        avg_velocity = np.random.uniform(0.1, 0.3)  # CONSTANTS
        v_max = 2*avg_velocity

        length = np.linalg.norm(end - start)
        t = length / avg_velocity

        n = int(t/self.dt)
        #t_vec = np.linspace(0, t, n)
        if n%2 == 0:
            v = np.concatenate((np.linspace(0, v_max, int(n/2)), np.linspace(v_max, 0, int(n/2))))
        else:
            v = np.concatenate((np.linspace(0, v_max, int(n/2) + 1), np.linspace(v_max, 0, int(n/2))))
        direction_vector = (end - start)/length #Direction vector of length one
        trajectory = np.zeros((n, 3))
        trajectory[0, :] = start
        for i in range(n-1):
            trajectory[i+1, :] = trajectory[i, :] + v[i] * direction_vector * self.dt
        return trajectory


    def generate_rotations(self, trajectory, omega, starting_rot = None):
        """Generates interpolated rotational trajectory for given trajectory."""
        if starting_rot is None:
            q1 = Quaternion.random()
        else:
            q1 = starting_rot
        q2 = Quaternion.random()
        while not self.check_rotation_distance(q1,q2, trajectory, omega):
            q2 = Quaternion.random()
        rotations = []
        for rot in Quaternion.intermediates(q1, q2, trajectory.shape[0]):
            rotations.append(rot)
        return rotations



class CircleGenerator(TrajectoryGenerator):
    """
    Class for generating circles inside workspace bounds.
    Parameters:
        std : float (standard deviation of measured noise)
        workspace_bounds: list of tuples (Every point on the trajectory must be located inside of these x, y, z bounds)
        dt : float (Time that passes between each measurement)
        r_min : float (minimal radius for circle generation)
        r_max : float (maximal radius for circle generation)
    """

    def __init__(self, std, workspace_bounds = None, dt = 0.2, r_min = 0.2, r_max = 2, velocity = 0.2):
        super().__init__(std, velocity=velocity, dt = dt, workspace_bounds=workspace_bounds)
        self.r_min = r_min
        self.r_max = r_max
        self.filename_basis = "./dataset/newly_generated/circles/positions/circle"
        self.filename_basis_r = "./dataset/newly_generated/circles/rotations/rot"
        self.filename_basis_noise = "./dataset/newly_generated/circles/positions/circle_noise"
        self.filename_basis_noise_r = "./dataset/newly_generated/circles/rotations/rot_noise"


    def generate_1_trajectory(self):
        """Circle generation function"""
        r = np.random.uniform(self.r_min, self.r_max)
        normal = np.random.rand(3)
        center_x = np.random.uniform(self.workspace_bounds[0][0] + r, self.workspace_bounds[0][1] -r)
        center_y = np.random.uniform(self.workspace_bounds[1][0] + r, self.workspace_bounds[1][1] -r)
        center_z = np.random.uniform(self.workspace_bounds[2][0] + r, self.workspace_bounds[2][1] -r)
        center = np.array([center_x, center_y, center_z])
        return self.create_trajectory(center, r, normal)


    def create_trajectory(self, center, radius, normal):
        """
        Create circular trajectory of n points.
        """
        length = 2*np.pi*radius
        n = int(length / (self.v*self.dt))

        trajectory = create_circle(radius, center, normal, n)
        rotation_trajectory = self.generate_rotations(trajectory, np.deg2rad(25))
        noisy_trajectory = self.add_noise(trajectory, self.std)
        noisy_rotations = self.add_rotation_noise(rotation_trajectory)
        return trajectory, rotation_trajectory, noisy_trajectory, noisy_rotations


    def generate_rotations(self, trajectory, omega, starting_rot = None):
        """Generates interpolated rotational trajectory for given trajectory."""
        if starting_rot is None:
            q1 = Quaternion.random()
        else:
            q1 = starting_rot
        q2 = Quaternion.random()
        while not self.check_rotation_distance(q1,q2,trajectory, omega):
            q2 = Quaternion.random()
        rotations = []
        for rot in Quaternion.intermediates(q1, q2, trajectory.shape[0]):
            rotations.append(rot)
        return rotations



class SplineGenerator(TrajectoryGenerator):

    def __init__(self, std, exp_points, distance_limit, workspace_bounds = None, dt = 0.2, velocity = 0.2):
        super().__init__(std, velocity=velocity, dt=dt, workspace_bounds=workspace_bounds)
        self.exp_points = exp_points
        self.distance_limit = distance_limit
        self.filename_basis = "./dataset/newly_generated/splines/positions/spline"
        self.filename_basis_r = "./dataset/newly_generated/splines/rotations/rot"
        self.filename_basis_noise = "./dataset/newly_generated/splines/positions/spline_noise"
        self.filename_basis_noise_r = "./dataset/splines/newly_generated/rotations/rot_noise"


    def generate_1_trajectory(self):
        num_points = np.random.poisson(lam = self.exp_points)
        if num_points < 3: #Minimum of 3 points for a spline
            num_points = 3
        key_points = []
        for i in range(num_points):
            x_range, y_range, z_range = self.workspace_bounds
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = np.random.uniform(z_range[0], z_range[1])
            while not self.check_min_distance([x, y, z], key_points):
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                z = np.random.uniform(z_range[0], z_range[1])
            key_points.append([x, y, z])
        return self.create_trajectory(key_points)


    def create_trajectory(self, keypoints):
        """
        Given list of keypoints, create a spline along with rotational trajectory and added noise to both.
        """
        keypoints = np.array(keypoints)
        trajectory = np.zeros((40, 3))
        trajectory[:, :3] = fit_polynomial(keypoints,40)
        new_n = int(trajectory_length(trajectory)/(self.v * self.dt))
        trajectory = fit_polynomial(keypoints, new_n)
        starting_rot = None
        rotations = []
        for i in range(len(keypoints) - 1):
            rotations.extend(self.generate_rotations(trajectory, keypoints[i], keypoints[i + 1], np.deg2rad(25), starting_rot))
            starting_rot = rotations[-1]
        rotations.append(rotations[-1])
        noisy_trajectory = self.add_noise(trajectory, self.std)
        noisy_rotations = self.add_rotation_noise(rotations)
        return trajectory, rotations, noisy_trajectory, noisy_rotations


    def generate_rotations(self, trajectory, start, end, omega, starting_rot = None):
        """Generates interpolated rotational trajectory for given trajectory."""
        if starting_rot is None:
            q1 = Quaternion.random()
        else:
            q1 = starting_rot
        q2 = Quaternion.random()
        start_row = np.argmin(np.linalg.norm(trajectory - start, axis=1))
        end_row = np.argmin(np.linalg.norm(trajectory - end, axis=1))
        trajectory_section = trajectory[start_row:end_row, :]
        while not self.check_rotation_distance(q1,q2,trajectory_section, omega):
            q2 = Quaternion.random()
        rotations = []
        for rot in Quaternion.intermediates(q1, q2, trajectory_section.shape[0]):
            rotations.append(rot)
        return rotations




if __name__ == '__main__':
    # for i in range(1,7,1):
    #     filename = "./dataset/circles/rotations/rot" + str(i) + ".npy"
    #     array = np.load(filename)
    #     array = np.delete(array, -1, 0)
    #     np.save(filename, array)
    #     arr_new = np.load(filename)
    #     print("updated, saved and newly loaded array:", arr_new)
    #     print("------------------------------------------")
    with open("./configs/MOT_trajectory_parameters.json") as json_file:
        traj_params = json.load(json_file)
    print(traj_params)
    scenario_gen = MultipleTrajectoryGenerator(3,traj_params)
    print(scenario_gen.generate_1_scenario())
    # for i in range(1,7,1):
    #     filename = "./dataset/lines_acc/rotations/rot_noise" + str(i) + ".npy"
    #     array = np.load(filename)
    #     array = np.vstack((array, [99, 99, 99, 99]))
    #     #print("loaded and appended array:", array)
    #     np.save(filename, array)
    #     arr_new = np.load(filename)
    #     print("updated, saved and newly loaded array:", arr_new)
    #     print("------------------------------------------")





