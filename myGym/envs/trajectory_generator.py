"""
This file serves as a dataset generator used to evaluate particle filter performance.
"""
import numpy as np
from pyquaternion import Quaternion
from myGym.utils.filter_helpers import  fit_polynomial, create_circle, trajectory_length


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
        noisy_rotations = self.add_rotation_noise(rotations, self.std)
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


    def add_rotation_noise(self, rotations, sigma_q):
        """
        Add noise to rotational trajectory (series of quaternions)
        """
        noisy_rotations = []
        for i in range(len(rotations)):
            rot = rotations[i]
            w, x, y, z = np.random.randn() * sigma_q, np.random.randn() * sigma_q, np.random.randn() * sigma_q, np.random.randn() * sigma_q
            noise_q = Quaternion(w, x, y, z)
            quat_noisy = rot + noise_q
            noisy_rotations.append(quat_noisy)
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
            self.filename_basis = "./dataset/lines/positions/line"
            self.filename_basis_r = "./dataset/lines/rotations/rot"
            self.filename_basis_noise = "./dataset/lines/positions/line_noise"
            self.filename_basis_noise_r = "./dataset/lines/rotations/rot_noise"
        else:
            self.filename_basis = "./dataset/lines_acc/positions/line"
            self.filename_basis_r = "./dataset/lines_acc/rotations/rot"
            self.filename_basis_noise = "./dataset/lines_acc/positions/line_noise"
            self.filename_basis_noise_r = "./dataset/lines_acc/rotations/rot_noise"
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
        noisy_rotations = self.add_rotation_noise(rot_lines, self.std)
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
        noisy_rotations = self.add_rotation_noise(rot_lines, self.std)
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
        print("starting rot:", starting_rot)
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
        self.filename_basis = "./dataset/circles/positions/circle"
        self.filename_basis_r = "./dataset/circles/rotations/rot"
        self.filename_basis_noise = "./dataset/circles/positions/circle_noise"
        self.filename_basis_noise_r = "./dataset/circles/rotations/rot_noise"


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
        noisy_rotations = self.add_rotation_noise(rotation_trajectory, self.std)
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
        self.filename_basis = "./dataset/splines/positions/spline"
        self.filename_basis_r = "./dataset/splines/rotations/rot"
        self.filename_basis_noise = "./dataset/splines/positions/spline_noise"
        self.filename_basis_noise_r = "./dataset/splines/rotations/rot_noise"


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
        trajectory[:, :3] = fit_polynomial(keypoints, 40)
        new_n = int(trajectory_length(trajectory)/(self.v * self.dt))
        trajectory = fit_polynomial(keypoints, new_n)
        starting_rot = None
        rotations = []
        for i in range(len(keypoints) - 1):
            rotations.extend(self.generate_rotations(trajectory, keypoints[i], keypoints[i + 1], np.deg2rad(25), starting_rot))
            starting_rot = rotations[-1]
        rotations.append(rotations[-1])
        noisy_trajectory = self.add_noise(trajectory, self.std)
        noisy_rotations = self.add_rotation_noise(rotations, self.std)
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
    generator = LineGenerator(0.02, 3, 0.2, [(-4,4),(0.3, 5), (0, 5)], accelerate=True)
    print(generator.generate_1_trajectory())



