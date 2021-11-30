import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import results_plotter
import os
import math
from math import sqrt
from myGym.vector import Vector


class Reward:
    """
    Reward base class for reward signal calculation and visualization

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task=None):
        self.env = env
        self.task = task
        self.rewards_history = []

    def compute(self, observation=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def visualize_reward_over_steps(self):
        """
        Plot and save a graph of reward values assigned to individual steps during an episode. Call this method after the end of the episode.
        """
        save_dir = os.path.join(self.env.logdir, "rewards")
        os.makedirs(save_dir, exist_ok=True)
        if self.env.episode_steps > 0:
            results_plotter.EPISODES_WINDOW=50
            results_plotter.plot_curves([(np.arange(self.env.episode_steps),np.asarray(self.rewards_history[-self.env.episode_steps:]))],'step','Step rewards')
            plt.ylabel("reward")
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(save_dir + "/reward_over_steps_episode{}.png".format(self.env.episode_number))
            plt.close()

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = Vector([0,0,0], gripper_position, self.env)
        return direction_vector.add_vector(gripper)


    def visualize_reward_over_episodes(self):
        """
        Plot and save a graph of cumulative reward values assigned to individual episodes. Call this method to plot data from the current and all previous episodes.
        """
        save_dir = os.path.join(self.env.logdir, "rewards")
        os.makedirs(save_dir, exist_ok=True)
        if self.env.episode_number > 0:
            results_plotter.EPISODES_WINDOW=10
            results_plotter.plot_curves([(np.arange(self.env.episode_number),np.asarray(self.env.episode_final_reward[-self.env.episode_number:]))],'episode','Episode rewards')
            plt.ylabel("reward")
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(save_dir + "/reward_over_episodes_episode{}.png".format(self.env.episode_number))
            plt.close()


class DistanceReward(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(DistanceReward, self).__init__(env, task)
        self.prev_obj1_position = None
        self.prev_obj2_position = None


    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        o1 = observation["actual_state"]
        o2 = observation["goal_state"]
        reward = self.calc_dist_diff(o1, o2)
        self.task.check_distance_threshold(observation)
        self.rewards_history.append(reward)
        return reward

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_obj1_position = None
        self.prev_obj2_position = None

    def calc_dist_diff(self, obj1_position, obj2_position):
        """
        Calculate change in the distance between 2 objects in previous and in current step. Normalize the change by the value of distance in previous step.

        Params:
            :param obj1_position: (list) Position of the first object
            :param obj2_position: (list) Position of the second object
        Returns:
            :return norm_diff: (float) Normalized difference of distances between 2 objects in previsous and in current step
        """
        if self.prev_obj1_position is None and self.prev_obj2_position is None:
            self.prev_obj1_position = obj1_position
            self.prev_obj2_position = obj2_position
        self.prev_diff = self.task.calc_distance(self.prev_obj1_position, self.prev_obj2_position)

        current_diff = self.task.calc_distance(obj1_position, obj2_position)
        norm_diff = (self.prev_diff - current_diff) / self.prev_diff

        self.prev_obj1_position = obj1_position
        self.prev_obj2_position = obj2_position

        return norm_diff

class ComplexDistanceReward(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 3 objects, e.g. 2 objects and gripper for complex tasks

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(ComplexDistanceReward,self).__init__(env, task)
        self.prev_obj3_position = None

    def compute(self, observation):
        """
        Compute reward signal based on distances between 3 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        reward = self.calc_dist_diff(observation["actual_state"], observation["goal_state"], observation["additional_obs"]["endeff_xyz"])
        self.task.check_distance_threshold(observation=observation)
        self.rewards_history.append(reward)
        return reward

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        super().reset()
        self.prev_obj3_position = None

    def calc_dist_diff(self, obj1_position, obj2_position, obj3_position):
        """
        Calculate change in the distances between 3 objects in previous and in current step. Normalize the change by the value of distance in previous step.

        Params:
            :param obj1_position: (list) Position of the first object
            :param obj2_position: (list) Position of the second object
            :param obj3_position: (list) Position of the third object
        Returns:
            :return norm_diff: (float) Sum of normalized differences of distances between 3 objects in previsous and in current step
        """
        if self.prev_obj1_position is None and self.prev_obj2_position is None and self.prev_obj3_position is None:
            self.prev_obj1_position = obj1_position
            self.prev_obj2_position = obj2_position
            self.prev_obj3_position = obj3_position

        prev_diff_12 = self.task.calc_distance(self.prev_obj1_position, self.prev_obj2_position)
        current_diff_12 = self.task.calc_distance(obj1_position, obj2_position)

        prev_diff_13 = self.task.calc_distance(self.prev_obj1_position, self.prev_obj3_position)
        current_diff_13 = self.task.calc_distance(obj1_position, obj3_position)

        prev_diff_23 = self.task.calc_distance(self.prev_obj2_position, self.prev_obj3_position)
        current_diff_23 = self.task.calc_distance(obj2_position, obj3_position)

        norm_diff = (prev_diff_13 - current_diff_13) / prev_diff_13 + (prev_diff_23 - current_diff_23) / prev_diff_23 + 10*(prev_diff_12 - current_diff_12) / prev_diff_12

        self.prev_obj1_position = obj1_position
        self.prev_obj2_position = obj2_position
        self.prev_obj3_position = obj3_position

        return norm_diff


class SparseReward(Reward):
    """
    Reward class for sparse reward signal

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(SparseReward, self).__init__(env, task)

    def reset(self):
        pass

    def compute(self, observation=None):
        """
        Compute sparse reward signal. Reward is 0 when goal is reached, -1 in every other step.

        Params:
            :param observation: Ignored
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        reward = -1

        if self.task.check_distance_threshold(observation):
            reward += 1.0

        self.rewards_history.append(reward)
        return reward


class PokeReachReward(Reward):

    def __init__(self, env, task):
        super(PokeReachReward, self).__init__(env, task)

        self.threshold = 0.1
        self.last_distance = None
        self.last_gripper_distance = None
        self.moved = False

        self.prev_poker_position = [None]*3

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """

        self.last_distance = None
        self.last_gripper_distance = None
        self.moved = False
        self.prev_poker_position = [None]*3

    def compute(self, observation=None):
        poker_position, distance, gripper_distance = self.init(observation)
        reward = self.count_reward(poker_position, distance, gripper_distance)
        self.finish(observation, poker_position, distance, gripper_distance, reward)
        return reward

    def init(self, observation):
        # load positions
        goal_position = observation[0:3]
        poker_position = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[6:9])

        for i in range(len(poker_position)):
            poker_position[i] = round(poker_position[i], 4)

        distance = round(self.env.task.calc_distance(goal_position, poker_position), 7)
        gripper_distance = self.env.task.calc_distance(poker_position, gripper_position)
        self.initialize_positions(poker_position, distance, gripper_distance)

        return poker_position, distance, gripper_distance

    def finish(self, observation, poker_position, distance, gripper_distance, reward):
        self.update_positions(poker_position, distance, gripper_distance)
        self.check_strength_threshold()
        self.task.check_poke_threshold(observation)
        self.rewards_history.append(reward)

    def count_reward(self, poker_position, distance, gripper_distance):
        reward = 0
        if self.check_motion(poker_position):
            reward += self.last_gripper_distance - gripper_distance
        reward += 100*(self.last_distance - distance)
        return reward

    def initialize_positions(self, poker_position, distance, gripper_distance):
        if self.last_distance is None:
            self.last_distance = distance

        if self.last_gripper_distance is None:
            self.last_gripper_distance = gripper_distance

        if self.prev_poker_position[0] is None:
            self.prev_poker_position = poker_position

    def update_positions(self, poker_position, distance, gripper_distance):
        self.last_distance = distance
        self.last_gripper_distance = gripper_distance
        self.prev_poker_position = poker_position

    def check_strength_threshold(self):
        if self.task.check_object_moved(self.env.task_objects[1], 2):
            self.env.episode_over = True
            self.env.episode_failed = True
            self.env.episode_info = "too strong poke"

    def is_poker_moving(self, poker):
        if self.prev_poker_position[0] == poker[0] and self.prev_poker_position[1] == poker[1]:
            return False
        elif self.env.episode_steps > 25:   # it slightly moves on the beginning
            self.moved = True
        return True

    def check_motion(self, poker):
        if not self.is_poker_moving(poker) and self.moved:
            self.env.episode_over = True
            self.env.episode_failed = True
            self.env.episode_info = "too weak poke"
            return True
        elif self.is_poker_moving(poker):
            return False
        return True

class VectorReward(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(VectorReward, self).__init__(env, task)

        self.prev_goals_positions = [None]*(len(self.env.task_objects_names))
        self.prev_distractors_positions = [None]*(len(self.env.distractors))
        self.prev_links_positions = [None]*11
        self.prev_gipper_position = [None, None, None]

        self.touches = 0

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_goals_positions = [None]*(len(self.env.task_objects_names))
        self.prev_distractors_positions = [None]*(len(self.env.distractors))
        self.prev_links_positions = [None]*(self.env.robot.end_effector_index+1)
        self.prev_gipper_position = [None, None, None]

        self.touches = 0
        self.env.distractor_stopped = False

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        goals, distractors, arm, links = [], [], [], []
        self.fill_objects_lists(goals, distractors, arm, links, observation)
        reward = 0

        contact_points = [self.env.p.getContactPoints(self.env.robot.robot_uid,
                          self.env.env_objects[-1].uid, x, -1)
                          for x in range(0,self.env.robot.end_effector_index+1)]
        for p in contact_points:
            if p:
                reward = -5

                self.touches += 1
                self.rewards_history.append(reward)
                if self.touches > 20:
                    if self.env.episode_reward > 0:
                        self.env.episode_reward = -1

                    self.episode_number = 1024
                    self.env.episode_info = "Arm crushed the distractor"
                    self.env.episode_over = True
                    self.env.episode_failed = True
                    return reward

        # end_effector  = links[self.env.robot.gripper_index]
        gripper = links[-1] # = end effector
        gripper       = links[-1] # = end effector
        goal_position = goals[0]
        distractor = self.env.env_objects[1]
        closest_distractor_points = self.env.p.getClosestPoints(self.env.robot.robot_uid, distractor.uid, self.env.robot.gripper_index)

        minimum = float('inf')
        for point in closest_distractor_points:
            if point[8] < minimum:
                minimum = point[8]
                closest_distractor_point = list(point[6])

        if not self.prev_gipper_position[0]:
            self.prev_gipper_position = gripper

        pull_amplifier = self.task.calc_distance(self.prev_gipper_position, goal_position)/1
        push_amplifier = (1/pow(self.task.calc_distance(closest_distractor_point, self.prev_gipper_position), 1/2))-2

        if push_amplifier < 0:
            push_amplifier = 0

        ideal_vector = Vector(self.prev_gipper_position, goal_position, self.env)
        push_vector = Vector(closest_distractor_point, self.prev_gipper_position, self.env)
        pull_vector = Vector(self.prev_gipper_position, goal_position, self.env)

        push_vector.set_len(push_amplifier)
        pull_vector.set_len(pull_amplifier)

        push_vector.add(pull_vector)
        force_vector = push_vector

        force_vector.add(ideal_vector)
        optimal_vector = force_vector
        optimal_vector.multiply(0.005)

        real_vector = Vector(self.prev_gipper_position, gripper, self.env)

        if real_vector.norm == 0:
            reward += 0
        else:
            reward += np.dot(self.set_vector_len(optimal_vector.vector, 1), self.set_vector_len(real_vector.vector, 1))

        self.prev_gipper_position = gripper
        self.task.check_distance_threshold([goal_position, gripper], threshold=0.1)
        self.rewards_history.append(reward)

        if self.task.calc_distance(goal_position, gripper) <= 0.11:
            self.env.episode_over = True
            if self.env.episode_steps == 1:
                self.env.episode_info = "Task completed in initial configuration"
            else:
                self.env.episode_info = "Task completed successfully"

        return reward

    def visualize_vectors(self, gripper, goal_position, force_vector, optimal_vector):
        self.env.p.addUserDebugLine(self.prev_gipper_position, goal_position, lineColorRGB=(255, 255, 255), lineWidth = 1, lifeTime = 0.1)
        self.env.p.addUserDebugLine(gripper, np.add(np.array(force_vector), np.array(gripper)), lineColorRGB=(255, 0, 0), lineWidth = 1, lifeTime = 0.1)
        self.env.p.addUserDebugLine(gripper, np.add(np.array(optimal_vector*100), np.array(gripper)), lineColorRGB=(0, 0, 255), lineWidth = 1, lifeTime = 0.1)

    def fill_objects_lists(self, goals, distractors, arm, links, observation):

        # observation:
        #              first n 3: goals         (n = number of goals)       (len(self.env.task_objects_names))
        #              last  n 3: distractors   (n = number of distractors) (len(self.env.distractors))
        #              next    3: arm
        #              lasting 3: links

        j = 0
        while len(goals) < len(self.env.task_objects_names):
            goals.append(list(observation[j:j+3]))
            j += 3

        while len(distractors) < len(self.env.distractors):
            distractors.append(list(observation[j:j+3]))
            j += 3

        while len(arm) < 1:
            arm.append(list(observation[j:j+3]))
            j += 3

        while len(links) < self.env.robot.observed_links_num:
            links.append(list(observation[j:j+3]))
            j += 3

    def add_vectors(self, v1, v2):
        r = []
        for i in range(len(v1)):
            r.append(v1[i] + v2[i])
        return r

    def count_vector_norm(self, vector):
        return math.sqrt(np.dot(vector, vector))

    def get_dot_product(self, v1, v2):
        product = 0
        for i in range(len(v1)):
            product += v1[i]* v2[i]
        return product

    def move_to_origin(self, vector):
        a = vector[1][0] - vector[0][0]
        b = vector[1][1] - vector[0][1]
        c = vector[1][2] - vector[0][2]
        return [a, b, c]

    def multiply_vector(self, vector, multiplier):
        return np.array(vector) * multiplier

    def set_vector_len(self, vector, len):
        norm = self.count_vector_norm(vector)
        vector = self.multiply_vector(vector, 1/norm)
        return self.multiply_vector(vector, len)


class SwitchReward(DistanceReward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects,
    angle of switch and difference between point and line (function used for that: calc_direction_3d()).
    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(SwitchReward, self).__init__(env, task)
        self.x_obj = None
        self.y_obj = None
        self.z_obj = None
        self.x_bot = None
        self.y_bot = None
        self.z_bot = None

        self.x_obj_curr_pos = None
        self.y_obj_curr_pos = None
        self.z_obj_curr_pos = None
        self.x_bot_curr_pos = None
        self.y_bot_curr_pos = None
        self.z_bot_curr_pos = None

        # auxiliary variables
        self.offset = None
        self.prev_val = None
        self.debug = True

        # coefficients used to calculate reward
        self.k_w = 0.4    # coefficient for distance between actual position of robot's gripper and generated line
        self.k_d = 0.3    # coefficients for absolute distance gripper and end position
        self.k_a = 1      # coefficient for calculated angle reward

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects, angle of switch and difference between point and line
        (function used for that: calc_direction_3d()).
        The position of the objects must be present in observation.
        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        o1 = observation["goal_state"]
        gripper_position = self.get_accurate_gripper_position(observation["actual_state"])
        self.set_variables(o1, gripper_position)    # save local positions of task_object and gripper to global positions
        self.set_offset(x=-0.1, z=0.25)
        if self.x_obj > 0:
            if self.debug:
                self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], [-0.7, self.y_obj, self.z_obj],
                                            lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1)
            w = self.calc_direction_3d(self.x_obj, self.y_obj, self.z_obj, 0.7, self.y_obj, self.z_obj,
                                       self.x_bot_curr_pos,
                                       self.y_bot_curr_pos, self.z_bot_curr_pos)

        else:
            if self.debug:
                self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], [0.7, self.y_obj, self.z_obj],
                                            lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1)
            w = self.calc_direction_3d(self.x_obj, self.y_obj, self.z_obj, -0.7, self.y_obj, self.z_obj,
                                       self.x_bot_curr_pos,
                                       self.y_bot_curr_pos, self.z_bot_curr_pos)

        d = self.abs_diff()
        a = self.calc_angle_reward()

        reward = - self.k_w * w - self.k_d * d + self.k_a * a
        self.task.check_distance_threshold(observation=observation)
        self.rewards_history.append(reward)
        if self.debug:
            self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], gripper_position,
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.03)
            self.env.p.addUserDebugText(f"reward:{reward:.3f}, w:{w * self.k_w:.3f}, d:{d * self.k_d:.3f},"
                                        f" a:{a * self.k_a:.3f}",
                                        [1, 1, 1], textSize=2.0, lifeTime=0.05, textColorRGB=[0.6, 0.0, 0.6])
        return reward

    def reset(self):
        """
        Reset current positions of switch, robot, initial position of switch, robot and previous angle of switch.
        Call this after the end of an episode.
        """
        self.x_obj = None
        self.y_obj = None
        self.z_obj = None
        self.x_bot = None
        self.y_bot = None
        self.z_bot = None

        self.x_obj_curr_pos = None
        self.y_obj_curr_pos = None
        self.z_obj_curr_pos = None
        self.x_bot_curr_pos = None
        self.y_bot_curr_pos = None
        self.z_bot_curr_pos = None

        # auxiliary variables
        self.offset = None
        self.prev_val = None

    def set_variables(self, o1, o2):
        """
        Assign local values to global variables
        Params:
            :param o1: (list) Position of switch in space [x, y, z]
            :param o2: (list) Position of robot in space [x, y, z]
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        if self.x_obj is None:
            self.x_obj = o1[0]

        if self.y_obj is None:
            self.y_obj = o1[1]

        if self.z_obj is None:
            self.z_obj = o1[2]

        if self.x_bot is None:
            self.x_bot = o2[0]

        if self.y_bot is None:
            self.y_bot = o2[1]

        if self.z_bot is None:
            self.z_bot = o2[2]

        self.x_obj_curr_pos = o1[0]
        self.y_obj_curr_pos = o1[1]
        self.z_obj_curr_pos = o1[2]
        self.x_bot_curr_pos = o2[0]
        self.y_bot_curr_pos = o2[1]
        self.z_bot_curr_pos = o2[2]

    def set_offset(self, x=0.0, y=0.0, z=0.0):
        """
        Set offset position of switch
        Params:
            :param x: (int) The number by which is coordinate x changed
            :param y: (int) The number by which is coordinate y changed
            :param z: (int) The number by which is coordinate z changed
        """
        if self.offset is None:
            self.offset = True
            if self.x_obj > 0:
                self.x_obj -= x
                self.y_obj += y
                self.z_obj += z
            else:
                self.x_obj += x
                self.y_obj += y
                self.z_obj += z

    @staticmethod
    def calc_direction_2d(x1, y1, x2, y2, x3, y3):
        """
        Calculate difference between point - (actual position of robot's gripper P - [x3, y3])
        and line - (perpendicular position from middle of switch: A - [x1, y1]; final position of robot: B - [x2, y2) in 2D
        Params:
            :param x1: (float) Coordinate x of switch
            :param y1: (float) Coordinate y of switch
            :param x2: (float) Coordinate x of final position of robot
            :param y2: (float) Coordinate y of final position of robot
            :param x3: (float) Coordinate x of robot's gripper
            :param y3: (float) Coordinate y of robot's gripper
        Returns:
            :return x: (float) The nearest point[x] to robot's gripper on the line
            :return y: (float) The nearest point[y] to robot's gripper on the line
            :return d: (float) Distance between line and robot's gripper
        """
        x = x1 + ((x1 - x2) * (x1 * x2 + x1 * x3 - x2 * x3 + y1 * y2 + y1 * y3 - y2 * y3 - x1 ** 2 - y1 ** 2)) / (
                x1 ** 2 - 2 * x1 * x2 + x2 ** 2 + y1 ** 2 - 2 * y1 * y2 + y2 ** 2)
        y = y1 + ((y1 - y2) * (x1 * x2 + x1 * x3 - x2 * x3 + y1 * y2 + y1 * y3 - y2 * y3 - x1 ** 2 - y1 ** 2)) / (
                x1 ** 2 - 2 * x1 * x2 + x2 ** 2 + y1 ** 2 - 2 * y1 * y2 + y2 ** 2)
        d = sqrt((x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2) ** 2 / (
                x1 ** 2 - 2 * x1 * x2 + x2 ** 2 + y1 ** 2 - 2 * y1 * y2 + y2 ** 2))
        return [x, y, d]

    @staticmethod
    def calc_direction_3d(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        """
        Calculate difference between point - (actual position of robot's gripper P - [x3, y3, z3])
        and line - (perpendicular position from middle of switch: A - [x1, y1, z1]; final position of robot: B - [x2, y2, z2]) in 3D
        Params:
            :param x1: (float) Coordinate x of initial position of robot
            :param y1: (float) Coordinate y of initial position of robot
            :param z1: (float) Coordinate z of initial position of robot

            :param x2: (float) Coordinate x of final position of robot
            :param y2: (float) Coordinate y of final position of robot
            :param z2: (float) Coordinate z of final position of robot

            :param x3: (float) Coordinate x of robot's gripper
            :param y3: (float) Coordinate y of robot's gripper
            :param z3: (float) Coordinate z of robot's gripper
        Returns:
            :return d: (float) Distance between line and robot's gripper
        """
        x = x1 - ((x1 - x2) * (
                x1 * (x1 - x2) - x3 * (x1 - x2) + y1 * (y1 - y2) - y3 * (y1 - y2) + z1 * (z1 - z2) - z3 * (
                z1 - z2))) / ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        y = y1 - ((y1 - y2) * (
                x1 * (x1 - x2) - x3 * (x1 - x2) + y1 * (y1 - y2) - y3 * (y1 - y2) + z1 * (z1 - z2) - z3 * (
                z1 - z2))) / ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        z = z1 - ((z1 - z2) * (
                x1 * (x1 - x2) - x3 * (x1 - x2) + y1 * (y1 - y2) - y3 * (y1 - y2) + z1 * (z1 - z2) - z3 * (
                z1 - z2))) / ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        d = sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2)

        return d

    def abs_diff(self):
        """
        Calculate absolute differance between task_object and gripper
        Returns:
            :return abs_diff: (float) Absolute distance to switch
        """
        x_diff = self.x_obj_curr_pos - self.x_bot_curr_pos
        y_diff = self.y_obj_curr_pos - self.y_bot_curr_pos
        z_diff = self.z_obj_curr_pos - self.z_bot_curr_pos

        abs_diff = sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
        return abs_diff

    def get_angle(self):
        """
        Calculate angle of switch/button
        Returns:
            :return angle: (int) Angle of switch
        """
        assert self.task.task_type in ["switch", "turn", "press"], "Expected task type switch, press or turn"
        pos = self.env.p.getJointState(self.env.task_objects["goal_state"].get_uid(), 0)
        angle = pos[0] * 180 / math.pi  # in degrees
        if self.task.task_type == "switch":
            return int(abs(angle))
        elif self.task.task_type == "press":
            if abs(angle)>1.71:
                print("Press")
            return abs(angle)
        else:
            return -angle

    def calc_angle_reward(self):
        """
        Calculate additional reward for switch task
        Returns:
            :return reward: (int) Additional reward value
        """
        angle = self.get_angle()
        assert self.task.task_type == "switch", "Function implemented only for switch"
        if self.prev_val is None:
            self.prev_val = angle
        k = angle // 2
        reward = k * angle
        if reward >= 162:
            reward += 50
        reward /= 100
        if self.prev_val == angle:
            reward = 0
        self.prev_val = angle
        return reward


class ButtonReward(SwitchReward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects,
    button's position and difference between point and line (function used for that: calc_direction_3d()).
    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(ButtonReward, self).__init__(env, task)
        self.k_w = self.env.coefficient_kw
        self.k_d = self.env.coefficient_kd
        self.k_a = self.env.coefficient_ka

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 objects, position of button and difference between point and line
        (function used for that: calc_direction_3d()).
        The position of the objects must be present in observation.
        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        goal = observation["goal_state"]
        gripper_position = self.get_accurate_gripper_position(observation["actual_state"])
        self.set_variables(goal, gripper_position)
        self.set_offset(z=0.16)

        w = self.calc_direction_3d(self.x_obj, self.y_obj, 1, self.x_obj, self.y_obj, self.z_obj,
                                   self.x_bot_curr_pos, self.y_bot_curr_pos, self.z_bot_curr_pos)
        d = self.abs_diff()
        if gripper_position[2] < 0.15:
            d *= 5
        a = self.calc_press_reward()
        reward = - self.k_w * w - self.k_d * d + self.k_a * a
        if self.debug:
            self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], [self.x_obj, self.y_obj, 1],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1)

            self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], gripper_position,
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.03)

            self.env.p.addUserDebugText(f"reward:{reward:.3f}, w:{w * self.k_w:.3f}, d:{d * self.k_d:.3f},"
                                        f" a:{a * self.k_a:.3f}",
                                        [1, 1, 1], textSize=2.0, lifeTime=0.05, textColorRGB=[0.6, 0.0, 0.6])

        self.task.check_distance_threshold(observation=observation)
        self.rewards_history.append(reward)
        return reward

    def set_offset(self, x=0.0, y=0.0, z=0.0):
        if self.offset is None:
            self.offset = True
            self.x_obj += x
            self.y_obj += y
            self.z_obj += z

    @staticmethod
    def set_vector_len(vector, len):
        """
        Scale given vector so its length is equal to len
        """
        norm = math.sqrt(np.dot(vector, vector))
        if norm == 0:
            return norm
        else:
           return (vector * (1/norm))*len

    def abs_diff(self):
        """
        Calculate absolute difference between task_object and gripper
        """
        x_diff = self.x_obj_curr_pos - self.x_bot_curr_pos
        y_diff = self.y_obj_curr_pos - self.y_bot_curr_pos
        z_diff = self.z_obj_curr_pos - self.z_bot_curr_pos
        abs_diff = sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
        return abs_diff

    def calc_press_reward(self):
        press = (self.get_angle() *100)
        if self.prev_val is None:
            self.prev_val = press
        k = press // 2
        reward = (k * press)/1000
        if reward >= 14:
            reward += 2
        reward /= 10
        if self.prev_val == press:
            reward = 0
        self.prev_val = press
        return reward

class TurnReward(SwitchReward):
    """
    Reward class for reward signal calculation based on distance between 2 points (robot gripper and middle point of predefined line) and angle of handle
    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(TurnReward, self).__init__(env, task)
        self.r = 0.45
        self.k_w = self.env.coefficient_kw
        self.k_d = self.env.coefficient_kd
        self.k_a = self.env.coefficient_ka

    def compute(self, observation):
        """
        Compute reward signal based on distance between 2 points (robot gripper and middle point of predefined line)
        and angle of handle
        The position of the objects must be present in observation.
        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        goal = observation["goal_state"]
        gripper_position = self.get_accurate_gripper_position(observation["actual_state"])
        self.set_variables(goal, gripper_position)
        self.set_offset(z=0.1)

        d = self.threshold_reached()
        a = self.calc_turn_reward()
        reward = - self.k_d * d + a * self.k_a
        if self.debug:
            self.env.p.addUserDebugText(f"reward:{reward:.3f}, d:{d * self.k_d:.3f}, a: {a * self.k_a:.3f}",
                                        [1, 1, 1], textSize=2.0, lifeTime=0.05, textColorRGB=[0.6, 0.0, 0.6])

        self.task.check_distance_threshold(observation=observation)
        self.rewards_history.append(reward)
        return reward


    def set_offset(self, x=0.0, y=0.0, z=0.0):
        if self.offset is None:
            self.offset = True
            self.x_obj += x
            self.y_obj += y
            self.z_obj += z

        self.x_obj_curr_pos += x
        self.y_obj_curr_pos += y
        self.z_obj_curr_pos += z

    def angle_adaptive_reward(self, change_reward=False, visualize=False):
        """
        Calculate difference distance between 2 points (robot gripper and middle point of predefined line)
        """
        alfa = np.deg2rad(-self.get_angle())  # in radians
        k = 0.2
        offset = 0.2
        normalize = 0.1
        coef = 3 * math.pi / 2

        Sx = self.x_obj
        Sy = self.y_obj
        Sz = self.z_obj

        Px = self.x_bot_curr_pos
        Py = self.y_bot_curr_pos
        Pz = self.z_bot_curr_pos

        l = coef - offset if change_reward  else coef + offset
        l += normalize
        Ax = self.r * math.cos(alfa + l) + Sx
        Ay = self.r * math.sin(alfa + l) + Sy

        Bx = k * math.cos(alfa + l) + Sx
        By = k * math.sin(alfa + l) + Sy

        AB_mid_x = (k + (self.r - k)/2) * math.cos(alfa + l) + Sx
        AB_mid_y = (k + (self.r - k)/2) * math.sin(alfa + l) + Sy

        P_MID_diff_x = AB_mid_x - Px
        P_MID_diff_y = AB_mid_y - Py
        P_MID_diff_z = Sz - Pz

        if visualize:
            self.env.p.addUserDebugLine([Ax, Ay, Sz], [Bx, By, Sz],
                                        lineColorRGB=(1, 0, 1), lineWidth=3, lifeTime=0.03)
            self.env.p.addUserDebugLine([P_MID_diff_x, P_MID_diff_y, P_MID_diff_z], [Ax, Ay, Sz],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=0.03)
            self.env.p.addUserDebugLine([Px, Py, Pz], [AB_mid_x, AB_mid_y, Sz],
                                        lineColorRGB=(0, 1, 1), lineWidth=3, lifeTime=0.03)

        d = sqrt(P_MID_diff_x ** 2 + P_MID_diff_y ** 2 + P_MID_diff_z ** 2)
        return d

    def is_touch(self):
        """
        Checking if gripper is touching handle
        """
        contact_points = [self.env.p.getContactPoints(self.env.robot.robot_uid, self.env.env_objects["goal_state"].uid, x, 0)
                          for x in range(0, self.env.robot.end_effector_index+1)]
        if any(contact_points):
                return True
        return False

    def threshold_reached(self):
        """
        Switching output reward according to distance
        """
        threshold = 0.1
        d = self.angle_adaptive_reward()
        if d < threshold and self.is_touch():
            return self.angle_adaptive_reward(change_reward=True, visualize=self.debug) - 1
        return self.angle_adaptive_reward(visualize=self.debug)

    def calc_turn_reward(self):
        turn = int(self.get_angle())
        reward = turn
        if self.prev_val is None:
            self.prev_val = turn
        if self.prev_val >= turn:
            reward = 0
        if reward < 0 and self.prev_val < turn:
            reward = 0
        self.prev_val = turn
        return reward
