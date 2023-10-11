import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import results_plotter
import os
import math
from math import sqrt, fabs, exp, pi, asin
from myGym.utils.vector import Vector
import random

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
        self.current_network = 0
        self.num_networks = env.num_networks
        #self.check_num_networks()
        self.network_rewards = [0] * self.num_networks
        self.use_magnet = True

    def network_switch_control(self, observation):
        if self.env.num_networks <= 1:
            print("Cannot switch networks in a single-network scenario")
        else:
           if self.env.network_switcher == "gt":
                self.current_network = self.decide(observation)
           else:
               raise NotImplementedError("Currently only implemented ground truth ('gt') network switcher")
        return self.current_network

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

    def get_magnetization_status(self):
        return self.use_magnet


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

    def decide(self, observation=None):
        return random.randint(0, self.env.num_networks-1)

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
        #self.task.check_distance_threshold(observation)
        self.task.check_goal()
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
        gripper_name = [x for x in self.env.task.obs_template["additional_obs"] if "endeff" in x][0]
        reward = self.calc_dist_diff(observation["actual_state"], observation["goal_state"], observation["additional_obs"][gripper_name])
        #self.task.check_distance_threshold(observation=observation)
        self.task.check_goal()
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

        if self.task.check_goal():
            reward += 1.0

        self.rewards_history.append(reward)
        return reward


# vector rewards
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

        while len(links) < len(self.env.robot.observe_all_links):
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
        self.offset     = None
        self.prev_val   = None
        self.debug      = True

        # coefficients used to calculate reward
        self.k_w = 0.4    # coefficient for distance between actual position of robot's gripper and generated line
        self.k_d = 0.3    # coefficient for absolute distance between gripper and end position
        self.k_a = 1      # coefficient for calculated angle reward

        self.reach_line     = False
        self.dist_offset    = 0.01

        self.x_bot_last_pos = None
        self.y_bot_last_pos = None
        self.z_bot_last_pos = None

        self.last_pos_on_line = None
        self.last_angle = None

        self.bonus_reward   = 10

        self.right_bonus    = False
        self.left_bonus     = False

    def compute(self, observation):
        goal_pos            = observation["goal_state"]
        gripper_position    = observation["actual_state"]
        self.set_variables(goal_pos, gripper_position)    # save local positions of task_object and gripper to global positions

        points   =  [(self.x_obj+0.2, self.y_obj, self.z_obj+0.2),
                    (self.x_obj-0.2, self.y_obj, self.z_obj+0.2)]
        
        cur_pos  =  (self.x_bot_curr_pos, self.y_bot_curr_pos, self.z_bot_curr_pos)
        last_pos =  (self.x_bot_last_pos, self.y_bot_last_pos, self.z_bot_last_pos)

        self.env.p.addUserDebugLine(points[0], points[1], lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=1)

        distances=  [self.get_distance(points[0], cur_pos), self.get_distance(points[1], cur_pos),
                     self.get_distance(goal_pos, cur_pos), self.get_distance_line_point(points[0], points[1], cur_pos)]
        
        last_dist=  [self.get_distance(points[0], last_pos), self.get_distance(points[1], last_pos),
                     self.get_distance(goal_pos, last_pos), self.get_distance_line_point(points[0], points[1], last_pos)]

        rew_1point  =   0
        rew_line    =   0
        rew_2point  =   0

        if not self.reach_line:
            rew_1point = self.lin_eval(distances[0]) if distances[0] < last_dist[0] else -1/self.lin_eval(distances[0]) 
            if distances[1] <= self.dist_offset:
                self.reach_line = True
        else:
            rew_line    =   self.lin_eval(distances[3]) if distances[3] < last_dist[3] else -1/self.lin_eval(distances[3]) 
            rew_2point  =   self.lin_eval(distances[1]) if distances[1] < last_dist[1] else -1/self.lin_eval(distances[1])
            # if distances[3] < self.dist_offset:
            #     rew_line += self.norm_eval_4_line(points[0], points[1], cur_pos)

        reward = rew_1point + rew_line + rew_2point + self.get_bonus(*distances) + self.get_angle_reward()

        if self.debug:
            self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], gripper_position,
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)

            self.env.p.addUserDebugText(f"line : {self.reach_line} reward: {reward}",
                                        [0.5, 0.5, 0.5], textSize=2.0, lifeTime=0.05, textColorRGB=[0.6, 0.0, 0.6])

        self.task.check_goal()
        self.rewards_history.append(reward)
        self.set_last_pos()
        return reward 

    def normap_dist_func(self, x: float, variance_sqrt = 0.2) -> float:
        return (1/(variance_sqrt*sqrt(2*pi)))*exp((-0.5)*((x/variance_sqrt)**2))

    def norm_eval_4_line(self, p1: tuple, p2: tuple, p:tuple, range: float = 3.2) -> float:
        if self.last_pos_on_line == None:
            self.last_pos_on_line = -range
        a = self.get_distance(p1, p)
        b = self.get_distance_line_point(p1,p2, p)
        c = self.get_distance(p2, p)

        dist  = self.get_distance(p1, p2)

        x1 = sqrt(a**2 - b**2)
        x2 = sqrt(c**2 - b**2)

        if x1 > dist or x2 > dist:
            return 0

        x = 2*range*(x1/(x1+x2)) - range
        if self.last_pos_on_line > x:
            return -fabs(x-last_pos_on_line)
        else:
            self.last_pos_on_line = x
        return self.normap_dist_func(x)

    def get_angle_reward(self):
        if self.last_angle == None:
            self.last_angle = 0
        delta = self.get_angle() - self.last_angle
        self.last_angle = self.get_angle()
        return 100*(delta)/(20)

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

        self.x_bot_last_pos = None
        self.y_bot_last_pos = None
        self.z_bot_last_pos = None

        # auxiliary variables
        self.x_last_curr_pos = None
        self.y_last_curr_pos = None
        self.z_last_curr_pos = None

        self.offset = None
        self.prev_val = None

        self.reach_line = False

        self.right_bonus    = False
        self.left_bonus     = False

        self.last_pos_on_line = None
        self.last_angle = None

    def set_last_pos(self):
        """set last robot possition"""
        self.x_bot_last_pos = self.x_bot_curr_pos
        self.y_bot_last_pos = self.y_bot_curr_pos 
        self.z_bot_last_pos = self.z_bot_curr_pos 

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

    def get_bonus(self, dist_1:float, dist_2: float, dist_g: float, dist_line: float) -> float:
        if dist_1 < dist_2:
            if dist_g < 0.05 and dist_line < self.dist_offset and not self.right_bonus:
                self.right_bonus = True
                return self.bonus_reward
        else:
            if dist_g < 0.05 and dist_line < self.dist_offset and not self.left_bonus:
                self.left_bonus = True
                return self.bonus_reward
        return 0

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

    # @staticmethod
    def lin_eval(self, dist : float, max_r: float = 1) -> float:
        if dist <= self.dist_offset:
            reward = max_r
        else:
            b = max_r + self.dist_offset
            reward = b - dist 
        return reward
    

    # @staticmethod
    def exp_eval(self, dist : float, max_r: float = 1) -> float:
        # if last_dist > dist:
        #     return 0
        if dist <= self.dist_offset:
            reward = max_r
        else:
            reward = exp(self.dist_offset - dist) + (max_r - 1) 
        return reward



    @staticmethod
    def get_distance(point1: tuple, point2: tuple = (0, 0, 0)) -> float:
        """ returns distance between two points in 3d,
         if point2 will not set it will return vector lenght,
         if one of points will be not defined return infinity"""
        a = np.array(point1)
        b = np.array(point2)
        if (a == None).any() or (b == None).any():
            return float('inf')
        return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2) 


    @staticmethod
    def get_distance_line_point(lpoint1: tuple, lpoint2: tuple, point: tuple) -> float:
        """calculate distance between line (represented by two points) and point"""
        point = np.array(point)
        if (point == None).any():
            return float('inf')
        dir_vector = np.array( [lpoint2[0]-lpoint1[0],
                                lpoint2[1]-lpoint1[1],
                                lpoint2[2]-lpoint1[2]] )
        
        plp1_vector = np.array( [lpoint1[0]-point[0],
                                lpoint1[1]-point[1],
                                lpoint1[2]-point[2]] )

        cross = np.cross(dir_vector, plp1_vector)

        dist = np.linalg.norm(cross) / np.linalg.norm(dir_vector)
        return dist

    def get_positions(self, observation):
        goal_position = observation["goal_state"]
        poker_position = observation["actual_state"]
        #gripper_name = [x for x in self.env.task.obs_template["additional_obs"] if "endeff" in x][0]
        #gripper_position = observation["additional_obs"]["endeff_xyz"]
        #if self.prev_poker_position[0] is None:
        #    self.prev_poker_position = poker_position
        return goal_position,poker_position,poker_position

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
                print("Button pressed")
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
        # print(reward)
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
        self.k_w = 0.1   # coefficient for distance between actual position of robot's gripper and generated line
        self.k_d = 0.1   # coefficient for absolute distance between gripper and end position
        self.k_a = 1     # coefficient for calculated angle reward

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
        goal_position, object_position, gripper_position = self.get_positions(observation)
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

        #self.task.check_distance_threshold(observation=observation)
        self.task.check_goal()
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
        self.offset     = None
        self.prev_val   = None
        self.debug      = True


        self.reach_line     = False
        self.dist_offset    = 0.05

        self.x_bot_last_pos = None
        self.y_bot_last_pos = None
        self.z_bot_last_pos = None

        self.last_pos_on_line = None
        self.last_angle = None

        self.bonus_reward   = 10

        self.right_bonus    = False
        self.left_bonus     = False

    def compute(self, observation):
        goal_pos            = observation["goal_state"]
        gripper_position    = observation["actual_state"]
        self.set_variables(goal_pos, gripper_position)    # save local positions of task_object and gripper to global positions

        points   =  [(self.x_obj+0.3, self.y_obj-0.2, self.z_obj+0.1),
                    (self.x_obj-0.4, self.y_obj-0.2, self.z_obj+0.1)]
        
        cur_pos  =  (self.x_bot_curr_pos, self.y_bot_curr_pos, self.z_bot_curr_pos)
        last_pos =  (self.x_bot_last_pos, self.y_bot_last_pos, self.z_bot_last_pos)

        self.env.p.addUserDebugLine(points[0], points[1], lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=1)

        distances =  [self.get_distance(points[0], cur_pos), self.get_distance(points[1], cur_pos),
                     self.get_distance(goal_pos, cur_pos), self.get_distance_line_point(points[0], points[1], cur_pos)]
        
        last_dist =  [self.get_distance(points[0], last_pos), self.get_distance(points[1], last_pos),
                     self.get_distance(goal_pos, last_pos), self.get_distance_line_point(points[0], points[1], last_pos)]

        rewards = {
            "max_1point":   1.,
            "min_1point":   0.,
            "max_2point":   2.,
            "min_2point":   1.,
            "max_line":     2.,
            "min_line":     0
        }

        rew_1point  =   0
        rew_line    =   0
        rew_2point  =   0

        if not self.reach_line:
            rew_1point = self.lin_hyp_eval(distances[0], max_r = rewards["max_1point"], default = rewards["min_1point"]) if distances[0] < last_dist[0] else -1.5*self.lin_hyp_eval(distances[0], max_r = rewards["max_1point"], default = rewards["min_1point"]) 
            if distances[3] <= self.dist_offset:
                self.reach_line = True
        else:
            rew_line    =   self.lin_hyp_eval(distances[3], max_r = rewards["max_line"], default = rewards["min_line"]) if distances[3] < last_dist[3] else -1.5*self.lin_hyp_eval(distances[3], max_r = rewards["max_line"], default = rewards["min_line"]) 
            rew_2point  =   self.lin_hyp_eval(distances[1], max_r = rewards["max_2point"], default = rewards["min_2point"]) if distances[1] < last_dist[1] else -1.5*self.lin_hyp_eval(distances[1], max_r = rewards["max_2point"], default = rewards["min_2point"])

        angle_rew = self.get_angle_reward()
        reward = rew_1point + rew_line + rew_2point + angle_rew + self.get_bonus(*distances)

        if self.debug:
            self.env.p.addUserDebugLine([self.x_obj, self.y_obj, self.z_obj], gripper_position,
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)

            self.env.p.addUserDebugText(f"line : {self.reach_line} reward: {reward}",
                                        [0.5, 0.5, 0.5], textSize=2.0, lifeTime=5, textColorRGB=[0.6, 0.0, 0.6])

        self.task.check_goal()
        self.rewards_history.append(reward)
        self.set_last_pos()
        return reward

    def reset(self):
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

        self.x_bot_last_pos = None
        self.y_bot_last_pos = None
        self.z_bot_last_pos = None

        # auxiliary variables
        self.x_last_curr_pos = None
        self.y_last_curr_pos = None
        self.z_last_curr_pos = None

        self.offset = None
        self.prev_val = None

        self.reach_line = False

        self.right_bonus    = False
        self.left_bonus     = False

        self.last_pos_on_line = None
        self.last_angle = None

    def lin_hyp_eval(self, dist: float, max_r: float = 1., default: float = 0.) -> float:
        reward = [-dist+max_r, ((0.5*max_r)**2)/dist][int(dist > 0.5*max_r)]
        return reward + default
  
  
    def hyperb_reward(self, dist: float, a: float = 0.1, maxR: float = float("inf"), minR: float =  0.) -> float:
        assert maxR > minR, "maxR must be higher then minR"
        return np.median([maxR, a/dist, minR])
    

    def lin_penalty(self, dist: float, k: float = 1., maxP: float = -float('inf'), minP: float = 0.) -> float:
        assert maxP < minP, "maxP must be less then minP"
        return np.median([maxP, -fabs(k)*dist, minP])


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

    def threshold_reached(self):
        """
        Switching output reward according to distance
        """
        threshold = 0.1
        d = self.angle_adaptive_reward()
        if d < threshold and self.env.robot.touch_sensors_active(self.env.env_objects["goal_state"]):
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

    def get_angle_between_vectors(self, v1, v2):
        return math.acos(np.dot(v1, v2)/(self.count_vector_norm(v1)*self.count_vector_norm(v2)))

class PokeReachReward(SwitchReward):

    def __init__(self, env, task):
        super(PokeReachReward, self).__init__(env, task)

        self.cube_offest               = 0.1
        self.dist_offset               = 0.05

        self.last_points               = None 
        self.point_was_reached         = False

    def reset(self):
        self.last_points               = None
        self.point_was_reached         = False
    
    def compute(self, observation=None):
        goal_pos, cube_pos, gripper = self.set_points(observation)
        cube_last_pos, gripper_last = self.last_points if self.last_points != None else [cube_pos, gripper]

        point_grg = self.get_point_grg(goal_pos, cube_pos)

        state = [(gripper, point_grg),(gripper, cube_pos)][int(self.point_was_reached)]
        state_last = [(gripper_last, point_grg),(gripper_last, cube_pos)][int(self.point_was_reached)]

        cur_dist = self.get_distance(*state)
        if cur_dist <= self.dist_offset:
            self.line_was_reached = True
        last_dist = self.get_distance(*state_last)
        dist_cube_goal = self.get_distance(cube_pos, goal_pos)


        rew_point = self.exp_eval(cur_dist) if cur_dist < last_dist else self.lin_penalty(last_dist, min_penalty = 1)
        rew_angle = self.get_angle_reward_2D(goal_pos,cube_last_pos, cube_pos)

        reward = rew_point + rew_angle 
        self.env.p.addUserDebugText(f"{rew_point}, {rew_angle}", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[1,0,0])
        self.set_last_positions(cube_pos, gripper)
        self.rewards_history.append(reward)
        self.task.check_goal()
        return reward


    # def __init__(self, env, task):
    #     super(PokeReachReward, self).__init__(env, task)

    #     self.threshold = 0.1
    #     self.last_distance = None
    #     self.last_gripper_distance = None
    #     self.moved = False
    #     self.prev_poker_position = [None]*3

    # def reset(self):
    #     """
    #     Reset stored value of distance between 2 objects. Call this after the end of an episode.
    #     """
    #     self.last_distance = None
    #     self.last_gripper_distance = None
    #     self.moved = False
    #     self.prev_poker_position = [None]*3

    # def compute(self, observation=None):
    #     poker_position, distance, gripper_distance = self.init(observation)
    #     self.task.check_goal()
    #     reward = self.count_reward(poker_position, distance, gripper_distance)
    #     self.finish(observation, poker_position, distance, gripper_distance, reward)
    #     return reward

    def init(self, observation):
        # load positions
        goal_position = observation["goal_state"]
        poker_position = observation["actual_state"]
        gripper_name = [x for x in self.env.task.obs_template["additional_obs"] if "endeff" in x][0]
        gripper_position = self.env.robot.get_accurate_gripper_position()

        for i in range(len(poker_position)):
            poker_position[i] = round(poker_position[i], 4)

        distance = round(self.env.task.calc_distance(goal_position, poker_position), 7)
        gripper_distance = self.env.task.calc_distance(poker_position, gripper_position)
        self.initialize_positions(poker_position, distance, gripper_distance)

        return poker_position, distance, gripper_distance

    def finish(self, observation, poker_position, distance, gripper_distance, reward):
        self.update_positions(poker_position, distance, gripper_distance)
        self.check_strength_threshold()
        self.task.check_distance_threshold(observation)
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
        if self.task.check_object_moved(self.env.task_objects["actual_state"], 2):
            self.env.episode_over = True
            self.env.episode_failed = True
            self.env.episode_info = "poke too strong"

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
    
    def set_points(self, observation) -> tuple:
        return np.array(observation["goal_state"]), np.array(observation["actual_state"]), np.array(observation["additional_obs"]["endeff_xyz"])
    
    def lin_penalty(self, dist: float, min_penalty: float = 0, k: float = 1) -> float:
        return -dist*fabs(k) - fabs(min_penalty)

    def get_point_grg(self, point1: 'numpy.ndarray', point2: 'numpy.ndarray') -> 'numpy.ndarray':
        dir_unit_vctor = (point2-point1) / np.linalg.norm(point2-point1)
        point3 = dir_unit_vctor*self.cube_offest + point2
        return point3
    
    def set_last_positions(self, cube: 'numpy.ndarray', gripper: 'numpy.ndarray'):
        self.last_points = [cube, gripper]
    
    @staticmethod
    def get_unit_vector(tail: 'numpy.ndarray', head: 'numpy.ndarray') -> 'numpy.ndarray':
        vector = head - tail
        return vector/np.linalg.norm(vector)

    @staticmethod
    def get_angle_2vec(vecU1: 'numpy.ndarray', vecU2: 'numpy.ndarray') -> float:
        M = np.array([[vecU1[0], -vecU1[1]], [vecU1[1], vecU1[0]]])
        cos, sin = np.linalg.solve(M, vecU2)
        return -asin(sin)*180/pi

    def is_moved_2D(self, last_pos: 'numpy.ndarray', cur_pos: 'numpy.ndarray') -> bool:
        return (np.abs(last_pos[:-1] - cur_pos[:-1]) > 0.0001).any()

    def get_angle_reward_2D(self, head1: 'numpy.ndarray', tail: 'numpy.ndarray', head2: 'numpy.ndarray',) -> float:
        vecU1 = self.get_unit_vector(tail[:-1], head1[:-1])
        vecU2 = self.get_unit_vector(tail[:-1], head2[:-1])
        if not self.is_moved_2D(tail, head2):
            return 0
        angle = fabs(self.get_angle_2vec(vecU1, vecU2)) 
        return [-angle, 200/angle][int(angle > 20)]

class PushReward2(PokeReachReward):

    def __init__(self, env, task):
        super(PushReward2, self).__init__(env, task)

        self.cube_offest               = 0.1
        self.dist_offset               = 0.05

        self.last_points               = None 
        self.point_was_reached         = False

        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None 

        self.k_p = 1   # distance coefficient between cube position and gripper position
        self.k_ct = 2   # distance coefficient between cube position and target position
        self.k_gt = 1   # distance coefficient between gripper position and target position
        self.k_a = 1    # angle(<GCT) coefficient (G-grip_pos, C-cub_pos, T-targ_pos)

        self.last_ct_dist = None # previous distance between cube and target
        self.last_gt_dist = None # previous distance between gripper and target


    def reset(self):
        self.last_points               = None
        self.point_was_reached         = False
    
    def compute(self, observation=None):
        reward = 0
        goal_pos, cube_pos, gripper = self.set_points(observation)
        cube_last_pos, gripper_last = self.last_points if self.last_points != None else [cube_pos, gripper]
        self.set_variables_push(cube_pos, gripper, goal_pos)

        point_grg = self.get_point_grg(goal_pos, cube_pos)

        state = [(gripper, point_grg),(gripper, cube_pos)][int(self.point_was_reached)]
        state_last = [(gripper_last, point_grg),(gripper_last, cube_pos)][int(self.point_was_reached)]
 
        cur_dist = self.get_distance(*state)
        
        if cur_dist <= self.dist_offset:
            self.line_was_reached = True
        last_dist = self.get_distance(*state_last)

        vector1 = [self.x_cube - self.x_target, self.y_cube - self.y_target, self.z_cube - self.z_target]
        vector2 = [self.x_cube - self.x_gripper, self.y_cube - self.y_gripper, self.z_cube - self.z_gripper]

        rew_point = self.exp_eval(cur_dist) if cur_dist < last_dist else self.lin_penalty(last_dist, min_penalty = 1)
        rew_angle, angle= self.angle_reward_push(vector1, vector2)
        rew_dist_ct = self.dist_reward_ct(cube_pos, goal_pos)
        rew_dist_gt = 0
        
        if angle > 170 and cur_dist <= self.dist_offset: 
            #best position -> start moving cube to target
            # reward += 5
            rew_dist_gt = self.dist_reward_gt(gripper, goal_pos) 
            print(rew_dist_gt, angle, cur_dist)
        
        reward = rew_point * self.k_p + rew_angle * self.k_a + rew_dist_ct * self.k_ct + rew_dist_gt * self.k_gt

        # print(reward, rew_point * self.k_p, rew_angle * self.k_a, rew_dist_ct * self.k_ct, rew_dist_gt * self.k_gt)

        self.set_last_positions(cube_pos, gripper)
        self.rewards_history.append(reward)
        self.task.check_goal()

        if self.debug:
            # self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[0, .2, 0, 1])
            # if (self.task.check_object_moved(self.env.task_objects["actual_state"], threshold=0.1)):
            #     print("mooooooooooooveee")
            # XYZ
            # self.env.p.addUserDebugLine([0, 0, -10], [0, 0, 10],
            #                             lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=1) # Z
            # self.env.p.addUserDebugLine([0, -10, 0], [0, 10, 0],
            #                             lineColorRGB=(0, 1, 0), lineWidth=3, lifeTime=1) # Y        
            # self.env.p.addUserDebugLine([-10, 0, 0], [10, 0, 0],
            #                             lineColorRGB=(0, 0, 1), lineWidth=3, lifeTime=1) # X
                
            # self.env.p.addUserDebugLine([-10, 1, 0], [10, 1, 0],
            #                             lineColorRGB=(0, 0, 1), lineWidth=3, lifeTime=1)
            
            # self.env.p.addUserDebugLine([0, 0.46, 0.1], [0, 0.55, 0.1],
            #                             lineColorRGB=(0, 0, 1), lineWidth=3, lifeTime=1)
            
            # self.env.p.addUserDebugLine([-0.5, 0.65, 0.05], [0.5, 0.7, 0.05],
            #                             lineColorRGB=(0, 0, 1), lineWidth=3, lifeTime=1)

            # self.env.p.addUserDebugLine([-0.5, 0.8, 0.05], [0.5, 0.4, 0.05],
            #                             lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=1)
                                                             
            # help lines
            self.env.p.addUserDebugLine([self.x_target, self.y_target, self.z_cube], [self.x_target, self.y_target, 0.5],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1) 
             
            self.env.p.addUserDebugLine([self.x_target, self.y_target, self.z_cube], gripper,
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
            
            self.env.p.addUserDebugLine(cube_pos, gripper,
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
            
            self.env.p.addUserDebugLine(point_grg, gripper,
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)

            self.env.p.addUserDebugLine([self.x_target, self.y_target, self.z_cube], point_grg,
                                        lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
            
            self.env.p.addUserDebugText(f"reward:{reward:.3f}, r_ct:{rew_dist_ct * self.k_ct:.3f}, r_gt:{rew_dist_gt * self.k_gt:.3f}, r_p:{rew_point * self.k_p:.3f},"
                                        f" r_a:{rew_angle * self.k_a:.3f}",
                                        [1, 1, 1], textSize=2.0, lifeTime=0.05, textColorRGB=[0.6, 0.0, 0.6])

        return reward

    def scalar_multiply(self,vector1, vector2):
        return vector1[0]*vector2[0]+vector1[1]*vector2[1]+vector1[2]*vector2[2]
    
    def module(self,vector):
        return math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])
    
    def angle_between_vectors(self,vector1, vector2): 
        return np.arccos(self.scalar_multiply(vector1, vector2)/(self.module(vector1)*self.module(vector2))) * 180 / math.pi
    
    def angle_reward_push(self, vector1, vector2):
        
        reward = 0
        if self.last_angle == None:
            self.last_angle = 0
        angle = self.angle_between_vectors(vector1, vector2)
        reward = (angle - self.last_angle)/10
        if angle > 170:
            reward += angle / 170
        self.last_angle = angle
        
        return reward, angle

    def dist_reward_ct(self, cube_position, target_position):
        if self.last_ct_dist == None:
            self.last_ct_dist = 0

        ct_dist = self.get_distance(cube_position, target_position)
        if round(ct_dist,2) == round(self.last_ct_dist,2):
            reward = 0
        else:
            reward = self.exp_eval(ct_dist) if ct_dist < self.last_ct_dist else self.lin_penalty(self.last_ct_dist, min_penalty = 1)
        self.last_ct_dist = self.get_distance(cube_position, target_position)

        return reward
    
    def dist_reward_gt(self, gripper_position, target_position):
        if self.last_gt_dist == None:
            self.last_gt_dist = 0

        gt_dist = self.get_distance(gripper_position, target_position)
        reward = self.exp_eval(gt_dist) if gt_dist < self.last_gt_dist else self.lin_penalty(self.last_gt_dist, min_penalty = 1)
        
        self.last_gt_dist = self.get_distance(gripper_position, target_position)

        return reward

    def set_variables_push(self, cube_position, gripper_position, target_position):
  
        self.x_cube = cube_position[0] 
        self.y_cube = cube_position[1] 
        self.z_cube = cube_position[2]
 
        self.x_gripper = gripper_position[0] 
        self.y_gripper = gripper_position[1] 
        self.z_gripper = gripper_position[2]
         
        self.x_target = target_position[0] 
        self.y_target = target_position[1] 
        self.z_target = target_position[2]

    def get_positions_push(self, observation):

        target_position = observation["goal_state"]
        cube_position = observation["actual_state"] 
        gripper_position = observation["additional_obs"]["endeff_xyz"] 

        return target_position,cube_position,gripper_position

class PushReward3(PokeReachReward):
    #simple version 1Network
    def __init__(self, env, task):
        super(PushReward3, self).__init__(env, task)

        self.cube_range    = 0.01
        self.gripper_range = 0.01
        self.target_range  = 0.01

    def compute(self, observation=None):
        reward = 0
        target_position, cube_position, gripper_position = self.get_positions_push(observation)
        
        dist_cg = self.task.calc_distance(cube_position, gripper_position)
        dist_ct = self.task.calc_distance(cube_position, target_position)

        approach_reward  = round(10 * self.gripper_range / dist_cg, 3)
        move_reward      = round(10 * self.cube_range / dist_ct, 3)

        reward = approach_reward * move_reward

        # if (self.task.check_object_moved(self.env.task_objects["actual_state"], threshold=1)):
        #     reward = -1
            
        
        self.env.p.addUserDebugText(f"{reward},{approach_reward},{move_reward}", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[1,0,0])

        self.rewards_history.append(reward)
        self.task.check_goal()
        return reward

    def get_positions_push(self, observation):

        target_position = observation["goal_state"]
        cube_position = observation["actual_state"] 
        gripper_position = observation["additional_obs"]["endeff_xyz"] 

        return target_position,cube_position,gripper_position

class PushReward(PokeReachReward):
    def __init__(self, env, task):
        super(PushReward,self).__init__(env, task)
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None 
        
        self.angle        = None
        self.approach     = None
        self.cube_move    = None
        self.gripper_move = None 

        self.last_angle        = None  #previouse angle GCT (G-gripper, C-cube, T-target)
        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.dist_grip_to_cube = None 

        self.k_approach = 1
        self.k_gripper = 1
        self.k_cube = 2
        self.k_angle = 0

    def reset(self):
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None

        self.last_angle = None
    
    #previouse angle GCT (G-gripper, C-cube, T-target)
        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.dist_grip_to_cube = None  

        self.k_approach = 1
        self.k_gripper = 1
        self.k_cube = 2
        self.k_angle = 0

    def angle_between_vectors(self,vector1, vector2): 
        return np.arccos(self.scalar_multiply(vector1, vector2)/(self.module(vector1)*self.module(vector2))) * 180 / math.pi
    
    def scalar_multiply(self,vector1, vector2):
        return vector1[0]*vector2[0]+vector1[1]*vector2[1]+vector1[2]*vector2[2]
    
    def module(self,vector):
        return math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])
    
    def grip_rew(self): 
        if self.last_gripper_move == None:
            self.last_gripper_move = self.gripper_move
        reward = round(self.last_gripper_move - self.gripper_move, 3)   
        self.last_gripper_move = self.gripper_move
        return reward
    
    def cube_rew(self):
        # print("moveRew")
        # if self.last_cube_move == None:
        #     self.last_cube_move = self.cube_move 
        # reward = round(self.last_cube_move - self.cube_move,3)  
        # if reward > 0:
        #     print("cube move")
        #     reward += 1
        # self.last_cube_move = self.cube_move
        reward = 1 / self.cube_move
        return reward

    def angle_rew(self):
        if self.last_angle == None:
            self.last_angle = self.angle 
        reward = round(self.angle - self.last_angle,3) / 10  
        
        self.last_angle = self.angle
        return reward

    def approach_rew(self):
        if self.last_approach == None:
            self.last_approach = self.approach
        reward = round(self.last_approach - self.approach,3)
        self.last_approach = self.approach
        return reward

    def check_cube_position(self, cube_position): # check if cube is on the table
        # table diagonal coordinates
        # -0.75; 0.05
        # 0.75; 1.05
        if cube_position[0] > 0.75 or cube_position[0] < -0.75 or cube_position[1] > 1.05 or cube_position[1] < 0.05:
            return False
        return True

    def set_variables_push(self, target_position,cube_position,gripper_position):
  
        self.x_cube = cube_position[0] 
        self.y_cube = cube_position[1] 
        self.z_cube = cube_position[2]
 
        self.x_gripper = gripper_position[0] 
        self.y_gripper = gripper_position[1] 
        self.z_gripper = gripper_position[2]
         
        self.x_target = target_position[0] 
        self.y_target = target_position[1] 
        self.z_target = target_position[2]

    def get_positions_push(self, observation):

        target_position = observation["goal_state"]
        cube_position = observation["actual_state"] 
        gripper_position = observation["additional_obs"]["endeff_xyz"] 

        return target_position,cube_position,gripper_position
    
    def compute(self, observation=None): 
        target_position, cube_position, gripper_position = self.get_positions_push(observation)
        self.set_variables_push(target_position, cube_position, gripper_position)

        vector1 = [self.x_cube - self.x_target, self.y_cube - self.y_target, self.z_cube - self.z_target]
        vector2 = [self.x_cube - self.x_gripper, self.y_cube - self.y_gripper, self.z_cube - self.z_gripper]
        
        self.angle = round(self.angle_between_vectors(vector1, vector2),2)

        cube_offset_pos = self.get_point_grg(np.array(target_position),np.array(cube_position))

        self.approach = self.task.calc_distance(gripper_position, cube_offset_pos)
        self.cube_move = self.task.calc_distance(target_position, cube_position)
        self.gripper_move = self.task.calc_distance(target_position, gripper_position)
        self.dist_grip_to_cube = self.task.calc_distance(gripper_position, cube_position)
        
        if self.approach > 0.1:
            self.k_gripper = 0
        reward = self.approach_rew() * self.k_approach + self.grip_rew() * self.k_gripper # + self.cube_rew() * self.k_cube + self.angle_rew() * self.k_angle

        if self.angle > 160 and self.dist_grip_to_cube < 0.1:
            print("best position")
            reward += self.angle / 100 + 1 / self.dist_grip_to_cube

        # if self.check_cube_position(cube_position) == False: #in purpose to slow gripper 
        #     reward = -0.5
        #     print("cube is not on the table")

        self.task.check_goal()
        self.rewards_history.append(reward)
        self.env.p.addUserDebugText(f"{self.approach}", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[1,0,0])

        self.env.p.addUserDebugLine(cube_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(target_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(cube_position, target_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        
        self.env.p.addUserDebugLine([self.x_target, self.y_target, self.z_cube], [self.x_target, self.y_target, 0.5],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1) 
        self.env.p.addUserDebugLine([self.x_cube, self.y_cube, self.z_cube], [self.x_cube + 0.1, self.y_cube, self.z_cube],
                                        lineColorRGB=(1, 0.5, 1), lineWidth=3, lifeTime=1) 

        return reward

class ThreeStagePushReward(PokeReachReward):

    def __init__(self, env, task):
        super(ThreeStagePushReward,self).__init__(env, task)
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None 
        
        self.angle        = None 
        self.approach     = None
        self.cube_move    = None
        self.gripper_move = None

        self.last_angle        = None  #previouse angle GCT (G-gripper, C-cube, T-target)
        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.current_network = 0
        self.num_networks = env.num_networks
        self.check_num_networks()
        # self.network_rewards = [0] * self.num_networks 

    def check_num_networks(self):
        assert self.num_networks <= 3, "ThreeStagePushReward reward can work with maximum 3 networks"

    def reset(self):
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None

        self.last_angle        = None  #previouse angle GCT (G-gripper, C-cube, T-target)
        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
    
    def compute(self, observation=None):  
        target_position, cube_position, gripper_position = self.get_positions_push(observation)
        self.set_variables_push(target_position, cube_position, gripper_position)
        
        vector1 = [self.x_cube - self.x_target, self.y_cube - self.y_target, self.z_cube - self.z_target]
        vector2 = [self.x_cube - self.x_gripper, self.y_cube - self.y_gripper, self.z_cube - self.z_gripper]
        
        self.angle = round(self.angle_between_vectors(vector1, vector2),2)

        cube_offset_pos = self.get_point_grg(np.array(target_position),np.array(cube_position))
        self.approach = self.task.calc_distance(gripper_position, cube_offset_pos)

        stage = self.decide(observation)
        target = [[target_position, cube_position, gripper_position],[],[target_position, cube_position ,gripper_position]][stage]

        reward = [self.approach_rew, self.angle_rew, self.move_rew][stage](*target) + stage
        # self.network_rewards
        self.task.check_goal()
        self.rewards_history.append(reward)
        self.env.p.addUserDebugText(f"{reward}, {stage}", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[1,0,0])
        self.env.p.addUserDebugLine(cube_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(target_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(cube_position, target_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        
        return reward
    
    def decide(self, observation=None):
        if self.approach_network():
            self.current_network = 0
        if self.angle_network():
            self.current_network = 1
        if self.move_network():
            self.current_network = 2
        
        return self.current_network
    
    def angle_network(self):
        if self.angle != None and self.approach != None:
            if self.angle < 170 and self.approach <= 0.05:
                return True
        return False
    
    def approach_network(self):
        if self.approach != None:
            if self.approach > 0.05:
                return True
        return False
    
    def move_network(self):
        if self.angle != None and self.approach != None:
            if self.angle >= 170 and self.approach <= 0.05:
                return True
        return False

    def move_rew(self, target_pos, cube_pos, grip_pos):
        # print("moveRew")
        self.cube_move = self.task.calc_distance(target_pos, cube_pos)
        self.gripper_move = self.task.calc_distance(target_pos, grip_pos)
        
        if self.last_cube_move == None:
            self.last_cube_move = self.cube_move
        if self.last_gripper_move == None:
            self.last_gripper_move = self.gripper_move

        reward = self.last_gripper_move - self.gripper_move
        reward += (self.last_cube_move - self.cube_move)*10
        
        self.last_cube_move = self.cube_move
        self.last_gripper_move = self.gripper_move
        return reward

    def approach_rew(self, target_pos, cube_pos, grip_pos):
        # print("approachRew")
        if self.last_approach == None:
            self.last_approach = self.approach
        reward = self.last_approach - self.approach

        self.last_approach = self.approach
        return reward

    def angle_rew(self):
        # print("angleRew")
        if self.last_angle == None:
            self.last_angle = self.angle
        
        reward = (self.angle - self.last_angle)/10
        #if self.angle > 170:
        #    reward += self.angle / 170
        self.last_angle = self.angle
        
        return reward
    
    def angle_between_vectors(self,vector1, vector2): 
        return np.arccos(self.scalar_multiply(vector1, vector2)/(self.module(vector1)*self.module(vector2))) * 180 / math.pi
    
    def scalar_multiply(self,vector1, vector2):
        return vector1[0]*vector2[0]+vector1[1]*vector2[1]+vector1[2]*vector2[2]
    
    def module(self,vector):
        return math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])
    
    def set_variables_push(self, target_position,cube_position,gripper_position):
  
        self.x_cube = cube_position[0] 
        self.y_cube = cube_position[1] 
        self.z_cube = cube_position[2]
 
        self.x_gripper = gripper_position[0] 
        self.y_gripper = gripper_position[1] 
        self.z_gripper = gripper_position[2]
         
        self.x_target = target_position[0] 
        self.y_target = target_position[1] 
        self.z_target = target_position[2]

    def get_positions_push(self, observation):

        target_position = observation["goal_state"]
        cube_position = observation["actual_state"] 
        gripper_position = observation["additional_obs"]["endeff_xyz"] 

        return target_position,cube_position,gripper_position
    
class TwoStagePushReward2(PokeReachReward):

    def __init__(self, env, task):
        super(TwoStagePushReward2,self).__init__(env, task)
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None 
        
        self.approach     = None
        self.cube_move    = None
        self.gripper_move = None

        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.current_network = 0
        self.num_networks = env.num_networks
        self.check_num_networks()
        # self.network_rewards = [0] * self.num_networks 

    def check_num_networks(self):
        assert self.num_networks <= 2, "ThreeStagePushReward reward can work with maximum 3 networks"

    def reset(self):
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None

        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
    
    def compute(self, observation=None):  
        target_position, cube_position, gripper_position = self.get_positions_push(observation)
        self.set_variables_push(target_position, cube_position, gripper_position)

        cube_offset_pos = self.get_point_grg(np.array(target_position),np.array(cube_position))
        self.approach = self.task.calc_distance(gripper_position, cube_offset_pos)

        stage = self.decide(observation)
        target = [[target_position, cube_position, gripper_position],[target_position, cube_position ,gripper_position]][stage]

        reward = [self.approach_rew, self.move_rew][stage](*target) + stage
        # self.network_rewards
        self.task.check_goal()
        self.rewards_history.append(reward)
        self.env.p.addUserDebugText(f"{reward}, {stage}", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[1,0,0])
        self.env.p.addUserDebugLine(cube_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(target_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(cube_position, target_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        
        return reward
    
    def decide(self, observation=None):
        if self.approach != None:
            if self.approach > 0.05:
                self.current_network = 0
            else:
                self.current_network = 1
                self.approach != None
        
        return self.current_network

    def move_rew(self, target_pos, cube_pos, grip_pos):
        # print("moveRew")
        self.cube_move = self.task.calc_distance(target_pos, cube_pos)
        self.gripper_move = self.task.calc_distance(target_pos, grip_pos)
        
        if self.last_cube_move == None:
            self.last_cube_move = self.cube_move
        if self.last_gripper_move == None:
            self.last_gripper_move = self.gripper_move

        reward = self.last_gripper_move - self.gripper_move
        reward += (self.last_cube_move - self.cube_move)*10
        
        self.last_cube_move = self.cube_move
        self.last_gripper_move = self.gripper_move
        return reward

    def approach_rew(self, target_pos, cube_pos, grip_pos):
        # print("approachRew")
        if self.last_approach == None:
            self.last_approach = self.approach
        reward = self.last_approach - self.approach

        self.last_approach = self.approach
        return reward
    
    def set_variables_push(self, target_position,cube_position,gripper_position):
  
        self.x_cube = cube_position[0] 
        self.y_cube = cube_position[1] 
        self.z_cube = cube_position[2]
 
        self.x_gripper = gripper_position[0] 
        self.y_gripper = gripper_position[1] 
        self.z_gripper = gripper_position[2]
         
        self.x_target = target_position[0] 
        self.y_target = target_position[1] 
        self.z_target = target_position[2]

    def get_positions_push(self, observation):

        target_position = observation["goal_state"]
        cube_position = observation["actual_state"] 
        gripper_position = observation["additional_obs"]["endeff_xyz"] 

        return target_position,cube_position,gripper_position

class TwoStagePushReward3(PokeReachReward):

    def __init__(self, env, task):
        super(TwoStagePushReward3,self).__init__(env, task)
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None 
        
        self.angle        = None
        self.approach     = None
        self.cube_move    = None
        self.gripper_move = None

        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.dist_grip_to_cube = None
        self.current_network = 0
        self.num_networks = env.num_networks
        self.check_num_networks()
        # self.network_rewards = [0] * self.num_networks 

    def angle_between_vectors(self,vector1, vector2): 
        return np.arccos(self.scalar_multiply(vector1, vector2)/(self.module(vector1)*self.module(vector2))) * 180 / math.pi
    
    def scalar_multiply(self,vector1, vector2):
        return vector1[0]*vector2[0]+vector1[1]*vector2[1]+vector1[2]*vector2[2]
    
    def module(self,vector):
        return math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])
    
    def move_rew(self, target_pos, cube_pos, grip_pos):
        # print("moveRew")
        if self.last_cube_move == None:
            self.last_cube_move = self.cube_move
        if self.last_gripper_move == None:
            self.last_gripper_move = self.gripper_move

        reward = self.last_gripper_move - self.gripper_move
        reward += (self.last_cube_move - self.cube_move)*10
        
        self.last_cube_move = self.cube_move
        self.last_gripper_move = self.gripper_move
        return reward

    def approach_rew(self, target_pos, cube_pos, grip_pos):
        # print("approachRew")
        if self.last_approach == None:
            self.last_approach = self.approach
        reward = self.last_approach - self.approach

        self.last_approach = self.approach
        return reward
    
    def set_variables_push(self, target_position,cube_position,gripper_position):
  
        self.x_cube = cube_position[0] 
        self.y_cube = cube_position[1] 
        self.z_cube = cube_position[2]
 
        self.x_gripper = gripper_position[0] 
        self.y_gripper = gripper_position[1] 
        self.z_gripper = gripper_position[2]
         
        self.x_target = target_position[0] 
        self.y_target = target_position[1] 
        self.z_target = target_position[2]

    def get_positions_push(self, observation):

        target_position = observation["goal_state"]
        cube_position = observation["actual_state"] 
        gripper_position = observation["additional_obs"]["endeff_xyz"] 

        return target_position,cube_position,gripper_position

    def check_num_networks(self):
        assert self.num_networks <= 2, "ThreeStagePushReward reward can work with maximum 3 networks"

    def reset(self):
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None

        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.dist_grip_to_cube = None
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
    
    def compute(self, observation=None):  
        target_position, cube_position, gripper_position = self.get_positions_push(observation)
        self.set_variables_push(target_position, cube_position, gripper_position)

        vector1 = [self.x_cube - self.x_target, self.y_cube - self.y_target, self.z_cube - self.z_target]
        vector2 = [self.x_cube - self.x_gripper, self.y_cube - self.y_gripper, self.z_cube - self.z_gripper]
        
        self.angle = round(self.angle_between_vectors(vector1, vector2),2)

        cube_offset_pos = self.get_point_grg(np.array(target_position),np.array(cube_position))

        self.approach = self.task.calc_distance(gripper_position, cube_offset_pos)
        self.cube_move = self.task.calc_distance(target_position, cube_position)
        self.gripper_move = self.task.calc_distance(target_position, gripper_position)
        self.dist_grip_to_cube = self.task.calc_distance(gripper_position, cube_position)

        stage = self.decide(observation)
        target = [[target_position, cube_position, gripper_position],[target_position, cube_position ,gripper_position]][stage]

        reward = [self.approach_rew, self.move_rew][stage](*target) + stage

        if self.dist_grip_to_cube != None and self.angle != None:
            if self.dist_grip_to_cube <= 0.05 and self.angle > 160:
                reward += self.angle / 100 - self.dist_grip_to_cube * 10 

        if self.cube_move != None and self.gripper_move != None:
            if self.cube_move <= 0.2 and self.gripper_move <= 0.2:
                reward += 1 - self.cube_move
                
        self.task.check_goal()
        self.rewards_history.append(reward)
        self.env.p.addUserDebugText(f"{reward}, {stage}", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[1,0,0])

        self.env.p.addUserDebugLine(cube_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(target_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(cube_position, target_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        
        self.env.p.addUserDebugLine([self.x_target, self.y_target, self.z_cube], [self.x_target, self.y_target, 0.5],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1) 
        self.env.p.addUserDebugLine([self.x_target, self.y_target, self.z_cube], [self.x_target + 0.1, self.y_target, self.z_cube],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1) 
        return reward
    
    def decide(self, observation=None):
        if self.approach != None:
            if self.approach > 0.05:
                self.current_network = 0
            else:
                self.current_network = 1
        
        return self.current_network

class TwoStagePushReward(PokeReachReward):

    def __init__(self, env, task):
        super(TwoStagePushReward,self).__init__(env, task)
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None 
        
        self.angle        = None
        self.approach     = None
        self.cube_move    = None
        self.gripper_move = None
        self.proj_length = None

        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.dist_grip_to_cube = None

        self.last_proj_length = None

        self.current_network = 0
        self.num_networks = env.num_networks
        self.check_num_networks()
        # self.network_rewards = [0] * self.num_networks 

    def angle_between_vectors(self,vector1, vector2): 
        return np.arccos(self.scalar_multiply(vector1, vector2)/(self.module(vector1)*self.module(vector2))) * 180 / math.pi
    
    def scalar_multiply(self,vector1, vector2):
        return vector1[0]*vector2[0]+vector1[1]*vector2[1]+vector1[2]*vector2[2]
    
    def module(self,vector):
        return math.sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2])
    
    def move_rew(self, target_pos, cube_pos, grip_pos):  
        # print("moveRew") 
        if self.last_gripper_move == None:
            self.last_gripper_move = self.gripper_move 
        reward = self.last_gripper_move - self.gripper_move   
        self.last_gripper_move = self.gripper_move

        if self.last_cube_move == None:
            self.last_cube_move = self.cube_move
        # cube_rew = self.last_cube_move - self.cube_move
        # if cube_rew > 0:
        #     reward += cube_rew * 10
        #     print("cube_rew = ", cube_rew)
        self.last_cube_move = self.cube_move
        return reward

    def approach_rew(self, target_pos, cube_pos, grip_pos):
        # print("approachRew")
        if self.last_approach == None:
            self.last_approach = self.approach
        reward = self.last_approach - self.approach 
        self.last_approach = self.approach
        return reward

    def dist_to_line(self):
        x0 = self.x_target
        y0 = self.y_target
        z0 = self.z_target

        x1 = self.x_gripper
        y1 = self.y_gripper
        z1 = self.z_gripper

        x2 = self.x_cube
        y2 = self.y_cube
        z2 = self.z_cube 

        point_on_line = np.array([x1,y1,z1])
        line_direction = np.array([x2-x1,y2-y1,z2-z1])
        point = np.array([x0,y0,z0])

        projection = np.dot(point - point_on_line, line_direction) / np.dot(line_direction, line_direction)
        closest_point = point_on_line + projection * line_direction
        dist = np.linalg.norm(point - closest_point)

        return dist
    
    def check_cube_position(self, cube_position): # check if cube is not on the table
        # table diagonal coordinates
        # -0.75; 0.05
        # 0.75; 1.05
        if cube_position[0] > 0.75 or cube_position[0] < -0.75 or cube_position[1] > 1.05 or cube_position[1] < 0.05:
            print("reset")
            self.env.episode_failed = True
            return True
        return False

    def set_variables_push(self, target_position,cube_position,gripper_position):
  
        self.x_cube = cube_position[0] 
        self.y_cube = cube_position[1] 
        self.z_cube = cube_position[2]
 
        self.x_gripper = gripper_position[0] 
        self.y_gripper = gripper_position[1] 
        self.z_gripper = gripper_position[2]
         
        self.x_target = target_position[0] 
        self.y_target = target_position[1] 
        self.z_target = target_position[2]

    def get_positions_push(self, observation):

        target_position = observation["goal_state"]
        cube_position = observation["actual_state"] 
        gripper_position = observation["additional_obs"]["endeff_xyz"] 

        return target_position,cube_position,gripper_position

    def check_num_networks(self):
        assert self.num_networks <= 2, "TwoStagePushReward reward can work with maximum 2 networks"

    def reset(self):
        self.x_cube = None
        self.y_cube = None
        self.z_cube = None

        self.x_gripper = None
        self.y_gripper = None
        self.z_gripper = None

        self.x_target = None
        self.y_target = None
        self.z_target = None

        self.last_approach     = None  #previouse gripper distance to cube
        self.last_cube_move    = None  #previouse cube distance to target
        self.last_gripper_move = None  #previouse gripper position to target
        self.last_proj_length = None

        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
    
    def compute(self, observation=None):  
        target_position, cube_position, gripper_position = self.get_positions_push(observation)
        self.set_variables_push(target_position, cube_position, gripper_position) 
        
        self.env.episode_over = self.check_cube_position(cube_position)
        vector1 = [self.x_cube - self.x_target, self.y_cube - self.y_target, self.z_cube - self.z_target]
        vector2 = [self.x_cube - self.x_gripper, self.y_cube - self.y_gripper, self.z_cube - self.z_gripper]
        
        self.angle = round(self.angle_between_vectors(vector1, vector2),2)
        self.proj_length = self.dist_to_line()  

        cube_offset_pos = self.get_point_grg(np.array(target_position),np.array(cube_position))

        self.approach = self.task.calc_distance(gripper_position, cube_offset_pos)
        self.cube_move = self.task.calc_distance(target_position, cube_position)
        self.gripper_move = self.task.calc_distance(target_position, gripper_position)
        self.dist_grip_to_cube = self.task.calc_distance(gripper_position, cube_position)
        
        stage = self.decide(observation)
        target = [[target_position, cube_position, gripper_position],[target_position, cube_position ,gripper_position]][stage]

        reward = [self.approach_rew, self.move_rew][stage](*target) + stage
        
        # if self.check_cube_position(cube_position) == False: #in purpose to slow gripper 
        #     reward = -1
        #     print("cube is not on the table reward = ", reward)
             
        self.task.check_goal()
        self.rewards_history.append(reward)
        self.env.p.addUserDebugText(f" {stage},{round(self.angle,1)}, {round(self.proj_length,3)}, {round(self.dist_grip_to_cube, 3)}", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[1,0,0])

        self.env.p.addUserDebugLine(cube_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(target_position, gripper_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        self.env.p.addUserDebugLine(cube_position, target_position, lineColorRGB=(1, 0, 0), lineWidth=3, lifeTime=0.05)
        
        self.env.p.addUserDebugLine([self.x_target, self.y_target, self.z_cube], [self.x_target, self.y_target, 0.5],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1) 
        self.env.p.addUserDebugLine([self.x_target, self.y_target, self.z_cube], [self.x_target + 0.1, self.y_target, self.z_cube],
                                        lineColorRGB=(0, 0.5, 1), lineWidth=3, lifeTime=1) 
        return reward
    
    def decide(self, observation=None):

        if self.angle != None and self.proj_length != None and self.dist_grip_to_cube != None:
            if self.angle < 135 or self.proj_length > 0.03 or self.dist_grip_to_cube > 0.11:
                self.current_network = 0
            else:
                self.current_network = 1
        else:
            self.current_network = 0
        
        return self.current_network

class DualPoke(PokeReachReward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(DualPoke, self).__init__(env, task)
        self.prev_poker_position = [None]*3
        self.touched = False
        self.last_aimer_dist = 0
        self.last_poker_dist = 0

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_poker_position = [None]*3
        self.touched = False
        self.last_aimer_dist = 0
        self.last_poker_dist = 0
        self.network_rewards = [0,0]
        self.current_network = 0

    def is_poker_moving(self, poker):
        if not any(self.prev_poker_position):
            return False
        if sum([poker[i] - self.prev_poker_position[i] for i in range(len(poker))]) < 0.01:
            return False
        else:
            self.touched = True
            return True

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.
        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        self.network_switch_control(observation)
        reward = [self.aimer_compute(observation), self.poker_compute(observation)][self.current_network]
        #self.task.check_distance_threshold(observation, threshold=0.05)
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def aimer_compute(self, observation=None):
        self.env.p.addUserDebugText("XXX", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,125,0])
        goal_position, poker_position, gripper_position = self.get_positions(observation)
        aim_point = self.get_aim_point(goal_position, poker_position)
        aimer_dist = self.task.calc_distance(gripper_position, aim_point)
        if self.current_network != 0 or self.last_aimer_dist is None:
            self.last_aimer_dist = aimer_dist
        reward = self.last_aimer_dist - aimer_dist
        self.network_rewards[0] += reward
        self.env.p.addUserDebugText(str(self.network_rewards[0]), [0.5,0.5,0.5], lifeTime=0.1, textColorRGB=[0,125,0])
        self.last_aimer_dist = aimer_dist
        if self.env.episode_steps > 25:
            self.touched = self.is_poker_moving(poker_position)
        self.prev_poker_position = poker_position
        return reward

    def poker_compute(self, observation=None):
        self.env.p.addUserDebugText("XXX", [-0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        goal_position, poker_position, gripper_position = self.get_positions(observation)
        poker_dist = round(self.task.calc_distance(poker_position, goal_position), 5)
        if self.current_network != 1 or self.last_poker_dist is None:
            self.last_poker_dist = poker_dist
        reward = 5*(self.last_poker_dist - poker_dist)
        if self.task.check_object_moved(self.env.task_objects["actual_state"], 1): # if pushes goal too far
            self.env.episode_over = True
            self.env.episode_failed = True
            self.env.episode_info = "too strong poke"
            reward = -5
        self.network_rewards[1] += reward

        self.env.p.addUserDebugText(str(self.network_rewards[1]), [-0.5,0.5,0.5], lifeTime=0.1, textColorRGB=[0,0,125])
        self.last_poker_dist = poker_dist
        self.prev_poker_position = poker_position
        if self.touched and not self.is_poker_moving(observation["actual_state"]):
            self.env.episode_over = True
            if self.last_poker_dist > 0.1:
                self.env.episode_failed = True
                self.env.episode_info = "bad poke"
            else:
                self.env.episode_info = "good poke"
        return reward

    def get_aim_point(self, goal_position, poker_position):
        aim_vector = Vector(poker_position, goal_position, self.env.p)
        aim_vector.set_len(-0.2)
        aim_point = poker_position + aim_vector.vector
        return aim_point

    def get_positions(self, observation):
        goal_position = observation["goal_state"]
        poker_position = observation["actual_state"]
        gripper_name = [x for x in self.env.task.obs_template["additional_obs"] if "endeff" in x][0]
        gripper_position = observation["additional_obs"][gripper_name][:3]
        if self.prev_poker_position[0] is None:
            self.prev_poker_position = poker_position
        return goal_position,poker_position,gripper_position

    def decide(self, observation=None):
        goal_position, poker_position, gripper_position = self.get_positions(observation)
        is_aimed = self.is_aimed(goal_position, poker_position, gripper_position)
        if self.touched:
            self.env.p.addUserDebugText("touched", [-0.3,0.3,0.3], lifeTime=0.1, textColorRGB=[0,0,125])
        if is_aimed or self.touched:
            owner = 1  # poke
        elif not is_aimed and not self.touched:
            owner = 0  # aim
        return owner

    def did_touch(self):
        contact_points = [self.env.p.getContactPoints(self.env.robot.robot_uid, self.env.env_objects[1].uid, x, -1) for x in range(0,self.env.robot.end_effector_index+1)]
        for point in contact_points:
            if len(point) > 0:
                return True
        return False

    def is_aimed(self, goal_position, poker_position, gripper_position):
        poke_vector = Vector(poker_position, goal_position, self.env.p)
        poke_vector.set_len(-0.2)
        gripper_in_XY = poker_position + 3*poke_vector.vector
        len = self.distance_of_point_from_abscissa(gripper_in_XY, poker_position, gripper_position)
        if len < 0.1:
            return True
        return False

    def triangle_height(self, a, b, c):
        p = (a+b+c)/2
        v = math.sqrt(p*(p-a)*(p-b)*(p-c))
        return (2/a)*v

    def distance_of_point_from_abscissa(self, A, B, point):
        a = self.task.calc_distance(A, B)
        b = self.task.calc_distance(A, point)
        c = self.task.calc_distance(point, B)
        height_a = self.triangle_height(a, b, c)
        distance = height_a
        if b > a or c > a:
            distance = c
        return distance

# pick and place rewards
class SingleStagePnP(DistanceReward):
    """
    Pick and place with simple Distance reward. The gripper is operated automatically.
    Applicable for 1 network.

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(SingleStagePnP, self).__init__(env, task)
        self.before_pick = True

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
        if self.gripper_reached_object(observation):
            self.before_pick = False
        reward = self.calc_dist_diff(o1, o2)
        if self.task.calc_distance(o1, o2) < 0.1:
            self.env.robot.release_all_objects()
            self.env.episode_over = True
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def gripper_reached_object(self, observation):
        gripper_name = [x for x in self.env.task.obs_template["additional_obs"] if "endeff" in x][0]
        gripper = observation["additional_obs"][gripper_name][:3]
        object = observation["actual_state"]
        self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        if self.before_pick:
            self.env.robot.magnetize_object(self.env.env_objects["actual_state"])
        if self.env.env_objects["actual_state"] in self.env.robot.magnetized_objects.keys():
            self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[0, 255, 0, 1])
            return True
        return False

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_obj1_position = None
        self.prev_obj2_position = None
        self.before_pick = True

class TwoStagePnP(DualPoke):
    """
    Pick and place with two rewarded stages - find and move, gripper is operated automatically.
    Applicable for 1 or 2 networks.

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(TwoStagePnP, self).__init__(env, task)
        self.last_owner = None
        self.last_find_dist  = None
        self.last_lift_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.current_network = 0
        self.num_networks = env.num_networks
        self.check_num_networks()
        self.network_rewards = [0] * self.num_networks

    def check_num_networks(self):
        assert self.num_networks <= 2, "TwosStagePnP reward can work with maximum 2 networks"

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.last_owner = None
        self.last_find_dist  = None
        self.last_lift_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks


    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.
        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if owner   == 0: reward = self.find_compute(gripper_position, object_position)
        elif owner == 1: reward = self.move_compute(object_position, goal_position)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network= 1
        return self.current_network

    def gripper_reached_object(self, gripper, object):
        #self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        #if self.current_network == 0:
        #    self.env.robot.magnetize_object(self.env.env_objects["actual_state"])
        if "gripper" in self.env.robot_action:
            if self.task.calc_distance(gripper, object) <= 0.05:
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[0, .2, 0, 1])
                return True
            else:
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[255, 0, 0, 1])
                return False
        else:
            if self.env.env_objects["actual_state"] in self.env.robot.magnetized_objects.keys():
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[0, 1, 0, 1])
                return True
        return False


    def find_compute(self, gripper, object):
        # initial reach
        self.env.p.addUserDebugText("Subgoal: Pick", [.65, 1., 0.5], lifeTime=0.1, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper, object)
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.1, textColorRGB=[0, 125, 0])
        if self.last_find_dist is None:
            self.last_find_dist = dist
        if self.last_owner != 0:
            self.last_find_dist = dist
        reward = self.last_find_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.65,1,0.65], lifeTime=0.1, textColorRGB=[0,125,0])
        self.last_find_dist = dist
        if self.task.check_object_moved(self.env.task_objects["actual_state"], threshold=1.2):
            self.env.episode_over   = True
            self.env.episode_failed = True
        self.network_rewards[0] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,1,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        return reward

    def move_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("move", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,125,0])
        dist = self.task.calc_distance(object, goal)
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.5,0.5,0.7], lifeTime=0.1, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None:
            self.last_place_dist = dist
        if self.last_owner != 1:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist
        reward = reward * 10
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.7,0.7,1.0], lifeTime=0.1, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        if self.last_owner == 1 and dist < 0.1:
            self.env.robot.release_all_objects()
            self.gripped = None
            self.env.episode_info = "Object was placed to desired position"
        if self.env.episode_steps <= 2:
            self.env.episode_info = "Task finished in initial configuration"
            self.env.robot.release_all_objects()
            self.env.episode_over = True
        ix = 1 if self.num_networks > 1 else 0
        self.network_rewards[ix] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[ix]}", [0.65,1,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        return reward

class ThreeStagePnP(TwoStagePnP):
    """
    Pick and place with three rewarded stages - find, move and place, gripper is operated automatically.
    Applicable for 1, 2 or 3 networks.

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(ThreeStagePnP, self).__init__(env, task)
        self.was_above = False
        self.last_traj_idx = 0

    def check_num_networks(self):
        assert self.num_networks == 3, "ThreeStagePnP reward can work with maximum 3 networks"

    def reset(self):
        super(ThreeStagePnP, self).reset()
        self.was_above = False

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, goal_position], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.place_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_above_goal(object_position, goal_position) or self.was_above:
            self.current_network = 2
            self.was_above = True
        self.env.p.addUserDebugText(f"Network:{self.current_network}", [0.7,0.7,1.0], lifeTime=0.1, textColorRGB=[55,125,0])
        return self.current_network

    def move_compute(self, object, goal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("Subgoal:move", [.65, 1., 0.5], lifeTime=0.1, textColorRGB=[0,0,125])
        object_XY = object[:3]
        goal_XY   = [goal[0], goal[1], goal[2]+0.2]
        self.env.p.addUserDebugLine(object_XY, goal_XY, lifeTime=0.1)
        dist = self.task.calc_distance(object_XY, goal_XY)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.1, textColorRGB=[0, 125, 0])
        if self.last_move_dist is None or self.last_owner != 1:
           self.last_move_dist = dist
        reward = self.last_move_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.7,0.7,1.2], lifeTime=0.1, textColorRGB=[0,125,0])
        self.last_move_dist = dist
        ix = 1 if self.num_networks > 1 else 0
        self.network_rewards[ix] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[ix]}", [0.65,1,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        return reward


    def place_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("Subgoal: place", [.65, 1., 0.5], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        dist = self.task.calc_distance(object, goal)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.1, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None or self.last_owner != 2:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist
        reward = reward * 10
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.7,0.7,1.2], lifeTime=0.1, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        
        if self.last_owner == 2 and dist < 0.1:
            self.env.robot.release_all_objects()
            self.gripped = None
            self.env.episode_info = "Object was placed to desired position"
            #if self.task.number_tasks == self.task.current_task + 1:
            #    self.env.episode_over = True
        if self.env.episode_steps <= 2:
            self.env.episode_info = "Task finished in initial configuration"
            self.env.episode_over = True
        self.network_rewards[-1] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        return reward

    def gripper_reached_object(self, gripper, object):
        #self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        #if self.current_network == 0:
        #    self.env.robot.magnetize_object(self.env.env_objects["actual_state"])
        if "gripper" in self.env.robot_action:
            if self.task.calc_distance(gripper, object) <= 0.1:
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[200, 10, 0, .7])
                return True
            else:
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[255, 0, 0, 1])
                return False
        else:
            if self.env.env_objects["actual_state"] in self.env.robot.magnetized_objects.keys():
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[200, 10, 0, .7])
                return True
        return False

    def object_lifted(self, object, object_before_lift):
        lifted_position = [object_before_lift[0], object_before_lift[1], object_before_lift[2]+0.1] # position of object before lifting but hightened with its height
        self.task.calc_distance(object, lifted_position)
        if object[2] < 0.079:
            self.lifted = False # object has fallen
            self.object_before_lift = object
        else:
            self.lifted = True
            return True
        return False

    def object_above_goal(self, object, goal):
        goal_XY   = [goal[0], goal[1], goal[2]+0.2]
        object_XY = object
        distance  = self.task.calc_distance(goal_XY, object_XY)
        if distance < 0.1:
            return True
        return False

class DropReward2Stage(PokeReachReward):
    
    def __init__(self, env, task):
        super(PokeReachReward, self).__init__(env, task)
        self.dist_offset               = 0.05

        self.last_points               = None 
        self.was_dropped               = False
        self.point_was_reached         = False
        self.touching                  = None
        self.was_thrown                = False
        self.wrong_drop                = False
        self.throw_penalty             = 0
        self.drop_penalty              = 0
        self.point_above_target        = None
        self.up                        = 0.25
        self.right_place               = False
        self.drop_episode              = None
        self.get_bonus                 = False
    
    def reset(self):
        self.env.task.current_task     = 0
        self.last_points               = None
        self.touching                  = None
        self.was_thrown                = False
        self.wrong_drop                = False
        self.right_place               = False
        if self.env.episode_steps <= 1:
            self.drop_episode              = None
            self.point_above_target        = None
            self.was_dropped               = False
            self.point_was_reached         = False
            self.get_bonus                 = False

    def compute(self, observation=None):
        owner = self.decide(observation)
        # print("drop episode", self.drop_episode, "owner", owner)

        # print(self.env.task.current_task, self.env.episode_steps)
        goal_pos, obj_pos, robot = self.set_points(observation)
        obj_last_pos, robot_last = self.last_points if self.last_points != None else [obj_pos, robot]
        if self.env.episode_steps == 1 and "gripper" not in self.env.robot_action:
            self.point_above_target  = goal_pos + np.array([0., 0., self.up]) 
            self.env.robot.magnetize_object(self.env.task_objects["actual_state"])
        
        self.env.p.addUserDebugLine(robot, [1,1,1], lineColorRGB=(255, 255, 255), lineWidth = 1, lifeTime = 0.1)

        self.touching = observation["additional_obs"]["touch"][0]

        owner = self.decide(observation)     

        if owner == 0:
            # print(1)
            reward = self.reward_move_object([obj_pos, obj_last_pos], self.point_above_target)
        elif "gripper" in self.env.robot_action:
            # print(2)
            reward = self.drop_reward([robot, robot_last],
                                      [obj_pos, obj_last_pos],
                                      goal_pos)
        else:
            reward = self.reward_move_object([robot, robot_last], self.point_above_target) + 1
            self.env.p.addUserDebugLine(robot, self.point_above_target, lineColorRGB=(255, 255, 255), lineWidth = 1, lifeTime = 0.1)
            # print(3)

        self.check_task([robot, robot_last],[obj_pos, obj_last_pos], goal_pos)
        
        self.task.check_goal()
        reward -= self.penalty_failed()
        if self.get_bonus:
            reward += 1000
            self.get_bonus = False
        self.rewards_history.append(reward)
        # print(reward, end = " ")
        self.set_last_positions(obj_pos, robot)
        if self.was_dropped and self.drop_episode == None and self.env.env_objects["actual_state"] not in self.env.robot.magnetized_objects.keys():
            self.drop_episode = self.env.episode_steps
        # print(reward)
        self.env.p.addUserDebugText(f"reward: {reward}", [0.5, 0.5, 0.5], textSize=2.0, lifeTime=0.05, textColorRGB=[0.6, 0.0, 0.6])
        # print("reward",r/eward)
        return reward
    

    def reward_move_object(self, obj_pos: list, goal: 'numpy.ndarray') -> float:
        """
        Params:
            :param obj_pos: list of two arrays, current possition and last possition of object.
            :param goal: (array), the point where we want to place an object.
        Returns:
            :return reward: (float), reward of moving an object to given point.
        """
        reward = 0.
        cur_dist = self.get_distance(obj_pos[0], goal)
        last_dist= self.get_distance(obj_pos[1], goal)
        if last_dist < cur_dist and False:
            reward = self.lin_penalty(cur_dist, min_penalty = -6)
        else:
            # reward = self.exp_eval(cur_dist, max_r = 5)
            # reward = self.lin_eval(cur_dist)
            reward = (last_dist - cur_dist) / last_dist
        return reward

    def is_free_falling(self, cur_pos: 'numpy.ndarray', last_pos: 'numpy.ndarray', threshold: float = 45.) -> bool:
        """
        Params:
            :param cur_pos: (array) current possition of object,
            :param last_pos: (array) last possition of object.
        Returns:
            :return reward: (bool) True if object is free falling else False.
        """
        angle = self.get_angle_2vec(np.array([0, 0, -1]), cur_pos - last_pos)
        if fabs(angle) < threshold and cur_pos[-1] < last_pos[-1] and not self.touching:
            return True
        else:
            return False

    def check_task(self, robot: list, obj_pos: list, target_pos: 'numpy.ndarray'):
        """
        Params:
            :param obj_pos: list of two arrays, current possition and last possition of object.
            :param target_pos: list 
        Controlls if task was compleated correctly.
        Contrlolls:
        - if object was dropped to target,
        - if wasn't thrown.
        """
        # assert "gripper" in self.env.robot_action, "This task or reward require to use gripper action (ex absolute_gripper)!"
    
        # if sqrt(sum((target_pos[:-1]-obj_pos[0][:-1])**2)) < 0.2 and obj_pos[0][-1] < 0.07:
        # print(obj_pos)
        # if obj_pos[0][-1] < 0.07  and self.env.episode_steps > 30:
        #     print("yahoooooo")
        # print(self.drop_episode)
        if self.drop_episode != None and self.env.episode_steps > self.drop_episode + 20:
            self.right_place = True
        self.is_free_falling(obj_pos[0], obj_pos[1])
        if not self.touching and self.get_distance(obj_pos[0], robot[0]) > 0.05:
            if self.is_free_falling(obj_pos[0], obj_pos[1]):
                if sqrt(sum((target_pos[:-1] - obj_pos[0][:-1])**2)) >= 0.5:
                    self.wrong_drop = True
            else:
                self.was_thrown = True

    def is_failed(self):
        if self.was_thrown:
            self.env.episode_info   = "Object was thrown!"
            self.env.episode_over   = True
            self.env.episode_failed = True
        if self.wrong_drop:
            self.env.episode_info   = "Object was dropped in wrong place!"
            self.env.episode_over   = True
            self.env.episode_failed = True

    def penalty_failed(self) -> float:
        penalty = 0.
        if self.was_thrown:
            penalty = -self.throw_penalty * ((512 - self.env.episode_steps)/512)
        if self.wrong_drop:
            penalty = -self.drop_penalty * ((512 - self.env.episode_steps)/512)
        return penalty

    def drop_reward(self, robot_pos: list, obj_pos: list, goal_pos: 'numpy.ndarray') -> float:
        """
        Params:
            :param robot_pos: list of two arrays, current possition and last possition of robot.
            :param obj_pos: list of two arrays, current possition and last possition of object.
            :param goal_pos: array of goal possition 
        Returns:
            :return reward: (float) reward for dropping
        """
        
        reward = 0.
        reward += 0.1 * self.reward_move_object(robot_pos, goal_pos)
        if self.touching:
            reward += 0.5 * self.reward_move_object(obj_pos, goal_pos)
            reward -= 10
        else:
            reward += 100 
        return reward
    
    def decide(self, observation=None) -> int:
        """
        Params:
            :param observation: (list) Observation of the environment.
        Returns:
            :return reward: (int) owner.
        """
        owner = 0
        goal_pos, cube_pos, robot = self.set_points(observation)
        point_above_target  = goal_pos + np.array([0., 0., self.up])
        if self.get_distance(point_above_target, cube_pos) <= 0.1 or self.point_was_reached:
            owner = 1
            if not self.point_was_reached:
                self.get_bonus = True
            self.point_was_reached = True
        return owner

    def get_angle_2vec(self, vec1: "numpy.ndarray", vec2: "numpy.ndarray") -> float:
        """
        Params:
            :param vec1: array, first vector
            :param vec2: array, second vector
        Returns:
            :return angle: angle (degree: from -180 to 180) between two givven vectors
        """
        scal_prod = sum(vec1*vec2)
        leng_prod = sqrt(sum(vec1**2)) * sqrt(sum(vec2**2))
        if scal_prod == 0 and leng_prod == 0:
            return 0. 
        return acos(scal_prod/leng_prod)*180/pi


class ThreeStagePnPRot(ThreeStagePnP):

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.last_owner = None
        self.last_find_dist  = None
        self.last_lift_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.last_rot_dist = None
        self.was_near = False
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
    
    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, goal_position], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, goal_position) or self.was_near:
            self.current_network = 2
            self.was_near = True
        #self.env.p.addUserDebugText(f"Network:{self.current_network}", [0.7,0.7,1.0], lifeTime=0.1, textColorRGB=[55,125,0])
        return self.current_network
    
    def find_compute(self, gripper, object):
        # initial reach
        self.env.p.addUserDebugText("find object", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper[:3], object[:3])
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_find_dist is None:
            self.last_find_dist = dist
        if self.last_owner != 0:
            self.last_find_dist = dist
        reward = self.last_find_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_find_dist = dist
        #if self.task.check_object_moved(self.env.task_objects["actual_state"], threshold=1.2):
        #    self.env.episode_over   = True
        #    self.env.episode_failed = True
        self.network_rewards[0] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    def move_compute(self, object, goal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("move", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[0,0,125])
        object_XY = object[:3]
        goal_XY   = goal[:3]
        self.env.p.addUserDebugLine(object_XY, goal_XY, lifeTime=0.1)
        dist = self.task.calc_distance(object_XY, goal_XY)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_move_dist is None or self.last_owner != 1:
           self.last_move_dist = dist
        reward = self.last_move_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_move_dist = dist
        ix = 1 if self.num_networks > 1 else 0
        self.network_rewards[ix] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[ix]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    
    def rotate_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("rotate", [0.63,1,0.5], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        dist = self.task.calc_distance(object, goal)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None or self.last_owner != 2:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist

        rot = self.task.calc_rot_quat(object, goal)
        self.env.p.addUserDebugText("Rotation: {}".format(round(rot,3)), [0.65,1,0.6], lifeTime=0.5, textColorRGB=[0, 222, 100])
        if self.last_rot_dist is None or self.last_owner != 2:
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        reward = reward + rewardrot

        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        self.last_rot_dist = rot
        #if self.last_owner == 2 and dist < 0.1:
        #    self.env.robot.release_all_objects()
        #    self.gripped = None
        #    self.env.episode_info = "Object was placed to desired position"
        #    if self.task.number_tasks == self.task.current_task + 1:
        #        self.env.episode_over = True
        #if self.env.episode_steps <= 2:
        #    self.env.episode_info = "Task finished in initial configuration"
        #    self.env.episode_over = True
        self.network_rewards[-1] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward

    def object_near_goal(self, object, goal):
        distance  = self.task.calc_distance(goal, object)
        if distance < 0.1:
            return True
        return False
    
class ThreeStageSwipeRot(ThreeStagePnP):

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.last_owner = None
        self.last_find_dist  = None
        self.last_lift_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.last_rot_dist = None
        self.was_near = False
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
    
    def subgoal_offset(self, goal_position):
        
        offset=[0.2,0.0,0.0]
        subgoal = [goal_position[0]-offset[0],goal_position[1]-offset[1],goal_position[2]-offset[2],goal_position[3],goal_position[4],goal_position[5],goal_position[6]]
        return subgoal

    
    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        pregoal = self.subgoal_offset(goal_position)
        target = [[gripper_position,object_position], [object_position, pregoal], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        pregoal = self.subgoal_offset(goal_position)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_ready_swipe(object_position, pregoal) or self.was_near:
            self.current_network = 2
            self.was_near = True
        #self.env.p.addUserDebugText(f"Network:{self.current_network}", [0.7,0.7,1.0], lifeTime=0.1, textColorRGB=[55,125,0])
        return self.current_network
    
    def find_compute(self, gripper, object):
        # initial reach
        self.env.p.addUserDebugText("find sponge", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper[:3], object[:3])
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.59,1,0.65], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_find_dist is None:
            self.last_find_dist = dist
        if self.last_owner != 0:
            self.last_find_dist = dist
        reward = self.last_find_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_find_dist = dist
        #if self.task.check_object_moved(self.env.task_objects["actual_state"], threshold=1.2):
        #    self.env.episode_over   = True
        #    self.env.episode_failed = True
        self.network_rewards[0] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.59,1,0.6], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    def move_compute(self, object, pregoal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("prepare swipe", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[0,0,125])
        self.env.p.addUserDebugLine(object[:3], pregoal[:3], lifeTime=0.1)
        
        dist = self.task.calc_distance(object, pregoal)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.59,1,0.65], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None or self.last_owner != 1:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist

        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        ix = 1 if self.num_networks > 1 else 0
        self.network_rewards[ix] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[ix]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    
    def rotate_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("swiping", [0.63,1,0.5], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        
        dist = self.task.calc_distance(object, goal)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.59,1,0.65], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None or self.last_owner != 2:
            self.last_place_dist = dist
        rewarddist = self.last_place_dist - dist

        rot = self.task.calc_rot_quat(object, goal)
        self.env.p.addUserDebugText("Rotation: {}".format(round(rot,3)), [0.57,1,0.7], lifeTime=0.5, textColorRGB=[0, 222, 100])
        if self.last_rot_dist is None or self.last_owner != 2:
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        
        z_difference = self.task.calc_height_diff (object, goal)
        self.env.p.addUserDebugText("Heightdiff: {}".format(round(z_difference,3)), [0.55,1,0.73], lifeTime=0.5, textColorRGB=[45, 222, 100])
        rewardheight = 0.05 - z_difference
        reward = rewarddist + rewardrot + rewardheight

        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        self.last_rot_dist = rot
        #if self.last_owner == 2 and dist < 0.1:
        #    self.env.robot.release_all_objects()
        #    self.gripped = None
        #    self.env.episode_info = "Object was placed to desired position"
        #    if self.task.number_tasks == self.task.current_task + 1:
        #        self.env.episode_over = True
        #if self.env.episode_steps <= 2:
        #    self.env.episode_info = "Task finished in initial configuration"
        #    self.env.episode_over = True
        self.network_rewards[2] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward

    def object_ready_swipe(self, object, goal):
        distance  = self.task.calc_distance(goal[:3], object[:3])
        if distance < 0.1:
            return True
        return False

class ThreeStageSwipe(ThreeStageSwipeRot):

    def object_ready_swipe(self, object, goal):
        distance  = self.task.calc_distance(goal[:3], object[:3])
        rot = self.task.calc_rot_quat(object, goal)
        if distance < 0.1 and rot < 0.1:
            return True
        return False
    
    def move_compute(self, object, pregoal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("prepare swipe", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[0,0,125])
        self.env.p.addUserDebugLine(object[:3], pregoal[:3], lifeTime=0.1)
        
        dist = self.task.calc_distance(object, pregoal)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.59,1,0.65], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None or self.last_owner != 1:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist

        rot = self.task.calc_rot_quat(object, pregoal)
        self.env.p.addUserDebugText("Rotation: {}".format(round(rot,3)), [0.57,1,0.7], lifeTime=0.5, textColorRGB=[0, 222, 100])
        if self.last_rot_dist is None or self.last_owner != 1:
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        
        reward = reward + rewardrot

        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        self.last_rot_dist = rot
        ix = 1 if self.num_networks > 1 else 0
        self.network_rewards[ix] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[ix]}", [0.59,1,0.6], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward


class FourStagePnP(ThreeStagePnP):

    def check_num_networks(self):
        assert self.num_networks <= 4, "FourStagePnP reward can work with maximum 4 networks"
    
    def above_compute(self, object, goal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("move", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        object_XY = object
        goal_XY   = [goal[0], goal[1], goal[2]+0.2]
        self.env.p.addUserDebugLine(object_XY, goal_XY, lifeTime=0.1)
        dist = self.task.calc_distance(object_XY, goal_XY)
        if self.last_move_dist is None: #or self.last_owner != 1:
           self.last_move_dist = dist
        reward = self.last_move_dist - dist
        self.last_move_dist = dist
        #ix = 1 if self.num_networks > 1 else 0
        self.network_rewards[0] += reward
        return reward

    def find_compute(self, gripper, object):
        # initial reach
        self.env.p.addUserDebugText("find object", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper, object)
        self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        if self.last_find_dist is None:
            self.last_find_dist = dist
        #if self.last_owner != 0:
        #    self.last_find_dist = dist
        reward = self.last_find_dist - dist
        self.last_find_dist = dist
        if self.task.check_object_moved(self.env.task_objects["actual_state"], threshold=1.2):
            self.env.episode_over   = True
            self.env.episode_failed = True
        self.network_rewards[1] += reward
        return reward
    
    def move_compute(self, object, goal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("move", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        object_XY = object
        goal_XY   = [goal[0], goal[1], goal[2]+0.2]
        self.env.p.addUserDebugLine(object_XY, goal_XY, lifeTime=0.1)
        dist = self.task.calc_distance(object_XY, goal_XY)
        if self.last_move_dist is None: #or self.last_owner != 1:
           self.last_move_dist = dist
        reward = self.last_move_dist - dist
        self.last_move_dist = dist
        #ix = 1 if self.num_networks > 1 else 0
        self.network_rewards[2] += reward
        return reward


    def place_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("place", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.p.addUserDebugLine(object, goal, lifeTime=0.1)
        dist = self.task.calc_distance(object, goal)
        if self.last_place_dist is None: # or self.last_owner != 2:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist
        reward = reward * 10
        self.last_place_dist = dist
        if self.last_owner == 3 and dist < 0.1:
            self.env.robot.release_all_objects()
            self.gripped = None
            self.env.episode_info = "Object was placed to desired position"
        if self.env.episode_steps <= 2:
            self.env.episode_info = "Task finished in initial configuration"
            self.env.episode_over = True
        self.network_rewards[3] += reward
        return reward

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position],[gripper_position,object_position], [object_position, goal_position], [object_position, goal_position]][owner]
        reward = [self.above_compute,self.find_compute,self.move_compute, self.place_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.object_above_goal(gripper_position, object_position) or self.was_above:
            self.current_network = 1
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 2
        if self.object_above_goal(object_position, goal_position) or self.was_above:
            self.current_network = 3
            self.was_above = True
        return self.current_network
    

class FourStagePnPRot(ThreeStagePnPRot):
    """
    PnP rotate with four rewarded stages = find, move, place and rotate. Simple version with no release of object before
    rotation. Applicable for up to 4 networks.

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule

    """

    def __init__(self, env, task):
        super(ThreeStagePnPRot, self).__init__(env, task)
        self.was_above = False
        self.picked_object = False

    def check_num_networks(self):
        assert self.num_networks <= 4, "FourStagePnPRot reward can work with maximum 4 networks"

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.last_owner = None
        self.last_find_dist  = None
        self.last_lift_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.last_rot_dist = None
        self.picked_object = False
        self.was_near = False
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
    
    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, goal_position], [object_position, goal_position], [object_position, goal_position]][owner]
        reward = [self.find_compute, self.move_compute, self.place_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
            self.picked_object = True
        if self.object_above_goal(object_position, goal_position) or self.was_above:
            self.current_network = 2
            self.was_above = True
        if self.object_near_goal(object_position, goal_position) or self.was_near:
            self.current_network = 3
            self.was_near = True
        
        #self.env.p.addUserDebugText(f"Network:{self.current_network}", [0.7,0.7,1.0], lifeTime=0.1, textColorRGB=[55,125,0])
        return self.current_network
    
    def find_compute(self, gripper, object):
        # initial reach
        self.env.p.addUserDebugText("find object", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper[:3], object[:3])
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_find_dist is None:
            self.last_find_dist = dist
        if self.last_owner != 0:
            self.last_find_dist = dist
        reward = self.last_find_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_find_dist = dist
        #if self.task.check_object_moved(self.env.task_objects["actual_state"], threshold=1.2):
        #    self.env.episode_over   = True
        #    self.env.episode_failed = True
        self.network_rewards[0] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    def place_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("Subgoal: place", [.65, 1., 0.5], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        dist = self.task.calc_distance(object, goal)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.1, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None or self.last_owner != 2:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist
        reward = reward * 10
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.7,0.7,1.2], lifeTime=0.1, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        if self.last_owner == 2 and dist < 0.12:
            self.env.robot.release_all_objects()
            self.gripped = None
            self.env.episode_info = "Object was placed to desired position"
            #if self.task.number_tasks == self.task.current_task + 1:
            #    self.env.episode_over = True
        if self.env.episode_steps <= 2:
            self.env.episode_info = "Task finished in initial configuration"
            self.env.episode_over = True
        self.network_rewards[-1] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        return reward
    
    def move_compute(self, object, goal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("move", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[0,0,125])
        object_XY = object[:3]
        goal_XY   = goal[:3]
        self.env.p.addUserDebugLine(object_XY, goal_XY, lifeTime=0.1)
        dist = self.task.calc_distance(object_XY, goal_XY)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_move_dist is None or self.last_owner != 1:
           self.last_move_dist = dist
        reward = self.last_move_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_move_dist = dist
        ix = 1 if self.num_networks > 1 else 0
        self.network_rewards[ix] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[ix]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    
    def rotate_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("rotate", [0.63,1,0.5], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        dist = self.task.calc_distance(object, goal)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None or self.last_owner != 2:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist

        rot = self.task.calc_rot_quat(object, goal)
        self.env.p.addUserDebugText("Rotation: {}".format(round(rot,3)), [0.65,1,0.6], lifeTime=0.5, textColorRGB=[0, 222, 100])
        if self.last_rot_dist is None or self.last_owner != 2:
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        reward = reward + rewardrot

        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        self.last_rot_dist = rot
        #if self.last_owner == 2 and dist < 0.1:
        #    self.env.robot.release_all_objects()
        #    self.gripped = None
        #    self.env.episode_info = "Object was placed to desired position"
        #    if self.task.number_tasks == self.task.current_task + 1:
        #        self.env.episode_over = True
        #if self.env.episode_steps <= 2:
        #    self.env.episode_info = "Task finished in initial configuration"
        #    self.env.episode_over = True
        self.network_rewards[-1] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward

    def object_near_goal(self, object, goal):
        distance  = self.task.calc_distance(goal, object)
        if distance < 0.1:
            return True
        return False
    
class TwoStagePnPBgrip(TwoStagePnP):

    def gripper_reached_object(self, gripper, object):
        self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        #if self.current_network == 0:
        #    self.env.robot.magnetize_object(self.env.env_objects["actual_state"])
        if "gripper" in self.env.robot_action:
            if self.task.calc_distance(gripper, object) <= 0.08:
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[0, 255, 0, 1])
                return True
            else:
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[255, 0, 0, 1])
                return False
        else:
            if self.env.env_objects["actual_state"] in self.env.robot.magnetized_objects.keys():
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[0, 255, 0, 1])
                return True
        return False

class TwoStagePnPBgrip(TwoStagePnP):

    def gripper_reached_object(self, gripper, object):
        self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        #if self.current_network == 0:
        #    self.env.robot.magnetize_object(self.env.env_objects["actual_state"])
        if "gripper" in self.env.robot_action:
            if self.task.calc_distance(gripper, object) <= 0.08:
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[0, 255, 0, 1])
                return True
            else:
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[255, 0, 0, 1])
                return False
        else:
            if self.env.env_objects["actual_state"] in self.env.robot.magnetized_objects.keys():
                self.env.p.changeVisualShape(self.env.env_objects["actual_state"].uid, -1, rgbaColor=[0, 255, 0, 1])
                return True
        return False


class GripperPickAndPlace():
    """
    Pick and place with three neural networks, gripper is operated by one of these networks

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(GripperPickAndPlace, self).__init__(env, task)
        self.prev_object_position = [None]*3
        self.current_network  = 0
        self.picked = False
        self.moved  = False
        self.last_pick_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.network_rewards = [0,0,0]
        self.gripper_name = [x for x in self.env.task.obs_template["additional_obs"] if "endeff" in x][0]

    def reset(self):
        self.prev_object_position = [None]*3
        self.current_network  = 0
        self.picked = False
        self.moved  = False
        self.last_pick_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.network_rewards = [0,0,0]

    def compute(self, observation=None):
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        reward = [self.pick(gripper_position, object_position),self.move(goal_position, object_position),
                  self.place(goal_position, object_position)][owner]
        self.task.check_distance_threshold(observation,threshold=0.05)
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        self.current_network = 0
        if self.picked:
            self.current_network = 1
        if self.moved:
            self.current_network = 2
        return self.current_network

    def pick(self, gripper, object):
        self.env.p.addUserDebugText("pick", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper, object)
        if self.last_pick_dist is None:
            self.last_pick_dist = dist

        reward = self.last_pick_dist - dist
        self.last_pick_dist = dist

        if self.task.check_object_moved(self.env.task_objects["actual_state"], threshold=1):
            self.env.episode_over   = True
            self.env.episode_failed = True

        if dist < 0.1 and self.env.robot.gripper_active:
            self.picked = True
            reward += 1
        self.network_rewards[0]  += reward
        return reward

    def move(self, goal, object):
        self.env.p.addUserDebugText("move", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,125,0])
        dist = self.task.calc_distance(object, goal)
        if self.last_move_dist is None:
            self.last_move_dist = dist
        reward = self.last_move_dist - dist
        reward = reward * 1
        self.last_move_dist = dist
        if not self.env.robot.gripper_active:
            reward += -1
            self.picked = False
        if dist < 0.1:
            reward += 1
            self.moved = True
        self.network_rewards[1] += reward
        return reward

    def place(self, goal, object):
        self.env.p.addUserDebugText("place", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,125,125])
        dist = self.task.calc_distance(object, goal)

        self.env.p.addUserDebugText("distance " + str(dist), [0.5,0.5,0.5], lifeTime=0.1, textColorRGB=[0,125,125])
        if self.last_place_dist is None:
            self.last_place_dist = dist

        reward = self.last_place_dist - dist
        self.last_place_dist = dist

        if dist < 0.1 and self.is_poker_moving(object):
            if not self.env.robot.gripper_active:
                reward += 1
            else:
                reward -= 1

        if dist < 0.1 and not self.is_poker_moving(object):
            reward += 1
            self.env.episode_info = "Object was placed to desired location"
            self.env.episode_over = True
        elif dist > 0.1:
            self.picked = False
            self.moved  = False
            reward -= 1

        self.network_rewards[2] += reward
        return reward

class FaMaR(ThreeStagePnP):

    def reset(self):
        self.last_owner = None
        self.last_find_dist  = None
        self.last_lift_dist  = None
        self.last_move_dist  = None
        self.last_place_dist = None
        self.last_rot_dist = None
        self.subgoaloffset_dist = None
        self.last_leave_dist = None
        self.was_near = False
        self.current_network = 0
        self.network_rewards = [0] * self.num_networks
        self.use_magnet = True
        self.has_left = False
        self.last_traj_idx = 0
    
    def compute(self, observation=None):
        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, goal_position], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, goal_position) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network
    
    def find_compute(self, gripper, object):
        self.env.p.addUserDebugText("find object", [0.63,1,0.5], lifeTime=0.5, textColorRGB=[125,0,0])
        self.env.robot.set_magnetization(True)
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        dist = self.task.calc_distance(gripper[:3], object[:3])
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_find_dist is None:
            self.last_find_dist = dist
        
        reward = self.last_find_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        
        self.last_find_dist = dist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[0]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward


    def transform_compute_test(self, object, goal, trajectory, magnetization):
        """Calculate reward based on following a trajectory
        params: object: self-explanatory
                goal: self-explanatory
                trajectory: (np.array) 3D trajectory, lists of points x, y, z
                magnetization: (boolean) sets magnetization on or off
        Reward is calculated based on distance of object from goal and square distance of object from trajectory.
        That way, object tries to approach goal while trying to stay on trajectory path.
        """

        self.env.robot.set_magnetization(magnetization)
        self.env.p.addUserDebugText("transform", [0.7, 0.7, 0.7], lifeTime=0.1, textColorRGB=[125, 125, 0])
        dist_g = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist_g
        rewarddist = self.last_place_dist - dist_g
        self.env.p.addUserDebugText(f"RewardDist:{rewarddist}", [0.61, 1, 0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        pos = object[:3]
        dist_t, self.last_traj_idx = self.task.trajectory_distance(trajectory, pos, self.last_traj_idx, 10)
        self.env.p.addUserDebugText(f"Traj_Dist:{dist_t}", [0.61, 1, 0.45], lifeTime=0.5, textColorRGB=[0, 125, 0])
        reward = rewarddist - 20*dist_t**2
        self.env.p.addUserDebugText(f"reward:{reward}", [0.61, 1, 0.35], lifeTime=0.5, textColorRGB=[0, 125, 0])
        self.network_rewards[self.current_network] += reward
        return reward


    def move_compute(self, object, goal):
        self.env.robot.set_magnetization(False)
        self.env.p.addUserDebugText("move", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        object_XY = object[:3]
        goal_XY   = goal[:3]
        dist = self.task.calc_distance(object_XY, goal_XY)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_move_dist is None:
           self.last_move_dist = dist
        
        reward = self.last_move_dist - dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        
        self.last_move_dist = dist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    
    def rotate_compute(self, object, goal):
        self.env.p.addUserDebugText("rotate", [0.63,1,0.5], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.robot.set_magnetization(False)
        dist = self.task.calc_distance(object, goal)
        self.env.p.addUserDebugLine(object[:3], goal[:3], lifeTime=0.1)
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_place_dist is None:
            self.last_place_dist = dist
        reward = self.last_place_dist - dist

        rot = self.task.calc_rot_quat(object, goal)
        self.env.p.addUserDebugText("Rotation: {}".format(round(rot,3)), [0.65,1,0.6], lifeTime=0.5, textColorRGB=[0, 222, 100])
        if self.last_rot_dist is None: 
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot
        
        reward = reward + rewardrot
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_place_dist = dist
        self.last_rot_dist = rot
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    def transform_compute(self, object, goal, offsetgoal):
        self.env.p.addUserDebugText("transform", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.robot.set_magnetization(True)
        dist = self.task.calc_distance(object, goal)
        if self.last_place_dist is None:
            self.last_place_dist = dist
        rewarddist = self.last_place_dist - dist

        rot = self.task.calc_rot_quat(object, goal)
        if self.last_rot_dist is None:
            self.last_rot_dist = rot
        rewardrot = self.last_rot_dist - rot

        dist = self.task.calc_distance(object, offsetgoal)
        if self.subgoaloffset_dist is None:
           self.subgoaloffset_dist  = dist
        rewardsubgoaloffset = self.subgoaloffset_dist + dist
        
        reward = rewarddist + rewardrot + rewardsubgoaloffset
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61, 1, 0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])

        self.last_place_dist = dist
        self.last_rot_dist = rot
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65, 1, 0.7], lifeTime=0.5,
                                    textColorRGB=[0, 0, 125])
        return reward
    
    def leave_compute(self, gripper, object):
        self.env.p.addUserDebugText("leave", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,125,0])
        self.env.p.addUserDebugLine(gripper[:3], object[:3], lifeTime=0.1)
        self.env.robot.set_magnetization(False)
        self.env.robot.release_all_objects()
        dist = self.task.calc_distance(gripper[:3], object[:3])
        self.env.p.addUserDebugText("Distance: {}".format(round(dist,3)), [0.65,1,0.55], lifeTime=0.5, textColorRGB=[0, 125, 0])
        if self.last_leave_dist is None:
            self.last_leave_dist = dist
        reward = dist - self.last_leave_dist
        self.env.p.addUserDebugText(f"Reward:{reward}", [0.61,1,0.55], lifeTime=0.5, textColorRGB=[0,125,0])
        self.last_leave_dist = dist
        self.network_rewards[self.current_network] += reward
        self.env.p.addUserDebugText(f"Rewards:{self.network_rewards[-1]}", [0.65,1,0.7], lifeTime=0.5, textColorRGB=[0,0,125])
        return reward
    
    def left_out_of_threshold(self, gripper, object, threshold = 0.2):
        distance = self.task.calc_distance(gripper ,object)
        if distance > threshold:
            return True
        return False
    

    def object_near_goal(self, object, goal):
        distance  = self.task.calc_distance(goal, object)
        if distance < 0.1:
            return True
        return False
    
    def subgoal_offset(self, goal_position,offset):
        
        subgoal = [goal_position[0]-offset[0],goal_position[1]-offset[1],goal_position[2]-offset[2],goal_position[3],goal_position[4],goal_position[5],goal_position[6]]
        return subgoal


class FaROaM(FaMaR):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position,[0.2,0.2,0.2])], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.rotate_compute, self.move_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,[0.2,0.2,0.2])) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network

class FaMOaR(FaMaR):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position,[0.2,0.2,0.2])], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,[0.2,0.2,0.2])) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network

class FaMOaT(FaMaR):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        #[object_position, goal_position, self.task.create_line(self.subgoal_offset(goal_position), [0.3,0.0,0.0], goal_position), True]
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position,[0.3,0.0,0.0])], [object_position, goal_position, self.task.create_line(self.subgoal_offset(goal_position, [0.3,0.0,0.0]),  goal_position), True]][owner]
        reward = [self.find_compute,self.move_compute, self.transform_compute_test][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,[0.3,0.0,0.0])) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network

class FaROaT(FaMaR):
    
    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, self.subgoal_offset(goal_position,[0.3,0.0,0.0])], [object_position, goal_position,self.subgoal_offset(goal_position,[0.3,0.0,0.0])]][owner]
        reward = [self.find_compute,self.rotate_compute, self.transform_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position):
            self.current_network = 1
        if self.object_near_goal(object_position, self.subgoal_offset(goal_position,[0.2,0.2,0.2])) or self.was_near:
            self.current_network = 2
            self.was_near = True
        return self.current_network
    

class FaMaLaFaR(FaMaR):
    def check_num_networks(self):
        assert self.num_networks <= 4, "Find&move&leave&find&rotate reward can work with maximum 4 networks"

    def compute(self, observation=None):

        owner = self.decide(observation)
        goal_position, object_position, gripper_position = self.get_positions(observation)
        target = [[gripper_position,object_position], [object_position, goal_position], [gripper_position, object_position], [object_position, goal_position]][owner]
        reward = [self.find_compute,self.move_compute, self.leave_compute, self.rotate_compute][owner](*target)
        self.last_owner = owner
        self.task.check_goal()
        self.rewards_history.append(reward)
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)
        if self.gripper_reached_object(gripper_position, object_position) and self.current_network == 0:
            self.current_network = 1
        if self.object_near_goal(object_position, goal_position) and self.has_left == False and self.current_network == 1:
            self.current_network = 2
        if self.left_out_of_threshold(gripper_position, goal_position, threshold = 0.3) and self.current_network == 2:
            self.current_network = 0
            self.has_left = True
        if self.gripper_reached_object(gripper_position, object_position) and self.has_left:
            self.current_network = 3
        return self.current_network