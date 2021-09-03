import numpy as np
from myGym.utils.vector import Vector
import matplotlib.pyplot as plt
from stable_baselines import results_plotter
import os
import math

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

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = Vector([0,0,0], gripper_position, self.env)
        return direction_vector.add_vector(gripper)

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
        observation = observation["observation"] if isinstance(observation, dict) else observation
        o1 = observation[0:3] if self.env.reward_type != "2dvu" else observation[0:int(len(observation[:-3])/2)]
        o2 = self.get_accurate_gripper_position(observation[3:6]) if self.env.reward_type != "2dvu" else observation[int(len(observation[:-3])/2):-3]
        reward = self.calc_dist_diff(o1, o2)         
        if self.task.check_object_moved(self.env.task_objects[0]): # if pushes goal too far
            self.env.episode_over   = True
            self.env.episode_failed = True
        self.task.check_reach_distance_threshold(observation)
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

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = Vector([0,0,0], gripper_position, self.env)
        return direction_vector.add_vector(gripper)

    def decide(self, observation=None):
        import random
        return random.randint(0, 1)


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
        reward = self.calc_dist_diff(observation[0:3], observation[3:6], observation[6:9])
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

        self.threshold             = 0.1
        self.last_distance         = None
        self.last_gripper_distance = None
        self.moved                 = False
        
        self.prev_poker_position   = [None]*3

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """

        self.last_distance         = None
        self.last_gripper_distance = None
        self.moved                 = False

        self.prev_poker_position   = [None]*3

    def compute(self, observation=None):
        poker_position, distance, gripper_distance = self.init(observation)
        
        reward = self.count_reward(poker_position, distance, gripper_distance)
        
        self.finish(observation, poker_position, distance, gripper_distance, reward)
        
        return reward

    def init(self, observation):
        # load positions
        goal_position    = observation[0:3]
        poker_position   = observation[3:6]
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
        self.last_distance         = distance
        self.last_gripper_distance = gripper_distance
        self.prev_poker_position   = poker_position

    def check_strength_threshold(self):
        if self.task.check_object_moved(self.env.task_objects[1], 2):
            self.env.episode_over   = True
            self.env.episode_failed = True
            self.env.episode_info   = "too strong poke"

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = Vector([0,0,0], gripper_position, self.env)
        return direction_vector.add_vector(gripper)

    def is_poker_moving(self, poker):
        if self.prev_poker_position[0] == poker[0] and self.prev_poker_position[1] == poker[1]:
            return False
        elif self.env.episode_steps > 25:   # it slightly moves on the beginning
            self.moved = True
        return True

    def check_motion(self, poker):
        if not self.is_poker_moving(poker) and self.moved:
            self.env.episode_over   = True
            self.env.episode_failed = True
            self.env.episode_info   = "too weak poke"
            return True
        elif self.is_poker_moving(poker):
            return False
        return True

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

        self.prev_goals_positions       = [None]*(len(self.env.task_objects_names))
        self.prev_distractors_positions = [None]*(len(self.env.distractors))
        self.prev_links_positions       = [None]*11
        self.prev_gipper_position       = [None, None, None]

        self.touches                    = 0

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_goals_positions       = [None]*(len(self.env.task_objects_names))
        self.prev_distractors_positions = [None]*(len(self.env.distractors))
        self.prev_links_positions       = [None]*(self.env.robot.end_effector_index+1)
        self.prev_gipper_position       = [None, None, None]

        self.touches                    = 0
        self.env.distractor_stopped     = False

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation

        goals, distractors, arm, links = [], [], [], []
        self.fill_objects_lists(goals, distractors, arm, links, observation)

        reward = 0

        contact_points = [self.env.p.getContactPoints(self.env.robot.robot_uid, self.env.env_objects[-1].uid, x, -1) for x in range(0,self.env.robot.end_effector_index+1)]
        for p in contact_points:
            if p:
                reward = -5

                self.touches += 1
                self.rewards_history.append(reward)
                if self.touches > 20:
                    if self.env.episode_reward > 0:
                        self.env.episode_reward = -1 

                    self.episode_number     = 1024
                    self.env.episode_info   = "Arm crushed the distractor"
                    self.env.episode_over   = True
                    self.env.episode_failed = True

                    return reward

        # end_effector  = links[self.env.robot.gripper_index]
        gripper       = links[-1] # = end effector
        goal_position = goals[0]
        distractor    = self.env.env_objects[1]
        closest_distractor_points = self.env.p.getClosestPoints(self.env.robot.robot_uid, distractor.uid, self.env.robot.gripper_index)

        minimum = 1000
        for point in closest_distractor_points:
            if point[8] < minimum:
                # if point[8] < 0:
                #     print("akruci, je potřeba udělat z point[8] abs hodnotu")
                minimum = point[8]
                closest_distractor_point = list(point[6])

        if not self.prev_gipper_position[0]:
            self.prev_gipper_position = gripper

        pull_amplifier = self.task.calc_distance(self.prev_gipper_position, goal_position)/1
        push_amplifier = (1/pow(self.task.calc_distance(closest_distractor_point, self.prev_gipper_position), 1/2))-2

        if push_amplifier < 0:
            push_amplifier = 0

        ideal_vector = Vector(self.prev_gipper_position, goal_position, self.env)
        push_vector  = Vector(closest_distractor_point, self.prev_gipper_position, self.env)
        pull_vector  = Vector(self.prev_gipper_position, goal_position, self.env)

        push_vector.set_len(push_amplifier)
        pull_vector.set_len(pull_amplifier)

        # ideal_vector.visualize(origin=self.prev_gipper_position, time = 0.3, color = (0, 255, 0))
        # push_vector.visualize(origin=closest_distractor_point, time = 0.3, color = (255, 0, 0))
        # pull_vector.visualize(origin=self.prev_gipper_position, time = 0.3, color = (0, 0, 255))
        
        push_vector.add(pull_vector)
        force_vector = push_vector

        # force_vector.visualize(origin=self.prev_gipper_position, time = 0.3, color = (255, 0, 255))


        force_vector.add(ideal_vector)
        optimal_vector = force_vector
        # optimal_vector.visualize(origin=self.prev_gipper_position, time = 0.3, color = (255, 255, 255))
        optimal_vector.multiply(0.005)

        real_vector = Vector(self.prev_gipper_position, gripper, self.env)
        # real_vector.visualize(origin=self.prev_gipper_position, time = 0.3, color = (0, 0, 0))

        # ideal_vector   = self.move_to_origin([self.prev_gipper_position, goal_position])
        # push_vector    = self.set_vector_len(self.move_to_origin([closest_distractor_point, self.prev_gipper_position]), push_amplifier)
        # pull_vector    = self.set_vector_len(self.move_to_origin([self.prev_gipper_position, goal_position]), pull_amplifier)   #(*2)
        # force_vector   = np.add(np.array(push_vector), np.array(pull_vector))
        # optimal_vector = np.add(np.array(ideal_vector), np.array(force_vector))
        # optimal_vector = optimal_vector * 0.005  # move to same řád as is real vector and divide by 2 (just for visualization aesthetics)
        # real_vector    = self.move_to_origin([self.prev_gipper_position, gripper])

        # self.visualize_vectors(gripper, goal_position, force_vector, optimal_vector)

        if real_vector.norm == 0:
            reward += 0
        else:
            reward += np.dot(self.set_vector_len(optimal_vector.vector, 1), self.set_vector_len(real_vector.vector, 1))

        self.prev_gipper_position = gripper

        # self.task.check_distance_threshold(observation=observation)
        self.task.check_distractor_distance_threshold(goal_position, gripper)

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
        # self.env.p.addUserDebugLine(closest_distractor_point, self.gripper, lineColorRGB=(255, 255, 0), lineWidth = push_amplifier, lifeTime = 1)
        # self.env.p.addUserDebugLine(gripper, goal_position, lineColorRGB=(255, 128, 0), lineWidth = pull_amplifier, lifeTime = 1)
        self.env.p.addUserDebugLine(gripper, np.add(np.array(force_vector), np.array(gripper)), lineColorRGB=(255, 0, 0), lineWidth = 1, lifeTime = 0.1)
        self.env.p.addUserDebugLine(gripper, np.add(np.array(optimal_vector*100), np.array(gripper)), lineColorRGB=(0, 0, 255), lineWidth = 1, lifeTime = 0.1)
        # self.env.p.addUserDebugLine(gripper, np.add(np.array(self.prev_gipper_position), np.array(end_effector)), lineColorRGB=(0, 255, 0), lineWidth = 10, lifeTime = 1)


        # ideal_vector,  lineColorRGB=(255, 255, 255)
        # push_vector,   lineColorRGB=(255, 255, 0)
        # pull_vector,   lineColorRGB=(255, 128, 0)
        # force_vector   lineColorRGB=(255, 0, 0)
        # optimal_vector lineColorRGB=(0, 0, 255)
        # real_vector    lineColorRGB=(0, 255, 0)

    def fill_objects_lists(self, goals, distractors, arm, links, observation):

        # observation:
        #              first n 3: goals         (n = number of goals)       (len(self.env.task_objects_names))
        #              last  n 3: distractors   (n = number of distractors) (len(self.env.distractors))
        #              next    3: arm
        #              lasting 3: links

        items_count = len(self.env.task_objects_names) + len(self.env.distractors) + 1 + self.env.robot.end_effector_index+1

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

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = Vector([0,0,0], gripper_position, self.env)
        return direction_vector.add_vector(gripper)

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
        norm    = self.count_vector_norm(vector)
        vector  = self.multiply_vector(vector, 1/norm)

        return self.multiply_vector(vector, len)

    def get_angle_between_vectors(self, v1, v2):
        return math.acos(np.dot(v1, v2)/(self.count_vector_norm(v1)*self.count_vector_norm(v2)))

# dual rewards

class DualPoke(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(DualPoke, self).__init__(env, task)

        self.prev_poker_position = [None]*3

        self.touched               = False

        self.last_aimer_dist       = 0
        self.last_poker_dist       = 0

        self.poker_reward = 0
        self.aimer_reward = 0

        self.last_owner = None

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_poker_position = [None]*3

        self.touched               = False

        self.last_aimer_dist       = 0
        self.last_poker_dist       = 0

        self.poker_reward = 0
        self.aimer_reward = 0

        self.last_owner = None

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        reward = 0
        owner  = self.decide(observation)

        if owner   == 0: reward = self.aimer_compute(observation)
        elif owner == 1: reward = self.poker_compute(observation)
        else:            exit("decision error")


        self.last_owner = owner
        self.task.check_poke_threshold(observation)
        self.rewards_history.append(reward)

        return reward

    def aimer_compute(self, observation=None):
        self.env.p.addUserDebugText("XXX", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,125,0])    
        goal_position, poker_position, gripper_position = self.get_positions(observation)
        aim_point = self.get_aim_point(goal_position, poker_position)

        aimer_dist = self.task.calc_distance(gripper_position, aim_point)

        if self.last_owner != 0 or self.last_aimer_dist is None:
            self.last_aimer_dist = aimer_dist

        reward = self.last_aimer_dist - aimer_dist        
        # if self.is_poker_moving(poker_position):
            # self.touched = True
            # reward = -5
            # self.env.episode_over = True
            # self.env.episode_failed = True
            # self.env.episode_info = "unintended poke"
        self.aimer_reward += reward
     
        self.env.p.addUserDebugText(str(self.aimer_reward), [0.5,0.5,0.5], lifeTime=0.1, textColorRGB=[0,125,0])   
        self.last_aimer_dist = aimer_dist
        if self.env.episode_steps > 25:
            self.tuched = self.is_poker_moving(poker_position)
        self.prev_poker_position = poker_position
        return reward

    def poker_compute(self, observation=None):
        self.env.p.addUserDebugText("XXX", [-0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,0,125])
        goal_position, poker_position, gripper_position = self.get_positions(observation)

        poker_dist = round(self.task.calc_distance(poker_position, goal_position), 5)

        if self.last_owner != 1 or self.last_poker_dist is None:
            self.last_poker_dist = poker_dist

        reward = 5*(self.last_poker_dist - poker_dist)
        if self.task.check_object_moved(self.env.task_objects[0], 1): # if pushes goal too far
            self.env.episode_over   = True
            self.env.episode_failed = True
            self.env.episode_info   = "too strong poke"
            reward = -5
        self.poker_reward += reward

        self.env.p.addUserDebugText(str(self.poker_reward), [-0.5,0.5,0.5], lifeTime=0.1, textColorRGB=[0,0,125])
        self.last_poker_dist = poker_dist
        self.prev_poker_position = poker_position
        
        if self.touched and not self.is_poker_moving(observation[3:6]):
            self.env.episode_over = True
            if self.last_poker_dist > 0.1:
                self.env.episode_failed = True
                self.env.episode_info   = "bad poke"
            else:
                self.env.episode_info   = "good poke"

        return reward

    def get_aim_point(self, goal_position, poker_position):
        aim_vector = Vector(poker_position, goal_position, self.env.p)
        aim_vector.set_len(-0.2)
        aim_point = poker_position + aim_vector.vector
        return aim_point

    def get_positions(self, observation):        
        observation      = observation["observation"] if isinstance(observation, dict) else observation
        goal_position    = observation[0:3]
        poker_position   = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[6:9])

        if self.prev_poker_position[0] is None:
            self.prev_poker_position = poker_position

        return goal_position,poker_position,gripper_position

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = Vector([0,0,0], gripper_position, self.env)
        return direction_vector.add_vector(gripper)



    def decide(self, observation=None):
        goal_position, poker_position, gripper_position = self.get_positions(observation)

        is_aimed = self.is_aimed(goal_position, poker_position, gripper_position)

        if self.touched:
            self.env.p.addUserDebugText("touched", [-0.3,0.3,0.3], lifeTime=0.1, textColorRGB=[0,0,125])

        if is_aimed or self.touched:
            owner = 1 # poke
        elif not is_aimed and not self.touched:
            owner = 0 # aim
        else:
            print("wat?")

        return owner

    def did_touch(self):
        contact_points = [self.env.p.getContactPoints(self.env.robot.robot_uid, self.env.env_objects[1].uid, x, -1) for x in range(0,self.env.robot.end_effector_index+1)]
        touched = False
        for point in contact_points:
            if len(point) > 0:
                touched = True
        return touched

    def is_aimed(self, goal_position, poker_position, gripper_position):
        poke_vector = Vector(poker_position, goal_position, self.env.p)
        poke_vector.set_len(-0.2)

        poker_in_XY = poker_position + poke_vector.vector

        # gripper_in_XY = [0.0, 0.3, poker_position[2]] # gripper initial position with z == 0
        poker_in_XY = [poker_in_XY[0], poker_in_XY[1], poker_position[2]] # gripper initial position with z == 0
        gripper_in_XY = poker_position + 3*poke_vector.vector
        len = self.distance_of_point_from_abscissa(gripper_in_XY, poker_position, gripper_position)

        if len < 0.1:
            return True
        return False
    
    def is_poker_moving(self, poker):
        if      round(self.prev_poker_position[0], 4) == round(poker[0], 4) \
            and round(self.prev_poker_position[1], 4) == round(poker[1], 4) \
            and round(self.prev_poker_position[2], 4) == round(poker[2], 4):
            return False
        else:
            self.touched = True
            return True

    def triangle_height(self, a, b, c):
        p = a+b+c
        p = p/2
        one = 2/a
        two = p*(p-a)
        three = (p-b)
        four = (p-c)
        five = two*three*four
        six = math.sqrt(five)
        return one*six

    def distance_of_point_from_abscissa(self, A, B, point):
        a = self.task.calc_distance(A, B)
        b = self.task.calc_distance(A, point)
        c = self.task.calc_distance(point, B)

        height_a = self.triangle_height(a, b, c)
        distance = height_a

        if b > a:
            distance = c
        if c > a:
            distance = c

        return distance

# pick and place rewards

class PickAndPlace(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(PickAndPlace, self).__init__(env, task)

        self.countdown = None

        self.last_owner = None

        self.object_before_lift = [None]*3

        self.lifted         = False

        self.grip           = None

        self.last_height    = None

        self.last_find_dist = None
        self.last_lift_dist = None
        self.last_move_dist = None
        self.last_place_dist = None
        self.owner = 0
        self.finder_reward  = 0
        self.lifter_reward  = 0
        self.mover_reward   = 0
        self.placer_reward  = 0

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.countdown = None

        self.last_owner = None

        self.object_before_lift = [None]*3

        self.lifted         = False

        self.grip           = None

        self.last_height    = None

        self.last_find_dist = None
        self.last_lift_dist = None
        self.last_move_dist = None
        self.last_place_dist = None
        self.owner = 0
        self.finder_reward  = 0
        self.lifter_reward  = 0
        self.mover_reward   = 0
        self.placer_reward  = 0

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
        elif owner == 2: reward = self.place_compute(object_position, goal_position)
        else:            exit("decision error")


        self.last_owner = owner
        self.task.check_pnp_threshold(observation)
        self.rewards_history.append(reward)

        return reward

    def find_compute(self, gripper, object):
        # initial reach
        self.env.p.addUserDebugText("find", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper, object)
        
        if self.last_find_dist is None:
            self.last_find_dist = dist
        if self.last_owner != 0:
            self.last_find_dist = dist

        reward = self.last_find_dist - dist
        self.last_find_dist = dist

        if self.task.check_object_moved(self.env.task_objects[1]):
            self.env.episode_over   = True
            self.env.episode_failed = True

        self.finder_reward += reward
        return reward

    def move_compute(self, object, goal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("move", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,0,125])

        object_XY = object
        goal_XY   = [goal[0], goal[1], goal[2]+0.2] 

        self.env.p.addUserDebugLine(object_XY, goal_XY, lifeTime=0.1)

        dist = self.task.calc_distance(object_XY, goal_XY)
        if self.last_move_dist is None:
           self.last_move_dist = dist
          #self.last_height    = height
        if self.last_owner != 1:
           self.last_move_dist = dist
          #self.last_height    = height

        reward = self.last_move_dist - dist
        self.last_move_dist = dist
        self.mover_reward += reward
        return reward

    def place_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("place", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,125,0])
        reward = 0

        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)

        self.owner = 1
        
       #if self.grip is None:
       #    self.grip_object()

        if not self.gripper_reached_object(gripper_position, object_position):
            self.owner = 0
        if self.object_above_goal(object_position, goal_position):
            self.owner = 2

        return self.owner

    def get_positions(self, observation):        
        observation      = observation["observation"] if isinstance(observation, dict) else observation
        goal_position    = observation[0:3]
        object_position  = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[-3:])

        if self.object_before_lift[0] is None:
            self.object_before_lift = object_position

        return goal_position,object_position,gripper_position

    def gripper_reached_object(self, gripper, object):
        self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        distance = self.task.calc_distance(gripper, object)

        if self.grip is not None:
            return True
        if distance < 0.1:
            self.grip_object()
            return True
        return False

    def object_lifted(self, object, object_before_lift):
        lifted_position = [object_before_lift[0], object_before_lift[1], object_before_lift[2]+0.1] # position of object before lifting but hightened with its height
        distance = self.task.calc_distance(object, lifted_position)
        if object[2] < 0.079:
            self.lifted = False # object has fallen
            self.object_before_lift = object
        # if distance < 0.05:
        #    self.lifted = True
        #    return True
        # elif self.lifted:
        #    return True
        else:
            self.lifted = True
            return True
        return False

    def object_above_goal(self, object, goal):
        goal_XY   = [goal[0],   goal[1],   0]
        object_XY = [object[0], object[1], 0]
        distance  = self.task.calc_distance(goal_XY, object_XY)
        # print(distance)
        if distance < 0.1:
            return True
        return False

    def grip_object(self):
        if self.countdown is None:
            object = self.env.env_objects[1]
            self.env.p.changeVisualShape(object.uid, -1, rgbaColor=[0, 255, 0, 1])
            self.grip = self.env.p.createConstraint(self.env.robot.robot_uid, self.env.robot.gripper_index, object.uid, -1, self.env.p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,-0.15])

    def release_object(self):
        if self.grip is not None:
            object = self.env.env_objects[1]
            self.env.p.changeVisualShape(object.uid, -1, rgbaColor=[0,0,0,1])
            self.env.p.removeConstraint(self.grip)
        self.grip = None


class ConsequentialPickAndPlace(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(ConsequentialPickAndPlace, self).__init__(env, task)

        self.countdown = None

        self.last_owner = None

        self.object_before_lift = [None]*3

        self.lifted         = False

        self.grip           = None

        self.last_height    = None

        self.last_find_dist = None
        self.last_lift_dist = None
        self.last_move_dist = None
        self.last_place_dist = None
        self.owner = 0
        self.finder_reward  = 0
        self.lifter_reward  = 0
        self.mover_reward   = 0
        self.placer_reward  = 0

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.countdown = None

        self.last_owner = None

        self.object_before_lift = [None]*3

        self.lifted         = False

        self.grip           = None

        self.last_height    = None

        self.last_find_dist = None
        self.last_lift_dist = None
        self.last_move_dist = None
        self.last_place_dist = None
        self.owner = 0
        self.finder_reward  = 0
        self.lifter_reward  = 0
        self.mover_reward   = 0
        self.placer_reward  = 0

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
        elif owner == 2: reward = self.place_compute(object_position, goal_position)
        else:            exit("decision error")


        self.last_owner = owner
        self.task.check_pnp_threshold(observation)
        self.rewards_history.append(reward)

        return reward

    def find_compute(self, gripper, object):
        # initial reach
        self.env.p.addUserDebugText("find", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper, object)
        
        if self.last_find_dist is None:
            self.last_find_dist = dist
        if self.last_owner != 0:
            self.last_find_dist = dist

        reward = self.last_find_dist - dist
        self.last_find_dist = dist

        if self.task.check_object_moved(self.env.task_objects[1]):
            self.env.episode_over   = True
            self.env.episode_failed = True

        self.finder_reward += reward
        return reward

    def move_compute(self, object, goal):
        # moving object above goal position (forced 2D reach)
        self.env.p.addUserDebugText("move", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[0,0,125])

        object_XY = object
        goal_XY   = [goal[0], goal[1], goal[2]+0.2] 

        self.env.p.addUserDebugLine(object_XY, goal_XY, lifeTime=0.1)

        dist = self.task.calc_distance(object_XY, goal_XY)
        if self.last_move_dist is None:
           self.last_move_dist = dist
          #self.last_height    = height
        if self.last_owner != 1:
           self.last_move_dist = dist
          #self.last_height    = height

        reward = self.last_move_dist - dist
        self.last_move_dist = dist
        self.mover_reward += reward
        return reward

    def place_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("place", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,125,0])
        dist = self.task.calc_distance(object, goal)
       
        if self.last_place_dist is None:
            self.last_place_dist = dist
        if self.last_owner != 2:
            self.last_place_dist = dist

        reward = self.last_place_dist - dist
        reward = reward * 10
        self.last_place_dist = dist

        if self.last_owner == 2 and dist < 0.1:
            self.release_object()
            self.env.episode_info = "Object was placed to desired position" 
            self.countdown = 0
        if self.env.episode_steps <= 5:
            self.env.episode_info = "Task finished in initial configuration"
            self.env.episode_over = True

        if self.countdown is not None:
            reward = 0
            if self.countdown == 5:
                self.env.episode_over = True
            else:
                self.countdown += 1

        self.placer_reward += reward
        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)

        self.owner = 1
        
       #if self.grip is None:
       #    self.grip_object()

        if not self.gripper_reached_object(gripper_position, object_position):
            self.owner = 0
        if self.object_above_goal(object_position, goal_position):
            self.owner = 2

        return self.owner

    def get_positions(self, observation):        
        observation      = observation["observation"] if isinstance(observation, dict) else observation
        goal_position    = observation[0:3]
        object_position  = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[-3:])

        if self.object_before_lift[0] is None:
            self.object_before_lift = object_position

        return goal_position,object_position,gripper_position

    def gripper_reached_object(self, gripper, object):
        self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        distance = self.task.calc_distance(gripper, object)

        if self.grip is not None:
            return True
        if distance < 0.1:
            self.grip_object()
            return True
        return False

    def object_lifted(self, object, object_before_lift):
        lifted_position = [object_before_lift[0], object_before_lift[1], object_before_lift[2]+0.1] # position of object before lifting but hightened with its height
        distance = self.task.calc_distance(object, lifted_position)
        if object[2] < 0.079:
            self.lifted = False # object has fallen
            self.object_before_lift = object
        # if distance < 0.05:
        #    self.lifted = True
        #    return True
        # elif self.lifted:
        #    return True
        else:
            self.lifted = True
            return True
        return False

    def object_above_goal(self, object, goal):
        goal_XY   = [goal[0],   goal[1],   0]
        object_XY = [object[0], object[1], 0]
        distance  = self.task.calc_distance(goal_XY, object_XY)
        # print(distance)
        if distance < 0.1:
            return True
        return False

    def grip_object(self):
        if self.countdown is None:
            object = self.env.env_objects[1]
            self.env.p.changeVisualShape(object.uid, -1, rgbaColor=[0, 255, 0, 1])
            self.grip = self.env.p.createConstraint(self.env.robot.robot_uid, self.env.robot.gripper_index, object.uid, -1, self.env.p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,-0.15])

    def release_object(self):
        if self.grip is not None:
            object = self.env.env_objects[1]
            self.env.p.changeVisualShape(object.uid, -1, rgbaColor=[0,0,0,1])
            self.env.p.removeConstraint(self.grip)
        self.grip = None

class DualPickAndPlace(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(DualPickAndPlace, self).__init__(env, task)

        self.countdown = None

        self.last_owner = None

        self.object_before_lift = [None]*3

        self.lifted         = False

        self.grip           = None

        self.last_height    = None

        self.last_find_dist = None
        self.last_lift_dist = None
        self.last_move_dist = None
        self.last_place_dist = None
        self.owner = 0
        self.finder_reward  = 0
        self.lifter_reward  = 0
        self.mover_reward   = 0
        self.placer_reward  = 0

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.countdown = None

        self.last_owner = None

        self.object_before_lift = [None]*3

        self.lifted         = False

        self.grip           = None

        self.last_height    = None

        self.last_find_dist = None
        self.last_lift_dist = None
        self.last_move_dist = None
        self.last_place_dist = None
        self.owner = 0
        self.finder_reward  = 0
        self.lifter_reward  = 0
        self.mover_reward   = 0
        self.placer_reward  = 0

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
        else:            exit("decision error")

        self.last_owner = owner
        self.task.check_pnp_threshold(observation)
        self.rewards_history.append(reward)

        return reward

    def decide(self, observation=None):
        goal_position, object_position, gripper_position = self.get_positions(observation)

        self.owner = 0

        if self.gripper_reached_object(gripper_position, object_position) or self.countdown is not None:
            self.owner = 1

        return self.owner

    def gripper_reached_object(self, gripper, object):
        self.env.p.addUserDebugLine(gripper, object, lifeTime=0.1)
        distance = self.task.calc_distance(gripper, object)

        if self.grip is not None:
            return True
        if distance < 0.1:
            self.grip_object()
            return True
        return False

    
    def find_compute(self, gripper, object):
        # initial reach
        self.env.p.addUserDebugText("find", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,0,0])
        dist = self.task.calc_distance(gripper, object)
        
        if self.last_find_dist is None:
            self.last_find_dist = dist
        if self.last_owner != 0:
            self.last_find_dist = dist

        reward = self.last_find_dist - dist
        self.last_find_dist = dist

        if self.task.check_object_moved(self.env.task_objects[1]):
            self.env.episode_over   = True
            self.env.episode_failed = True

        self.finder_reward += reward
        return reward

    def move_compute(self, object, goal):
        # reach of goal position + task object height in Z axis and release
        self.env.p.addUserDebugText("place", [0.7,0.7,0.7], lifeTime=0.1, textColorRGB=[125,125,0])
        dist = self.task.calc_distance(object, goal)
       
        if self.last_place_dist is None:
            self.last_place_dist = dist
        if self.last_owner != 1:
            self.last_place_dist = dist

        reward = self.last_place_dist - dist
        reward = reward * 10
        self.last_place_dist = dist

        if self.last_owner == 1 and dist < 0.1:
            self.release_object()
            self.env.episode_info = "Object was placed to desired position" 
            self.countdown = 0
        if self.env.episode_steps <= 5:
            self.env.episode_info = "Task finished in initial configuration"
            self.env.episode_over = True

        if self.countdown is not None:
            reward = 0
            if self.countdown == 20:
                self.env.episode_over = True
            else:
                self.countdown += 1

        self.placer_reward += reward
        return reward

    def get_positions(self, observation):        
        observation      = observation["observation"] if isinstance(observation, dict) else observation
        goal_position    = observation[0:3]
        object_position  = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[-3:])

        if self.object_before_lift[0] is None:
            self.object_before_lift = object_position

        return goal_position,object_position,gripper_position

    def grip_object(self):
        if self.countdown is None:
            object = self.env.env_objects[1]
            self.env.p.changeVisualShape(object.uid, -1, rgbaColor=[0, 255, 0, 1])
            self.grip = self.env.p.createConstraint(self.env.robot.robot_uid, self.env.robot.gripper_index, object.uid, -1, self.env.p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,-0.15])

    def release_object(self):
        if self.grip is not None:
            object = self.env.env_objects[1]
            self.env.p.changeVisualShape(object.uid, -1, rgbaColor=[0,0,0,1])
            self.env.p.removeConstraint(self.grip)
        self.grip = None

class Halt(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(Halt, self).__init__(env, task)
        self.last_position = [None]*3

        self.reward = 1

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.last_position = [None]*3

        self.reward = 1

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        gripper = observation[-3:]

        if self.last_position[0] is None:
            self.last_position = gripper

        for i in range(len(gripper)):
            self.reward -= round(abs(self.last_position[i] - gripper[i]), 5)

        self.rewards_history.append(self.reward)
        return self.reward

    def decide(self, observation=None):
        return 0