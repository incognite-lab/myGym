import numpy as np
import vector as v
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
        direction_vector    = v.Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = v.Vector([0,0,0], gripper_position, self.env)
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
        direction_vector    = v.Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = v.Vector([0,0,0], gripper_position, self.env)
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

class PokeReward(Reward):

    def __init__(self, env, task):
        super(PokeReward, self).__init__(env, task)

        self.prev_goal_position    = [None]*3
        self.prev_poker_position   = [None]*3
        self.prev_gripper_position = [None]*3

        self.threshold             = 0.1
        self.last_distance         = None

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_goal_position    = [None]*3
        self.prev_poker_position   = [None]*3
        self.prev_gripper_position = [None]*3

        self.last_distance         = None

    def compute(self, observation=None):

        # load positions
        goal_position    = observation[0:3]
        poker_position   = observation[3:6]
        gripper_position = observation[6:9]

        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)

        direction = [0, 0, 1]                       # length is 1
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])

        orientation_vector = m.dot(direction)       # length is 1
        orientation_vector = orientation_vector*0.1

        gripper_position[0] = gripper_position[0]+orientation_vector[0]
        gripper_position[1] = gripper_position[1]+orientation_vector[1]
        gripper_position[2] = gripper_position[2]+orientation_vector[2]

        # make sure none is None
        if self.prev_poker_position[0] is None:
            self.prev_poker_position   = poker_position

        if self.prev_gripper_position[0] is None:
            self.prev_gripper_position = gripper_position

        if self.prev_goal_position[0] is None:   #is stationary for now
            self.prev_goal_position    = goal_position

        # with gripper in [0.0, Y, Z] coords
        poke_vector  = self.move_to_origin([self.prev_poker_position, goal_position])  # goal is stationary (for now ;)
        align_vector = self.set_vector_len([-poke_vector[0], poke_vector[1], 0], 1) # to poke line
        real_vector  = self.move_to_origin([self.prev_gripper_position, gripper_position])

        self.env.p.addUserDebugLine(goal_position, poker_position, lineColorRGB=(255, 0, 0), lineWidth = 1, lifeTime = 0.1)
        # self.env.p.addUserDebugLine([goal_position[0], goal_position[1], goal_position[2]+0.05], [poker_position[0], poker_position[1], poker_position[2]+0.05], lineColorRGB=(0, 0, 255), lineWidth = 1, lifeTime = 1)
        poker_vector = self.set_vector_len(poke_vector, -10)
        self.env.p.addUserDebugLine(poker_position, np.add(poker_position, poker_vector), lineColorRGB=(255, 0, 0), lineWidth = 1, lifeTime = 0.1)
        self.env.p.addUserDebugLine(gripper_position, goal_position, lineColorRGB=(255, 0, 0), lineWidth = 1, lifeTime = 0.1)

        # vzdálenost gripperu od přímky protínající goal a poker
        distance = self.get_distance_from_poke_line(poke_vector, poker_position, goal_position)

        if self.last_distance is None:
            self.last_distance = distance

        # reward
        # if is poker in motion
        if self.is_poker_moving(poker_position): 

            self.prev_poker_position = poker_position
            self.task.check_poke_threshold(observation)

            # reward = -abs(self.count_vector_norm(real_vector))
            reward = 0

            self.rewards_history.append(reward)
            return reward

        speed = 1000*self.count_vector_norm(real_vector)

        # if is poker stationary
        if distance < self.threshold:
            # gripper is in align with poker and goal
            # go in direction of goal
            print("close")
            optimal_vector = self.set_vector_len(poke_vector, 1)

            if self.count_vector_norm(real_vector) == 0:
                reward = 0
            else:
                reward = np.dot(optimal_vector, self.set_vector_len(real_vector, 1))
                reward *= speed # rewards higher speed of motion (further poke)
        else:
            # gripper needs to align
            optimal_vector = align_vector

            reward = ((self.last_distance - distance) / self.last_distance)*speed # rewards higher speed of motion

        self.prev_poker_position   = poker_position
        self.prev_gripper_position = gripper_position
        self.prev_goal_position    = goal_position
        self.last_distance         = distance

        self.task.check_poke_threshold(observation)

        self.rewards_history.append(reward)
        return reward

    def move_to_origin(self, vector):
        a = vector[1][0] - vector[0][0]
        b = vector[1][1] - vector[0][1]
        c = vector[1][2] - vector[0][2]       
        
        return [a, b, c]

    def set_vector_len(self, vector, len):
        norm    = self.count_vector_norm(vector)
        vector  = self.multiply_vector(vector, 1/norm)

        return self.multiply_vector(vector, len)

    def multiply_vector(self, vector, multiplier):
        return np.array(vector) * multiplier

    def distance_between_vectors(self, v1, v2):
        return 0

    def distance_between_point_and_vector(self, point, vector):
        return 0

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

    def poke_line(self, x, goal_position, poker_position):

        align_factor = (goal_position[1]-poker_position[1])/(goal_position[0]-poker_position[0])
        addition     = poker_position[1] - align_factor*poker_position[0]
        return align_factor*x + addition

    def count_vector_norm(self, vector):
        return math.sqrt(np.dot(vector, vector))

    def is_poker_moving(self, poker):
        if self.prev_poker_position[0] == poker[0] and self.prev_poker_position[1] == poker[1] and self.prev_poker_position[1] == poker[1]:
            return False
        return True

    def get_distance_from_poke_line(self, poke_vector, poker_position, goal_position):
        z = self.prev_gripper_position[2]

        a = self.count_vector_norm(poke_vector)
        b = math.sqrt((self.prev_gripper_position[0]-poker_position[0])**2 + (self.prev_gripper_position[1]-poker_position[1])**2)
        c = math.sqrt((self.prev_gripper_position[0]-goal_position[0])**2 + (self.prev_gripper_position[1]-goal_position[1])**2)

        if a < 0.1:
            print("_____")
            self.env.episode_info = "Successfull poke"
            self.env.episode_over = True

        a = round(a, 5)
        b = round(b, 5)
        c = round(c, 5)

        while b+c <= a:
            c += 0.00001

        while a+c <= b:
            c += 0.00001

        while a+b <= c:
            b += 0.00001

        distanceXY = math.sqrt(self.triangle_height(a, b, c))
        distanceZ  = z-0.0 # moving the line into height of center of the cubes
        distance   = math.sqrt(distanceZ**2+distanceXY**2)

        return distance

class PokeVectorReward(Reward):

    def __init__(self, env, task):
        super(PokeVectorReward, self).__init__(env, task)

        # self.prev_goal_position    = [None]*3
        self.prev_poker_position   = [None]*3
        self.prev_gripper_position = [None]*3

        self.last_align            = 0
        self.last_len              = 0

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        # self.prev_goal_position    = [None]*3
        self.prev_poker_position   = [None]*3
        self.prev_gripper_position = [None]*3

        self.last_align            = 0
        self.last_len              = 0

    def compute(self, observation=None):

        # load positions
        goal_position    = observation[0:3]
        poker_position   = observation[3:6]
        gripper_position = observation[6:9]

        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)

        # direction = [0, 0, 0.1]                         # length is 0.1
        direction_vector = v.Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])

        # orientation_vector = m.dot(direction)           # length is 0.1
        direction_vector.rotate_with_matrix(m)

        # gripper_position = np.add(gripper_position, direction_vector.vector)
        gripper = v.Vector([0,0,0], gripper_position, self.env)
        gripper_position = direction_vector.add_vector(gripper)

        # make sure none is None
        if self.prev_poker_position[0] is None:
            self.prev_poker_position   = poker_position

        if self.prev_gripper_position[0] is None:
            self.prev_gripper_position = gripper_position

        # if self.prev_goal_position[0] is None:   #is stationary for now
        #     self.prev_goal_position    = goal_position

        # align
        poke_vector = v.Vector(self.prev_poker_position, goal_position, self.env)
        aim_vector  = v.Vector(self.prev_gripper_position, self.prev_poker_position, self.env)

        # align = poke_vector.get_align(aim_vector)

        align = np.dot(self.set_vector_len(poke_vector.vector, 1), self.set_vector_len(aim_vector.vector, 1))

        # reward
        align_factor = 0
        poke_factor  = 0

        len = aim_vector.norm
        align_factor = (align - self.last_align) + (self.last_len - len)

        if align > 0.95:
            real_vector = v.Vector(self.prev_gripper_position, gripper_position, self.env)
            # poke_factor = poke_vector.get_align(real_vector)
            poke_factor = np.dot(self.set_vector_len(poke_vector.vector, 1), self.set_vector_len(real_vector.vector, 1))
            # real_vector.visualize(self.prev_gripper_position, color=(0, 0, 255))

        # poke_vector.visualize(self.prev_poker_position, color=(255, 0, 0))
        # aim_vector.visualize(self.prev_gripper_position, color=(0, 255, 0))

        # print()
        # print("align:        ", align)
        # print("align_factor: ", align_factor)
        # print("poke_factor:  ", poke_factor)
        # print("reward:       ", align_factor + poke_factor)

        reward = align_factor + poke_factor

        if self.is_poker_moving(poker_position): 
            reward = 0

        self.prev_poker_position   = poker_position
        self.prev_gripper_position = gripper_position
        self.last_align            = align
        self.last_len              = len

        if self.task.check_object_moved(self.env.task_objects[1], 2):
            self.env.episode_over   = True
            self.env.episode_failed = True
            self.env.episode_info   = "too strong poke"
        self.task.check_poke_threshold(observation)

        self.rewards_history.append(reward)
        return reward

    def is_poker_moving(self, poker):
        if self.prev_poker_position[0] == poker[0] and self.prev_poker_position[1] == poker[1] and self.prev_poker_position[1] == poker[1]:
            return False
        return True

    def set_vector_len(self, vector, len):
        norm    = self.count_vector_norm(vector)
        vector  = self.multiply_vector(vector, 1/norm)

        return self.multiply_vector(vector, len)

    def count_vector_norm(self, vector):
        return math.sqrt(np.dot(vector, vector))

    def multiply_vector(self, vector, multiplier):
        return np.array(vector) * multiplier

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

        ideal_vector = v.Vector(self.prev_gipper_position, goal_position, self.env)
        push_vector  = v.Vector(closest_distractor_point, self.prev_gipper_position, self.env)
        pull_vector  = v.Vector(self.prev_gipper_position, goal_position, self.env)

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

        real_vector = v.Vector(self.prev_gipper_position, gripper, self.env)
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
        direction_vector    = v.Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = v.Vector([0,0,0], gripper_position, self.env)
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

class DualCPoke(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(DualCPoke, self).__init__(env, task)

        # self.prev_goal_position    = [None]*3
        self.prev_poker_position   = [None]*3
        self.prev_gripper_position = [None]*3

        self.last_align            = 0
        self.last_len              = 0

        self.owner_id              = -1

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        # self.prev_goal_position    = [None]*3
        self.prev_poker_position   = [None]*3
        self.prev_gripper_position = [None]*3

        self.last_align            = 0
        self.last_len              = 0

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        poker_position, gripper_position, poke_vector, aim_vector = self.init(observation)

        # align = poke_vector.get_align(aim_vector)
        align = np.dot(self.set_vector_len(poke_vector.vector, 1), self.set_vector_len(aim_vector.vector, 1))

        len = aim_vector.norm
        align_factor = (align - self.last_align) + 0.1*(self.last_len - len)
        self.env.p.addUserDebugText("align_factor: " + str(round(align - self.last_align, 4)), [-0.5,0.5,0.5], lifeTime=0.1)
        self.env.p.addUserDebugText("dist_factor: " + str(round((self.last_len - len)*0.1, 4)), [-0.7,0.7,0.7], lifeTime=0.1)

        self.env.p.addUserDebugText("Align: " + str(round(align, 3)), [0.5,0.5,0.5], lifeTime=0.1)

        if align > 0.95:
            real_vector = v.Vector(self.prev_gripper_position, gripper_position, self.env)
            # poke_factor = poke_vector.get_align(real_vector)
            poke_factor = np.dot(self.set_vector_len(poke_vector.vector, 1), self.set_vector_len(real_vector.vector, 1))
            real_vector.visualize(self.prev_gripper_position, color=(0, 0, 255))
            poke_vector.visualize(self.prev_poker_position, color=(255, 0, 0))
            reward = poke_factor
        else:
            reward = align_factor
            aim_vector.visualize(self.prev_gripper_position, color=(0, 255, 0))

        if self.env.episode_steps > 25:
            if self.is_poker_moving(poker_position):
                reward = 0
        elif self.env.episode_steps < 2:
            reward = 0

        self.finish(observation, poker_position, gripper_position, align, len, reward)

        self.env.p.addUserDebugText("Reward: " + str(round(self.env.episode_reward, 7)), [0.7,0.7,0.7], lifeTime=0.1)
        return reward

    def init(self, observation):
        # load positions
        goal_position    = observation[0:3]
        poker_position   = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[6:9])

        self.initialize_positions(poker_position, gripper_position)

        # align
        poke_vector = v.Vector(self.prev_poker_position, goal_position, self.env)
        aim_vector  = v.Vector(self.prev_gripper_position, self.prev_poker_position, self.env)
        return poker_position,gripper_position,poke_vector,aim_vector

    def finish(self, observation, poker_position, gripper_position, align, len, reward):
        self.prev_poker_position   = poker_position
        self.prev_gripper_position = gripper_position
        self.last_align            = align
        self.last_len              = len

        if self.task.check_object_moved(self.env.task_objects[1], 2):
            self.env.episode_over   = True
            self.env.episode_failed = True
            self.env.episode_info   = "too strong poke"
        self.task.check_poke_threshold(observation)

        self.rewards_history.append(reward)

    def initialize_positions(self, poker_position, gripper_position):
        # make sure none is None
        if self.prev_poker_position[0] is None:
            self.prev_poker_position   = poker_position

        if self.prev_gripper_position[0] is None:
            self.prev_gripper_position = gripper_position

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = v.Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = v.Vector([0,0,0], gripper_position, self.env)
        return direction_vector.add_vector(gripper)

    def is_poker_moving(self, poker):
        if round(self.prev_poker_position[0], 4) == round(poker[0], 4) and round(self.prev_poker_position[1], 4) == round(poker[1], 4) and round(self.prev_poker_position[1], 4) == round(poker[1], 4):
            return False
        return True

    def decide(self, observation=None):

        observation = observation["observation"] if isinstance(observation, dict) else observation
        observation = observation[0]
        # load positions
        goal_position    = observation[0:3]
        poker_position   = observation[3:6]
        gripper_position = observation[6:9]

        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)

        direction_vector = v.Vector([0, 0, 0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3],
                       gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])

        direction_vector.rotate_with_matrix(m)
        
        gripper          = v.Vector([0, 0, 0], gripper_position, self.env)
        gripper_position = direction_vector.add_vector(gripper)

        if self.prev_poker_position[0] is None:
            self.prev_poker_position = poker_position

        if self.prev_gripper_position[0] is None:
            self.prev_gripper_position = gripper_position

        # align
        poke_vector = v.Vector(self.prev_poker_position, goal_position, self.env)
        aim_vector  = v.Vector(self.prev_gripper_position, self.prev_poker_position, self.env)

        self.prev_gripper_position = gripper_position
        self.prev_poker_position   = poker_position
        
        align = np.dot(self.set_vector_len(poke_vector.vector, 1), self.set_vector_len(aim_vector.vector, 1))

        if align > 9.5:
            owner = 1
        else:
            owner = 0

        return owner

    def set_vector_len(self, vector, len):
        norm    = self.count_vector_norm(vector)
        vector  = self.multiply_vector(vector, 1/norm)

        return self.multiply_vector(vector, len)

    def multiply_vector(self, vector, multiplier):
        return np.array(vector) * multiplier
    
    def count_vector_norm(self, vector):
        return math.sqrt(np.dot(vector, vector))

class DualPoke(Reward):
    """
    Reward class for reward signal calculation based on distance differences between 2 objects

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task: (object) Task that is being trained, instance of a class TaskModule
    """
    def __init__(self, env, task):
        super(DualPoke, self).__init__(env, task)

        self.prev_poker_position   = [None]*3
        self.prev_gripper_position = [None]*3

        self.last_dist             = 0
        self.last_len              = 0

        self.a = 0
        self.b = 0

        self.measured = False

        self.owner = None

    def reset(self):
        """
        Reset stored value of distance between 2 objects. Call this after the end of an episode.
        """
        self.prev_poker_position   = [None]*3
        self.prev_gripper_position = [None]*3

        self.last_dist             = 0
        self.last_len              = 0

        self.a = 0
        self.b = 0

        self.measured = False

        self.owner = None

    def compute(self, observation=None):
        """
        Compute reward signal based on distance between 2 objects. The position of the objects must be present in observation.

        Params:
            :param observation: (list) Observation of the environment
        Returns:
            :return reward: (float) Reward signal for the environment
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation

        goal_position    = observation[0:3]
        poker_position   = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[6:9])

        if self.prev_poker_position[0] is None:
            self.prev_poker_position   = poker_position

        if self.prev_gripper_position[0] is None:
            self.prev_gripper_position = gripper_position


        gripper_in_XY = [0.0, 0.3, poker_position[2]] # gripper initial position with z == 0
        poker_in_XY = [poker_position[0], poker_position[1]-0.05, poker_position[2]] # gripper initial position with z == 0
        self.env.p.addUserDebugLine(gripper_in_XY, poker_in_XY, lifeTime=0.1)
        abs = self.distance_of_point_from_abscissa(gripper_in_XY, poker_position, gripper_position)

        poker_in_XY = [poker_position[0], poker_position[1]-0.1, poker_position[2]]

        len = self.task.calc_distance(gripper_position, poker_in_XY)
        self.env.p.addUserDebugLine(gripper_position, poker_in_XY, lifeTime=0.1)

        if self.last_len is None:
            self.last_len = len

        if abs < 0.1:
            if not self.measured:
                reward = 0
                self.measured = True
                self.last_dist = self.task.calc_distance(goal_position, poker_position)
            else:
                dist = self.task.calc_distance(goal_position, poker_position)
                reward = 1*round(self.last_dist - dist, 5)
                self.last_dist = dist
            self.a += reward
            self.owner = 0
            # poke_vector = v.Vector(self.prev_poker_position, observation[0:3], self.env)    
            # real_vector = v.Vector(self.prev_gripper_position, gripper_position, self.env)
            # align  = np.dot(self.set_vector_len(poke_vector.vector, 1), self.set_vector_len(real_vector.vector, 1))
            # reward = align
        else:
            if self.owner == 0:
              self.last_len = len
            if not self.is_poker_moving(poker_position):
                reward = self.last_len - len
                self.b += reward
            else:
                reward = 0
                self.b += reward
            self.owner = 1
        # if self.env.episode_steps > 25:
        #     if self.is_poker_moving(poker_position):
        #         reward = 0
        # el
        if self.env.episode_steps < 2:
            self.a = 0
            self.b = 0
            reward = 0

        self.finish(observation, poker_position, gripper_position, len, reward)

        # self.env.p.addUserDebugText("Reward: " + str(round(self.env.episode_reward, 7)), [0.7,0.7,0.7], lifeTime=0.1)
        # self.env.p.addUserDebugText("poke reward: " + str(round(self.a, 7)), [-0.5,0.5,0.5], lifeTime=0.1)
        # self.env.p.addUserDebugText("align reward: " + str(round(self.b, 7)), [-0.7,0.7,0.7], lifeTime=0.1)
        return reward

    def init(self, observation):
        # load positions
        goal_position    = observation[0:3]
        poker_position   = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[6:9])

        self.initialize_positions(poker_position, gripper_position)

        return poker_position,gripper_position

    def finish(self, observation, poker_position, gripper_position, len, reward):
        self.prev_poker_position   = poker_position
        self.prev_gripper_position = gripper_position
        self.last_len              = len

        if self.task.check_object_moved(self.env.task_objects[1], 2):
            self.env.episode_over   = True
            self.env.episode_failed = True
            self.env.episode_info   = "too strong poke"
        self.task.check_poke_threshold(observation)

        self.rewards_history.append(reward)

    def initialize_positions(self, poker_position, gripper_position):
        # make sure none is None
        if self.prev_poker_position[0] is None:
            self.prev_poker_position   = poker_position

        if self.prev_gripper_position[0] is None:
            self.prev_gripper_position = gripper_position

    def get_accurate_gripper_position(self, gripper_position):
        gripper_orientation = self.env.p.getLinkState(self.env.robot.robot_uid, self.env.robot.end_effector_index)[1]
        gripper_matrix      = self.env.p.getMatrixFromQuaternion(gripper_orientation)
        direction_vector    = v.Vector([0,0,0], [0, 0, 0.1], self.env)
        m = np.array([[gripper_matrix[0], gripper_matrix[1], gripper_matrix[2]], [gripper_matrix[3], gripper_matrix[4], gripper_matrix[5]], [gripper_matrix[6], gripper_matrix[7], gripper_matrix[8]]])
        direction_vector.rotate_with_matrix(m)
        gripper = v.Vector([0,0,0], gripper_position, self.env)
        return direction_vector.add_vector(gripper)

    def is_poker_moving(self, poker):
        if round(self.prev_poker_position[0], 4) == round(poker[0], 4) and round(self.prev_poker_position[1], 4) == round(poker[1], 4) and round(self.prev_poker_position[1], 4) == round(poker[1], 4):
            return False
        return True

    def decide(self, observation=None):

        observation = observation["observation"] if isinstance(observation, dict) else observation
        observation = observation[0]

        poker_position   = observation[3:6]
        gripper_position = self.get_accurate_gripper_position(observation[6:9])


        gripper_in_XY = [0.0, 0.3, poker_position[2]] # gripper initial position with z == 0
        self.env.p.addUserDebugLine(gripper_in_XY, poker_position, lifeTime=0.1)
        len = self.distance_of_point_from_abscissa(gripper_in_XY, poker_position, gripper_position)

        if len < 0.1:
            owner = 1
        else:
            owner = 0

        return owner

    def set_vector_len(self, vector, len):
        norm    = self.count_vector_norm(vector)
        vector  = self.multiply_vector(vector, 1/norm)

        return self.multiply_vector(vector, len)

    def multiply_vector(self, vector, multiplier):
        return np.array(vector) * multiplier
    
    def calc_direction_3d(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
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

        d = math.sqrt((x - x3) ** 2 + (y - y3) ** 2 + (z - z3) ** 2)
        dot_product = (x-x1)*(x-x2)+(y-y1)*(y-y2)+(z-z1)*(z-z2)
        d1 = (x1-x)**2+(y1-y)**2+(z1-z)**2
        d2 = (x2-x)**2+(y2-y)**2+(z2-z)**2
        if dot_product > 0: # out
            d = min(d1, d2)
        return d

    def count_vector_norm(self, vector):
        return math.sqrt(np.dot(vector, vector))

    def get_distance_from_poke_line(self, poke_vector, poker_position, goal_position):
        z = self.prev_gripper_position[2]

        a = self.count_vector_norm(poke_vector.vector)
        b = math.sqrt((self.prev_gripper_position[0]-poker_position[0])**2 + (self.prev_gripper_position[1]-poker_position[1])**2)
        c = math.sqrt((self.prev_gripper_position[0]-goal_position[0])**2 + (self.prev_gripper_position[1]-goal_position[1])**2)

        if a < 0.1:
            print("_____")
            self.env.episode_info = "Successfull poke"
            self.env.episode_over = True

        a = round(a, 5)
        b = round(b, 5)
        c = round(c, 5)

        while b+c <= a:
            c += 0.00001

        while a+c <= b:
            c += 0.00001

        while a+b <= c:
            b += 0.00001

        distanceXY = math.sqrt(self.triangle_height(a, b, c))
        distanceZ  = z-0.0 # moving the line into height of center of the cubes
        distance   = math.sqrt(distanceZ**2+distanceXY**2)

        return distance

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