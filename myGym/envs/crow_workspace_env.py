from myGym.envs import robot, env_object
from myGym.envs import task as t
from myGym.envs.base_env import CameraEnv
from myGym.envs.gym_env import GymEnv
from myGym.envs.rewards import DistanceReward, ComplexDistanceReward, SparseReward
import pybullet
import time
import numpy as np
import math
from gym import spaces
import os
import inspect
import random
import pkg_resources
currentdir = pkg_resources.resource_filename("myGym", "envs")
repodir = pkg_resources.resource_filename("myGym", "")
print("current_dir=" + currentdir)


class CrowWorkspaceEnv(GymEnv):
    """
    Environment class for particular environment based on myGym basic environment classes

    Parameters:
        :param object_sampling_area: (list of floats) Volume in the scene where objects can appear ([x,x,y,y,z,z])
        :param dimension_velocity: (float) Maximum allowed velocity for robot movements in individual x,y,z axis
        :param used_objects: (list of strings) Names of extra objects (not task objects) that can appear in the scene
        :param num_objects_range: (list of ints) Minimum and maximum number of extra objects that may appear in the scene
        :param action_repeat: (int) Amount of steps the robot takes between the simulation steps (updates)
        :param color_dict: (dict) Dictionary of specified colors for chosen objects, Key: object name, Value: RGB color
        :param robot: (string) Type of robot to train in the environment (kuka, panda, ur3, ...)
        :param robot_position: (list) Position of the robot's base link in the coordinate frame of the environment ([x,y,z])
        :param robot_orientation: (list) Orientation of the robot's base link in the coordinate frame of the environment (Euler angles [x,y,z])
        :param robot_action: (string) Mechanism of robot control (absolute, step, joints)
        :param robot_init_joint_poses: (list) Configuration in which robot will be initialized in the environment. Specified either in joint space as list of joint poses or in the end-effector space as [x,y,z] coordinates.
        :param task_type: (string) Type of learned task (reach, push, ...)
        :param task_objects: (list of strings) Objects that are relevant for performing the task
        :param reward_type: (string) Type of reward signal source (gt, 3dvs, 2dvu)
        :param reward: (string) Defines how to compute the reward
        :param distance_type: (string) Way of calculating distances (euclidean, manhattan)
        :param active_cameras: (int) The number of camera used to render and record images
        :param dataset: (bool) Whether this environment serves for image dataset generation
        :param obs_space: (string) Type of observation data. Specify "dict" for usage of HER training algorithm.
        :param visualize: (bool) Whether to show helping visualization during training and testing of environment
        :param logdir: (string) Directory for logging
        :param vae_path: (string) Path to a trained VAE in 2dvu reward type
        :param yolact_path: (string) Path to a trained Yolact in 3dvu reward type
        :param yolact_config: (string) Path to saved Yolact config obj or name of an existing one in the data/Config script or None for autodetection
    """
    def __init__(self,
                 object_sampling_area = [-0.7, 0.7, 0.3, 0.9, 1, 1],
                 robot_position=[0.0, 0.97, 1.05],
                 robot_orientation=[0, 0, -0.5*np.pi],
                 **kwargs
                 ):

        self.robot_position = robot_position
        self.robot_orientation = robot_orientation
        if object_sampling_area is None:
            object_sampling_area = [-0.7, 0.7, 0.3, 0.9, 1, 1]
        super(CrowWorkspaceEnv, self).__init__(object_sampling_area=object_sampling_area, robot_position=robot_position, robot_orientation=robot_orientation, **kwargs)

    def _setup_scene(self):
        """
        Set-up environment scene. Load static objects, apply textures. Load robot.
        """
        self._add_scene_object_uid(self.p.loadURDF(os.path.join(
            self.urdf_root, "plane.urdf"), [0, 0, 0]), "floor")

        # Uncomment to load a spaceship
        #self._add_scene_object_uid(self.p.loadURDF(
        #     pkg_resources.resource_filename("myGym", "/envs/rooms/corridor.urdf"),
        #                                     [0.0, 0.0, 0.0], useMaximalCoordinates=True), "sci_fi_corridor")

        # corridor_texture_id = self.p.loadTexture(
        #     pkg_resources.resource_filename("myGym", "envs/rooms/wall.png"))
        #
        # self.p.changeVisualShape(self.get_scene_object_uid_by_name("sci_fi_corridor"), -1,
        #                          rgbaColor=[1, 1, 1, 1], textureUniqueId=corridor_texture_id)

        floor_texture_id = self.p.loadTexture(
            pkg_resources.resource_filename("myGym", "/envs/textures/texture_7.png"))

        if self.reward_type != "2dvu":
             self.p.changeVisualShape(self.get_scene_object_uid_by_name("floor"), -1,
                                         rgbaColor=[1, 1, 1, 1], textureUniqueId=floor_texture_id)

        self._add_scene_object_uid(self.p.loadURDF(
            pkg_resources.resource_filename("myGym", "/envs/rooms/collision/table_crow.urdf"),
                                            [0.0, 0.0, 0.0]), "table")
        table_texture_id = self.p.loadTexture(
            pkg_resources.resource_filename("myGym", "/envs/textures/texture_2.png"))
        if self.reward_type != "2dvu":
            self.p.changeVisualShape(self.get_scene_object_uid_by_name("table"), -1,
                                     rgbaColor=[1, 1, 1, 1], textureUniqueId=table_texture_id)
        self.robot = robot.Robot(self.robot_type,
                                 position=self.robot_position,
                                 orientation=self.robot_orientation,
                                 init_joint_poses=self.robot_init_joint_poses,
                                 robot_action=self.robot_action,
                                 dimension_velocity=self.dimension_velocity,
                                 pybullet_client=self.p)

    def _set_observation_space(self):
        """
        Set observation space type, dimensions and range
        """
        if self.obs_space == "dict":
            goaldim = int(self.task.obsdim/2) if self.task.obsdim % 2 == 0 else int(self.task.obsdim / 3)
            self.observation_space = spaces.Dict({"observation": spaces.Box(low=-10, high=10, shape=(self.task.obsdim,)),
                                                  "achieved_goal": spaces.Box(low=-10, high=10, shape=(goaldim,)),
                                                  "desired_goal": spaces.Box(low=-10, high=10, shape=(goaldim,))})
        else:
            observationDim = self.task.obsdim
            observation_high = np.array([100] * observationDim)
            self.observation_space = spaces.Box(-observation_high,
                                                observation_high)

    def _set_action_space(self):
        """
        Set action space dimensions and range
        """
        action_dim = self.robot.get_action_dimension()
        if self.robot_action == "step":
            self.action_low = np.array([-1] * action_dim)
            self.action_high = np.array([1] * action_dim)
        elif self.robot_action == "absolute":
            self.action_low = np.array(self.objects_area_boarders[0:7:2])
            self.action_high = np.array(self.objects_area_boarders[1:7:2])
        else:
            self.action_low = np.array(self.robot.joints_limits[0])
            self.action_high = np.array(self.robot.joints_limits[1])
        self.action_space = spaces.Box(np.array([-1]*action_dim), np.array([1]*action_dim))

    def _rescale_action(self, action):
        """
        Rescale action returned by trained model to fit environment action space

        Parameters:
            :param action: (list) Action data returned by trained model
        Returns:
            :return rescaled_action: (list) Rescaled action to environment action space
        """
        return [(sub + 1) * (h - l) / 2 + l for sub, l, h in zip(action, self.action_low, self.action_high)]

    def reset(self, random_pos=True, hard=False, random_robot=False):
        """
        Environment reset called at the beginning of an episode. Reset state of objects, robot, task and reward.

        Parameters:
            :param random_pos: (bool) Whether to initiate objects to random locations in the scene
            :param hard: (bool) Whether to do hard reset (resets whole pybullet scene)
            :param random_robot: (bool) Whether to initiate robot in random pose
        Returns:
            :return self._observation: (list) Observation data of the environment
        """
        super().reset(hard=hard)

        self.robot.reset(random_robot=random_robot)
        self.env_objects = []
        self.task_objects = []
        if self.used_objects is not None:
            if self.num_objects_range is not None:
                num_objects = int(np.random.uniform(self.num_objects_range[0], self.num_objects_range[1]))
            else:
                num_objects = int(np.random.uniform(0, len(self.used_objects)))
            self.env_objects = self._randomly_place_objects(num_objects, self.used_objects, random_pos)
        for obj_name in self.task_objects_names:
            self.task_objects.append(self._randomly_place_objects(1, [obj_name], random_pos)[0])
        self.env_objects += self.task_objects
        self.task.reset_task()
        self.reward.reset()
        self.p.stepSimulation()
        self._observation = self.get_observation()
        self.prev_gripper_position = self.robot.get_observation()[:3]
        return self._observation

    def _set_cameras(self):
        """
        Add cameras to the environment
        """
        self.add_camera(position=[0.0, -1.4, 2.0], target_position=[0.0, 1, 1.7], distance=0.001, is_absolute_position=True)
        self.add_camera(position=[0.0, 2.0, 2.2], target_position=[0.0, 0.8, 1.7], distance=0.001, is_absolute_position=True)
        self.add_camera(position=[-1.7, 0.6, 2.2], target_position=[-1.6, 0.6, 2.155], distance=0.001, is_absolute_position=True)
        self.add_camera(position=[1.7, 0.6, 2.2], target_position=[1.6, 0.6, 2.155], distance=0.001, is_absolute_position=True)
        self.add_camera(position=[0.0, 0.7, 2.1], target_position=[0.0, 0.65, 1.5], distance=0.001, is_absolute_position=True)

    def get_observation(self):
        """
        Get observation data from the environment

        Returns:
            :return observation: (array) Represented position of task relevant objects
        """
        if self.obs_space == "dict":
            observation = self.task.get_observation()
            if self.reward_type != "2dvu":
                self._observation = {"observation": observation,
                                     "achieved_goal": observation[0:3],
                                     "desired_goal": observation[3:6]}
            else:
                self._observation = {"observation": self.task.get_observation(),
                                     "achieved_goal": observation[0:int(len(observation)/2)],
                                     "desired_goal": observation[int(len(observation)/2):]}
            return self._observation
        else:
            self.observation["task_objects"] = self.task.get_observation()
            if self.dataset:
                self.observation["camera_data"] = self.render(mode="rgb_array")
                self.observation["objects"] = self.env_objects
                return self.observation
            else:
                return self.observation["task_objects"]

    def step(self, action):
        """
        Environment step in simulation

        Parameters:
            :param actions: (list) Action data to send to robot to perform movement in this step
        Returns:
            :return self._observation: (list) Observation data from the environment after this step
            :return reward: (float) Reward value assigned to this step
            :return done: (bool) Whether this stop is episode's final
            :return info: (dict) Additional information about step
        """
        action = self._rescale_action(action)
        self._apply_action_robot(action)

        self._observation = self.get_observation()
        if self.dataset:
            reward = 0
            done = False
            info = {}
        else:
            reward = self.reward.compute(observation=self._observation)
            self.episode_reward += reward
            self.task.check_goal()
            done = self.episode_over
            info = {'d': self.task.last_distance / self.task.init_distance,
                    'f': int(self.episode_failed)}

        if done:
            self.episode_final_reward.append(self.episode_reward)
            self.episode_final_distance.append(self.task.last_distance / self.task.init_distance)
            self.episode_number += 1
            self._print_episode_summary(info)

        return self._observation, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        #@TODO: Reward computation for HER, argument for .compute()
        reward = self.reward.compute(np.append(achieved_goal, desired_goal))
        return reward

    def _apply_action_robot(self, action):
        """
        Apply desired action to robot in simulation

        Parameters:
            :param action: (list) Action data returned by trained model
        """
        for i in range(self.action_repeat):
            self.robot.apply_action(action)
            self.p.stepSimulation()
            self.episode_steps += 1

    def draw_bounding_boxes(self):
        """
        Show bounding box of object in the scene in GUI
        """
        for object in self.env_objects:
            object.draw_bounding_box()

    def _randomly_place_objects(self, n, object_names=None, random_pos=True):
        """
        Place dynamic objects to the scene randomly

        Parameters:
            :param n: (int) Number of objects to place in the scene
            :param object_names: (list of strings) Objects that may be placed to the scene
            :param random_pos: (bool) Whether to place object to random positions in the scene
        Returns:
            :return env_objects: (list of objects) Objects that are present in the current scene
        """
        env_objects = []
        objects_filenames = self._get_random_urdf_filenames(n, object_names)
        for object_filename in objects_filenames:
            if random_pos:
                pos = env_object.EnvObject.get_random_object_position(
                    self.objects_area_boarders)
                #orn = env_object.EnvObject.get_random_object_orientation()
                orn = [0, 0, 0, 1]
                object = env_object.EnvObject(object_filename, pos, orn, pybullet_client=self.p)
            else:
                object = env_object.EnvObject(
                    object_filename, [0.6, 0.6, 1.2], [0, 0, 0, 1], pybullet_client=self.p)
            if self.color_dict:
                object.set_color(self.color_of_object(object))
            env_objects.append(object)
        return env_objects

    def color_of_object(self, object):
        """
        Set object's color

        Parameters:
            :param object: (object) Object
        Returns:
            :return color: (list) RGB color
        """
        if object.name not in self.color_dict:
            return env_object.EnvObject.get_random_color()
        else:
            color = self.color_dict[object.name].copy()
            color = random.sample(self.color_dict[object.name], 1)
            color[:] = [x / 255 for x in color[0]]
            color.append(1)
        return color
