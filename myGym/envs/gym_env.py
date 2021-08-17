from myGym.envs import robot, env_object
from myGym.envs import task as t
from myGym.envs import distractor as d
from myGym.envs.base_env import CameraEnv
from myGym.envs.rewards import DistanceReward, ComplexDistanceReward, SparseReward, VectorReward, PokeReward, PokeVectorReward, PokeReachReward, SwitchReward, ButtonReward
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


class GymEnv(CameraEnv):
    """
    Environment class for particular environment based on myGym basic environment classes

    Parameters:
        :param workspace: (string) Workspace in gym, where training takes place (collabtable, maze, drawer, ...)
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
        :param num_subgoals: (int) Number of subgoals in task
        :param task_objects: (list of strings) Objects that are relevant for performing the task

        :param distractors: (list of strings) Objects distracting from performed task
        :param distractor_moveable: (bool) can distractors move
        :param distractor_constant_speed: (bool) is speed of distractors constant
        :param distractor_movement_dimensions: (int) number of dimensions of distractors motion
        :param distractor_movement_endpoints: (list of floats) borders of dostractors movement

        :param reward_type: (string) Type of reward signal source (gt, 3dvs, 2dvu)
        :param reward: (string) Defines how to compute the reward
        :param distance_type: (string) Way of calculating distances (euclidean, manhattan)
        :param active_cameras: (int) The number of camera used to render and record images
        :param dataset: (bool) Whether this environment serves for image dataset generation
        :param obs_space: (string) Type of observation data. Specify "dict" for usage of HER training algorithm.
        :param visualize: (bool) Whether to show helping visualization during training and testing of environment
        :param visgym: (bool) Whether to visualize whole gym building and other tasks (final visualization)
        :param logdir: (string) Directory for logging
        :param vae_path: (string) Path to a trained VAE in 2dvu reward type
        :param yolact_path: (string) Path to a trained Yolact in 3dvu reward type
        :param yolact_config: (string) Path to saved Yolact config obj or name of an existing one in the data/Config script or None for autodetection
    """
    def __init__(self,
                 workspace="table",
                 object_sampling_area=None,
                 dimension_velocity=0.05,
                 used_objects=None,
                 num_objects_range=None,
                 action_repeat=1,
                 color_dict=None,
                 robot='kuka',
                 robot_position=[0.0, 0.0, 0.0],
                 robot_orientation=[0, 0, 0],
                 robot_action="step",
                 robot_init_joint_poses=[],
                 task_type='reach',
                 num_subgoals=0,
                 task_objects=["virtual_cube_holes"],
                 
                 distractors=None,
                 distractor_moveable=0,
                 distractor_movement_endpoints=[-0.7+0.12, 0.7-0.15], 
                 distractor_constant_speed=1, 
                 distractor_movement_dimensions=2,
                 observed_links_num=11,

                 reward_type='gt',
                 reward = 'distance',
                 distance_type='euclidean',
                 active_cameras=None,
                 dataset=False,
                 obs_space=None,
                 visualize=0,
                 visgym=1,
                 logdir=None,
                 vae_path=None,
                 yolact_path=None,
                 yolact_config=None,
                 **kwargs
                 ):

        self.workspace              = workspace
        self.robot_type             = robot
        self.robot_position         = robot_position
        self.robot_orientation      = robot_orientation
        self.robot_init_joint_poses = robot_init_joint_poses
        self.robot_action           = robot_action
        self.dimension_velocity     = dimension_velocity
        self.active_cameras         = active_cameras

        self.objects_area_boarders  = object_sampling_area
        self.used_objects           = used_objects
        self.action_repeat          = action_repeat
        self.num_objects_range      = num_objects_range
        self.color_dict             = color_dict

        self.observed_links_num     = observed_links_num

        self.task_type              = task_type
        if dataset:
            task_objects = []
        self.task_objects_names     = task_objects

        self.has_distractor         = False if distractors == None else True
        self.distractors            = distractors

        self.reward_type            = reward_type
        self.distance_type          = distance_type

        self.task = t.TaskModule(task_type=self.task_type,
                                 num_subgoals=num_subgoals,
                                 task_objects=self.task_objects_names,
                                 reward_type=self.reward_type,
                                 vae_path=vae_path,
                                 yolact_path=yolact_path,
                                 yolact_config=yolact_config,
                                 distance_type=self.distance_type,
                                 env=self)
        
        self.dist = d.DistractorModule(distractor_moveable,
                                       distractor_movement_endpoints, 
                                       distractor_constant_speed, 
                                       distractor_movement_dimensions,
                                       env=self)  

        if reward == 'distance':
            self.reward = DistanceReward(env=self, task=self.task)
        elif reward == "complex_distance":
            self.reward = ComplexDistanceReward(env=self, task=self.task)
        elif reward == 'sparse':
            self.reward = SparseReward(env=self, task=self.task)
        elif reward == 'distractor':
            self.has_distractor = True
            if self.distractors == None:
                self.distractor = ['bus']
            self.reward = VectorReward(env=self, task=self.task)
        elif reward == 'poke':
            self.reward = PokeReachReward(env=self, task=self.task)
            # self.reward = PokeVectorReward(env=self, task=self.task)
            # self.reward = PokeReward(env=self, task=self.task)
        elif reward == 'switch':
            print("switch")
            self.reward = SwitchReward(env=self, task=self.task)
        elif reward == 'btn':
            print("btn")
            self.reward = ButtonReward(env=self, task=self.task)
        self.dataset = dataset
        self.obs_space = obs_space
        self.visualize = visualize
        self.visgym = visgym
        self.logdir = logdir
        self.workspace_dict =  {'baskets':  {'urdf': 'baskets.urdf', 'texture': 'baskets.jpg',
                                            'transform': {'position':[3.18, -3.49, -1.05], 'orientation':[0.0, 0.0, -0.4*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[0.56, -1.71, 0.6], [-1.3, 3.99, 0.6], [-3.43, 0.67, 1.0], [2.76, 2.68, 1.0], [-0.54, 1.19, 3.4]], 
                                                        'target': [[0.53, -1.62, 0.59], [-1.24, 3.8, 0.55], [-2.95, 0.83, 0.8], [2.28, 2.53, 0.8], [-0.53, 1.2, 3.2]]},
                                            'boarders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]}, 
                                'collabtable': {'urdf': 'collabtable.urdf', 'texture': 'collabtable.jpg', 
                                            'transform': {'position':[0.45, -5.1, -1.05], 'orientation':[0.0, 0.0, -0.35*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[-0.25, 3.24, 1.2], [-0.44, -1.34, 1.0], [-1.5, 2.6, 1.0], [1.35, -1.0, 1.0], [-0.1, 1.32, 1.4]], 
                                                        'target': [[-0.0, 0.56, 0.6], [-0.27, 0.42, 0.7], [-1, 2.21, 0.8], [-0.42, 2.03, 0.2], [-0.1, 1.2, 0.7]]},
                                            'boarders':[-0.7, 0.7, 0.5, 1.2, 0.2, 0.2]}, 
                                'darts':    {'urdf': 'darts.urdf', 'texture': 'darts.jpg', 
                                            'transform': {'position':[-1.4, -6.7, -1.05], 'orientation':[0.0, 0.0, -1.0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[-0.0, 2.1, 1.0], [0.0, -1.5, 1.2], [2.3, 0.5, 1.0], [-2.6, 0.5, 1.0], [-0.0, 1.1, 4.9]], 
                                                        'target': [[0.0, 0.0, 0.7], [-0.0, 1.3, 0.6], [1.0, 0.9, 0.9], [-1.6, 0.9, 0.9], [-0.0, 1.2, 3.1]]},
                                            'boarders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]}, 
                                'drawer':   {'urdf': 'drawer.urdf', 'texture': 'drawer.jpg', 
                                            'transform': {'position':[-4.81, 1.75, -1.05], 'orientation':[0.0, 0.0, 0.0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0, 0, 0.5*np.pi]}, 
                                            'camera': {'position': [[-0.14, -1.63, 1.0], [-0.14, 3.04, 1.0], [-1.56, -0.92, 1.0], [1.2, -1.41, 1.0], [-0.18, 0.88, 2.5]], 
                                                        'target': [[-0.14, -0.92, 0.8], [-0.14, 2.33, 0.8], [-0.71, -0.35, 0.7], [0.28, -0.07, 0.6], [-0.18, 0.84, 2.1]]},
                                            'boarders':[-0.7, 0.7, 0.4, 1.3, 0.8, 0.1]}, 
                                'football': {'urdf': 'football.urdf', 'texture': 'football.jpg', 
                                            'transform': {'position':[4.2, -5.4, -1.05], 'orientation':[0.0, 0.0, -1.0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[-0.0, 2.1, 1.0], [0.0, -1.7, 1.2], [3.5, -0.6, 1.0], [-3.5, -0.7, 1.0], [-0.0, 2.0, 4.9]], 
                                                        'target': [[0.0, 0.0, 0.7], [-0.0, 1.3, 0.2], [3.05, -0.2, 0.9], [-2.9, -0.2, 0.9], [-0.0, 2.1, 3.6]]},
                                            'boarders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]}, 
                                'fridge':   {'urdf': 'fridge.urdf', 'texture': 'fridge.jpg',
                                            'transform': {'position':[1.6, -5.95, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[0.0, -1.3, 1.0], [0.0, 2.35, 1.2], [-1.5, 0.85, 1.0], [1.4, 0.85, 1.0], [0.0, 0.55, 2.5]], 
                                                        'target': [[0.0, 0.9, 0.7], [0.0, 0.9, 0.6], [0.0, 0.55, 0.5], [0.4, 0.55, 0.7], [0.0, 0.45, 1.8]]},
                                            'boarders':[-0.7, 0.7, 0.3, 0.5, -0.9, -0.9]}, 
                                'maze':     {'urdf': 'maze.urdf', 'texture': 'maze.jpg', 
                                            'transform': {'position':[6.7, -3.1, 0.0], 'orientation':[0.0, 0.0, -0.5*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.1], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[0.0, -1.4, 2.3], [-0.0, 5.9, 1.9], [4.7, 2.7, 2.0], [-3.2, 2.7, 2.0], [-0.0, 3.7, 5.0]], 
                                                        'target': [[0.0, -1.0, 1.9], [-0.0, 5.6, 1.7], [3.0, 2.7, 1.5], [-2.9, 2.7, 1.7], [-0.0, 3.65, 4.8]]},
                                            'boarders':[-2.5, 2.2, 0.7, 4.7, 0.05, 0.05]}, 
                                'stairs':   {'urdf': 'stairs.urdf', 'texture': 'stairs.jpg',
                                            'transform': {'position':[-5.5, -0.08, -1.05], 'orientation':[0.0, 0.0, -0.20*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[0.04, -1.64, 1.0], [0.81, 3.49, 1.0], [-2.93, 1.76, 1.0], [4.14, 0.33, 1.0], [2.2, 1.24, 3.2]], 
                                                        'target': [[0.18, -1.12, 0.85], [0.81, 2.99, 0.8], [-1.82, 1.57, 0.7], [3.15, 0.43, 0.55], [2.17, 1.25, 3.1]]},
                                            'boarders':[-0.5, 2.5, 0.8, 1.6, 0.1, 0.1]},
                                'table':    {'urdf': 'table.urdf', 'texture': 'table.jpg', 
                                            'transform': {'position':[-0.0, -0.0, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0.0, 0.9, 1.3],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]], 
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.898, 1.28],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'boarders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]}, 
                                'verticalmaze': {'urdf': 'verticalmaze.urdf', 'texture': 'verticalmaze.jpg',
                                            'transform': {'position':[-5.7, -7.55, -1.05], 'orientation':[0.0, 0.0, 0.5*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[-0.0, -1.25, 1.0], [0.0, 1.35, 1.3], [1.7, -1.25, 1.0], [-1.6, -1.25, 1.0], [0.0, 0.05, 2.5]], 
                                                        'target': [[-0.0, -1.05, 1.0], [0.0, 0.55, 1.3], [1.4, -0.75, 0.9], [-1.3, -0.75, 0.9], [0.0, 0.15, 2.1]]},
                                            'boarders':[-0.7, 0.8, 0.65, 0.65, 0.7, 1.4]}  }
        super(GymEnv, self).__init__(active_cameras=active_cameras, **kwargs)

    def _setup_scene(self):
        """
        Set-up environment scene. Load static objects, apply textures. Load robot.
        """
        transform = self.workspace_dict[self.workspace]['transform']
        # Floor
        self._add_scene_object_uid(self.p.loadURDF(pkg_resources.resource_filename("myGym", "/envs/rooms/plane.urdf"),
                                        transform['position'],self.p.getQuaternionFromEuler(transform['orientation']),useFixedBase=True, useMaximalCoordinates=True), "floor")
        # Visualize gym background objects
        if self.visgym:
            self._add_scene_object_uid(self.p.loadURDF(
                    pkg_resources.resource_filename("myGym", "/envs/rooms/room.urdf"),
                                                transform['position'],self.p.getQuaternionFromEuler(transform['orientation']),useFixedBase=True, useMaximalCoordinates=True), "gym")
            for workspace in self.workspace_dict:
                if workspace != self.workspace:
                    self._add_scene_object_uid(self.p.loadURDF(
                        pkg_resources.resource_filename("myGym", "/envs/rooms/visual/"+self.workspace_dict[workspace]['urdf']),
                                                    transform['position'],self.p.getQuaternionFromEuler(transform['orientation']),useFixedBase=True, useMaximalCoordinates=True), workspace)
        # Load selected workspace
        self._add_scene_object_uid(self.p.loadURDF(
            pkg_resources.resource_filename("myGym", "/envs/rooms/collision/"+self.workspace_dict[self.workspace]['urdf']),
                                            transform['position'],self.p.getQuaternionFromEuler(transform['orientation']),useFixedBase=True, useMaximalCoordinates=True), self.workspace)
        # Add textures
        if self.task.reward_type != "2dvu":
             if self.workspace_dict[self.workspace]['texture'] is not None:
                 workspace_texture_id = self.p.loadTexture(
                     pkg_resources.resource_filename("myGym", "/envs/textures/"+self.workspace_dict[self.workspace]['texture']))
                 self.p.changeVisualShape(self.get_scene_object_uid_by_name(self.workspace), -1,
                                             rgbaColor=[1, 1, 1, 1], textureUniqueId=workspace_texture_id)
        else:
             workspace_texture_id = self.p.loadTexture(pkg_resources.resource_filename("myGym",
                                                "/envs/textures/grey.png"))
        self.p.changeVisualShape(self.get_scene_object_uid_by_name(self.workspace), -1,
                                     rgbaColor=[1, 1, 1, 1], textureUniqueId=workspace_texture_id)
        floor_texture_id = self.p.loadTexture(
            pkg_resources.resource_filename("myGym", "/envs/textures/parquet1.jpg"))
        self.p.changeVisualShape(self.get_scene_object_uid_by_name("floor"), -1,
                                         rgbaColor=[1, 1, 1, 1], textureUniqueId=floor_texture_id)
        # Define boarders
        if self.objects_area_boarders is None:
            self.objects_area_boarders = self.workspace_dict[self.workspace]['boarders']
        # Add robot
        self.robot = robot.Robot(self.robot_type,
                                 position=self.workspace_dict[self.workspace]['robot']['position'],
                                 orientation=self.workspace_dict[self.workspace]['robot']['orientation'],
                                 init_joint_poses=self.robot_init_joint_poses,
                                 robot_action=self.robot_action,
                                 dimension_velocity=self.dimension_velocity,
                                 pybullet_client=self.p)
        # Add human                         
        if self.workspace == 'collabtable':
            self.human = robot.Robot('human',
                                 position=self.workspace_dict[self.workspace]['robot']['position'],
                                 orientation=self.workspace_dict[self.workspace]['robot']['orientation'],
                                 init_joint_poses=self.robot_init_joint_poses,
                                 robot_action='joints',
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

            if self.has_distractor:
                observationDim = (len(self.task_objects_names) + self.observed_links_num + len(self.distractors) + 1) * 3

            observation_high = np.array([100] * observationDim)
            self.observation_space = spaces.Box(-observation_high,
                                                observation_high)

    def _set_action_space(self):
        """
        Set action space dimensions and range
        """
        action_dim = self.robot.get_action_dimension()
        if self.robot_action in ["step", "joints_step"]:
            self.action_low = np.array([-1] * action_dim)
            self.action_high = np.array([1] * action_dim)
        elif self.robot_action == "absolute":
            if any(isinstance(i, list) for i in self.objects_area_boarders):
                boarders_max = np.max(self.objects_area_boarders,0)
                boarders_min = np.min(self.objects_area_boarders,0)
                self.action_low = np.array(boarders_min[0:7:2])
                self.action_high = np.array(boarders_max[1:7:2])
            else:
                self.action_low = np.array(self.objects_area_boarders[0:7:2])
                self.action_high = np.array(self.objects_area_boarders[1:7:2])
        elif self.robot_action in ["joints", "joints_gripper"]:
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

        self.env_objects = []
        self.task_objects = []
        if self.used_objects is not None:
            if self.num_objects_range is not None:
                num_objects = int(np.random.uniform(self.num_objects_range[0], self.num_objects_range[1]))
            else:
                num_objects = int(np.random.uniform(0, len(self.used_objects)))
            self.env_objects = self._randomly_place_objects(num_objects, self.used_objects, random_pos)
        if self.task_type == 'push':
            self.task_objects.append(self._randomly_place_objects(1, [self.task_objects_names[0]], random_pos=False, pos=[0.0, 0.5, 0.05])[0])
            self.task_objects.append(self._randomly_place_objects(1, [self.task_objects_names[1]], random_pos=True, pos=[-0.0, 0.9, 0.05])[0])
            direction = np.array(self.task_objects[0].get_position()) - np.array(self.task_objects[1].get_position())
            direction = direction/(10*np.linalg.norm(direction))
            init_joint_poses = np.array(self.task_objects[0].get_position()) + direction
            init_joint_poses = [0, 0.42, 0.15]            
            self.robot.init_joint_poses = list(self.robot._calculate_accurate_IK(init_joint_poses))
        else:
            for obj_name in self.task_objects_names:
                self.task_objects.append(self._randomly_place_objects(1, [obj_name], random_pos)[0])

            if self.has_distractor:
                for distractor in self.distractors:
                    self.task_objects.append(self.dist.place_distractor(distractor, self.p))

        self.env_objects += self.task_objects
        self.robot.reset(random_robot=random_robot)
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
        camera_args = self.workspace_dict[self.workspace]['camera']
        for cam_idx in range(len(camera_args['position'])):
            self.add_camera(position=camera_args['position'][cam_idx], target_position=camera_args['target'][cam_idx], distance=0.001, is_absolute_position=True)

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

        if self.has_distractor:
            for distractor in self.distractors:
                self.dist.execute_distractor_step(distractor)

        self._observation = self.get_observation()
        if self.dataset:
            reward = 0
            done = False
            info = {}
        else:
            reward = self.reward.compute(observation=self._observation)
            if reward is None:
                reward = 0
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
            if hasattr(self, 'human'):
                self.human.apply_action(np.random.uniform(self.human.joints_limits[0], self.human.joints_limits[1]))
            self.p.stepSimulation()
            self.episode_steps += 1

    def draw_bounding_boxes(self):
        """
        Show bounding box of object in the scene in GUI
        """
        for object in self.env_objects:
            object.draw_bounding_box()

    def _randomly_place_objects(self, n, object_names=None, random_pos=True, pos=[0.6,0.6,1.2], orn=[0,0,0,1]):
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
                fixed = False
                borders = self.objects_area_boarders
                if self.task_type == 'poke':
                    if "poke" in object_filename:
                        borders = [0, 0, 0.7, 0.7, 0.1, 0.1]
                    if "cube" in object_filename:
                        borders = [0, 0, 1, 1, 0.1, 0.1]
                        fixed = True

                pos = env_object.EnvObject.get_random_object_position(borders)
                #orn = env_object.EnvObject.get_random_object_orientation()
                orn = [0, 0, 0, 1]
            for x in ["target", "crate", "bin", "box", "trash", "switch", "btn", "steering_wheel"]:
                if x in object_filename:
                    fixed = True
                    pos[2] = 0.05
            object = env_object.EnvObject(object_filename, pos, orn, pybullet_client=self.p, fixed=fixed)
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
