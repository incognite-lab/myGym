import copy
from typing import List, Any
import warnings
from myGym.envs import robot, env_object
from myGym.envs import task as t
from myGym.envs import distractor as d
from myGym.envs.base_env import CameraEnv
from myGym import oraculum
from collections import ChainMap

from myGym.envs.env_object import EnvObject
from myGym.envs.rewards import *
import numpy as np
from itertools import chain
import random

from myGym.utils.helpers import get_workspace_dict
import importlib.resources as pkg_resources
from myGym.envs.human import Human
import myGym.utils.colors as cs
from myGym.utils.helpers import get_module_type
from myGym.envs.natural_language import NaturalLanguage
from myGym.envs.task import TaskModule
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

currentdir = os.path.join(pkg_resources.files("myGym"), "envs")

# used to exclude these colors for other objects, in the order of goal, init, done, else
COLORS_RESERVED_FOR_HIGHLIGHTING = ["dark green", "green", "blue", "gray"]


class GymEnv(CameraEnv):
    """
    Environment class for particular environment based on myGym basic environment classes

    Parameters:
        :param observation: (dict) Template dictionary of what should be part of the network observation
        :param rddl: (dict) Template dictionary for task creation via rddl
        :param workspace: (string) Workspace in gym, where training takes place (collabtable, maze, drawer, ...)
        :param dimension_velocity: (float) Maximum allowed velocity for robot movements in individual x,y,z axis
        :param used_objects: (list of strings) Names of extra objects (not task objects) that can appear in the scene
        :param action_repeat: (int) Amount of steps the robot takes between the simulation steps (updates)
        :param color_dict: (dict) Dictionary of specified colors for chosen objects, Key: object name, Value: RGB color
        :param robot: (string) Type of robot to train in the environment (kuka, panda, ur3, ...)
        :param robot_action: (string) Mechanism of robot control (absolute, step, joints)
        :param robot_init_joint_poses: (list) Configuration in which robot will be initialized in the environment. Specified either in joint space as list of joint poses or in the end-effector space as [x,y,z] coordinates.
        :param task_type: (string) Type of learned task (reach, push, ...)
        :param num_subgoals: (int) Number of subgoals in task
        :param distractors: (dict) Objects distracting from performed task
        :param reward_type: (string) Type of reward signal source (gt, 3dvs, 2dvu)
        :param reward: (string) Defines how to compute the reward
        :param distance_type: (string) Way of calculating distances (euclidean, manhattan)
        :param active_cameras: (int) The number of camera used to render and record images
        :param dataset: (bool) Whether this envzironment serves for image dataset generation
        :param obs_space: (string) Type of observation data. Specify "dict" for usage of HER training algorithm.
        :param visualize: (bool) Whether to show helping visualization during training and testing of environment
        :param visgym: (bool) Whether to visualize whole gym building and other tasks (final visualization)
        :param logdir: (string) Directory for logging
        :param vae_path: (string) Path to a trained VAE in 2dvu reward type
        :param yolact_path: (string) Path to a trained Yolact in 3dvu reward type
        :param yolact_config: (string) Path to saved Yolact config obj or name of an existing one in the data/Config script or None for autodetection
        :param natural_language: (bool) Whether the natural language mode should be turned on
        :param training: (bool) Whether a training or a testing is taking place
    """
    @property
    def unwrapped(self):
        return self

    def __init__(self,
                 observation,
                 rddl={},
                 framework="ray",
                 workspace="table",
                 dimension_velocity=0.05,
                 action_repeat=1,
                 color_dict={},
                 robot='kuka',
                 robot_action="step",
                 max_velocity = 1,
                 max_force = 30,
                 robot_init_joint_poses=[],
                 task_type='reach',
                 num_networks=1,
                 network_switcher="gt",
                 distractors=None,
                 active_cameras=None,
                 dataset=False,
                 obs_space=None,
                 visualize=0,
                 visgym=1,
                 logdir=None,
                 vae_path=None,
                 yolact_path=None,
                 yolact_config=None,
                 natural_language=False,
                 training=True,
                 **kwargs
                 ):

        self.workspace              = workspace
        self.obs_type = observation
        self.robot_type             = robot
        self.num_networks           = num_networks
        self.network_switcher       = network_switcher
        self.robot_init_joint_poses = robot_init_joint_poses
        self.robot_action           = robot_action
        self.max_velocity           = max_velocity
        self.max_force              = max_force
        self.action_repeat          = action_repeat
        self.dimension_velocity     = dimension_velocity
        self.active_cameras         = active_cameras
        self.action_repeat          = action_repeat
        self.color_dict             = color_dict
        self.task_type              = task_type
        self.rddl_config = rddl
        self.framework = framework
        self.vision_source = get_module_type(self.obs_type)
        self.task_objects = []
        self.env_objects = []
        self.vae_path = vae_path
        self.yolact_path = yolact_path
        self.yolact_config = yolact_config
        self.has_distractor         = distractors["list"] != None
        self.distractors            = distractors
        self.objects_area_borders = None
        self.reachable_borders = None
        self.dist = d.DistractorModule(distractors["moveable"], distractors["movement_endpoints"],
                                       distractors["constant_speed"], distractors["movement_dims"], env=self)
        self.dataset   = dataset
        self.obs_space = obs_space
        self.visualize = visualize
        self.visgym    = visgym
        self.check_obs_template()
        self.logdir    = logdir
        self.workspace_dict = get_workspace_dict()
        self.robot = None
        self.task_checker = oraculum.Oraculum(self, kwargs["arg_dict"], max_steps=kwargs["max_steps"], robot_action=robot_action)
        kwargs.pop("arg_dict")
        if not hasattr(self, "task"):
          self.task = None

        self.reach_gesture = False
        if self.task_type == "reach_gesture":
            if self.workspace != "collabtable":
                exc = f"Expected collabtable workspace R the reach_gesture task is passed, got {self.workspace} instead"
                raise Exception(exc)

            self.reach_gesture = True
            self.task_type = "reach"

        self.nl_mode = natural_language
        if self.nl_mode:
            self.nl = NaturalLanguage(self)
        self.use_magnet = True # @TODO provisory solution, used when _gripper in self.robot_action. We should make adjustable in config whether to use asctual gripper control or magnet
        self.nl_text_id = None
        self.training = training
        if self.nl_mode:
            if not isinstance(self.task_objects_dict, dict):
                exc = f"Expected task_objects to be of type {dict} instead of {type(self.task_objects_dict)}"
                raise Exception(exc)
            # # just some dummy settings so that _set_observation_space() doesn't throw exceptions at the beginning

        self.rng = np.random.default_rng(seed=0)
        if self.reach_gesture and not self.nl_mode:
            raise Exception("Reach gesture task can't be started without natural language mode")

        super(GymEnv, self).__init__(active_cameras=active_cameras, **kwargs)


    def _init_task_and_reward(self):
        """Main communicator with rddl. Passes arguments from config to RDDLWorld, tells rddl to make a task sequence and to build
        a scene accordingly, including robot. Work in progress"""
        self.task = TaskModule(self, self.rddl_config["num_task_range"], self.rddl_config["protoactions"], self.rddl_config["allowed_objects"], self.rddl_config["allowed_predicates"], self.p)
        # generates task sequence and initializes scene with objects accordingly. The first action is set as self.task.current_task
        self.task.build_scene_for_task_sequence() # it also loads the robot. must be done his way so that rddl knows about the robot
        self.reward = self.task.current_action.reward # reward class

        print(f"Initial condition is: {self.task.rddl_task.current_action.initial.decide()}")
        print(f"Goal condition is: {self.task.rddl_task.current_action.goal.decide()}")

        self.compute_reward = self.task.current_action.reward # function that computes reward (no inputs needed, already bound to the objects)
        obs_entities = self.reward.get_relevant_entities() # does not work yet, must be done in rddl
        self.robot = self.task.rddl_robot # robot class as we know it



    def _setup_scene(self):
        """
        Set-up environment scene. Load static objects, apply textures.
        """
        self._add_scene_object_uid(self._load_static_scene_urdf(path="rooms/plane.urdf", name="floor"), "floor")
        if self.visgym:
            self._add_scene_object_uid(self._load_urdf(path="rooms/room.urdf"), "gym")
            [self._add_scene_object_uid(self._load_urdf(path="rooms/visual/" + self.workspace_dict[w]['urdf']), w)
             for w in self.workspace_dict if w != self.workspace]
        self._add_scene_object_uid(
            self._load_static_scene_urdf(path="rooms/collision/" + self.workspace_dict[self.workspace]['urdf'], name=self.workspace), self.workspace)
        ws_texture = self.workspace_dict[self.workspace]['texture'] if get_module_type(
            self.obs_type) != "vae" else "grey.png"
        if ws_texture: self._change_texture(self.workspace, self._load_texture(ws_texture))
        self._change_texture("floor", self._load_texture("parquet1.jpg"))
        self.objects_area_borders = self.workspace_dict[self.workspace]['borders']
        self.reachable_borders = self.workspace_dict[self.workspace]['reachable_borders']
        self.robot_kwargs = {"position": self.workspace_dict[self.workspace]['robot']['position'],
                  "orientation": self.workspace_dict[self.workspace]['robot']['orientation'],
                  "init_joint_poses": self.robot_init_joint_poses, "max_velocity": self.max_velocity,
                  "max_force": self.max_force, "dimension_velocity": self.dimension_velocity,
                  "pybullet_client": self.p, "env": self, "observation":self.vision_source, "vae_path":self.vae_path,
                  "yolact_path":self.yolact_path, "yolact_config": self.yolact_config}
        if self.workspace == 'collabtable': self.human = Human(model_name='human', pybullet_client=self.p)


    def _load_urdf(self, path, fixedbase=True, maxcoords=True):
        transform = self.workspace_dict[self.workspace]['transform']
        return self.p.loadURDF(pkg_resources.files("myGym").joinpath("envs",path),
                               transform['position'],  self.p.getQuaternionFromEuler(transform['orientation']),
                               useFixedBase=fixedbase,
                               useMaximalCoordinates=maxcoords)

    def _load_static_scene_urdf(self, path, name, fixedbase=True):
        transform = self.workspace_dict[self.workspace]['transform']
        object = env_object.EnvObject(pkg_resources.files("myGym").joinpath("envs",path), self, transform['position'], self.p.getQuaternionFromEuler(transform['orientation']), pybullet_client=self.p, fixed=fixedbase, observation=self.vision_source, vae_path=self.vae_path, yolact_path=self.yolact_path, yolact_config=self.yolact_config)
        self.static_scene_objects[name] = object
        return object.uid

    def _change_texture(self, name, texture_id):
        self.p.changeVisualShape(self.get_scene_object_uid_by_name(name), -1,
                                 rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id)


    def _load_texture(self, name):
        return self.p.loadTexture(str(pkg_resources.files("myGym").joinpath("envs/textures/",name)))


    def _set_observation_space(self):
        """
        Set observation space type, dimensions and range
        """
        if self.framework == "ray":
            from gymnasium import spaces
        else:
            from gymnasium import spaces
        if self.obs_space == "dict":
            goaldim = int(self.obsdim / 2) if self.obsdim % 2 == 0 else int(self.obsdim / 3)
            self.observation_space = spaces.Dict(
                {"observation": spaces.Box(low=-10, high=10, shape=(self.obsdim,)),
                 "achieved_goal": spaces.Box(low=-10, high=10, shape=(goaldim,)),
                 "desired_goal": spaces.Box(low=-10, high=10, shape=(goaldim,))})
        else:
            observationDim = self.obsdim
            observation_high = np.array([100] * observationDim)
            self.observation_space = spaces.Box(-observation_high,
                                                observation_high, dtype = np.float64)


    def _set_action_space(self):
        """
        Set action space dimensions and range
        """
        self._init_task_and_reward() # we first need rddl to make a robot
        if self.framework == "ray":
            from gymnasium import spaces
        else:
            from gymnasium import spaces
        action_dim = self.robot.get_action_dimension()
        if "step" in self.robot_action:
            self.action_low = np.array([-1] * action_dim)
            self.action_high = np.array([1] * action_dim)

        elif "absolute" in self.robot_action:
            if any(isinstance(i, list) for i in self.objects_area_borders):
                borders_max = np.max(self.objects_area_borders, 0)
                borders_min = np.min(self.objects_area_borders, 0)
                self.action_low = np.array(borders_min[0:7:2])
                self.action_high = np.array(borders_max[1:7:2])
            else:
                self.action_low = np.array(self.objects_area_borders[0:7:2],dtype = np.float64)
                self.action_high = np.array(self.objects_area_borders[1:7:2], dtype = np.float64)


        elif "joints" in self.robot_action:
            self.action_low = np.array(self.robot.joints_limits[0], dtype = np.float64)
            self.action_high = np.array(self.robot.joints_limits[1], dtype = np.float64)

        if "gripper" in self.robot_action:
            self.action_low = np.append(self.action_low, np.array(self.robot.gjoints_limits[0]))
            self.action_high = np.append(self.action_high, np.array(self.robot.gjoints_limits[1]))

        self.action_space = spaces.Box(self.action_low, self.action_high, dtype = np.float64)

    def _rescale_action(self, action):
        """
        Rescale action returned by trained model to fit environment action space

        Parameters:
            :param action: (list) Action data returned by trained model
        Returns:
            :return rescaled_action: (list) Rescaled action to environment action space
        """
        return [(sub + 1) * (h - l) / 2 + l for sub, l, h in zip(action, self.action_low, self.action_high)]

    def reset(self, random_pos=True, hard=False, random_robot=False, only_subtask=False, options=None, seed=None):
        """
        Environment reset called at the beginning of an episode. Reset state of objects, robot, task and reward.

        Parameters:
            :param random_pos: (bool) Whether to initiate objects to random locations in the scene
            :param hard: (bool) Whether to do hard reset (resets whole pybullet scene)
            :param random_robot: (bool) Whether to initiate robot in random pose
            :param only_subtask: (bool) if True, the robot's position is not reset and the next subtask is started
        Returns:
            :return self._observation: (list) Observation data of the environment
        """
        #super().reset(seed=seed)
        if not only_subtask:
            self.task.rddl_robot.reset(random_robot=random_robot)
            super().reset(hard=hard, seed=seed)
        # @TODO I removed support of nl_mode, which is dependent on the old structure. We need to add nl_support again in later phases
        self.task.reset_task()
        #self.reward.reset()
        self.p.stepSimulation()
        self._observation = self.get_observation()
        #TODO: after oraculum works successfully, implement saving of task descriptions which work
        env_copy = copy.deepcopy(self)
        info = {'d': 1, 'f': int(self.episode_failed),
                'o': self._observation}
        task_checker = oraculum.Oraculum(env_copy, info, self.max_episode_steps, self.robot_action)
        task_feasible = task_checker.check_task_feasibility()
        if task_feasible:
            print("Task is feasible, checked with IK control.")
        else:
            print("Task is NOT feasible, checked with IK control, generating next task.")
            #TODO: generate the next task
        if self.gui_on and self.nl_mode:
            if self.reach_gesture:
                self.nl.set_current_subtask_description("reach there")
            self.nl_text_id = self.p.addUserDebugText(self.nl.get_previously_generated_subtask_description(), [2, 0, 1], textSize=1)
            if only_subtask and self.nl_text_id is not None:
                self.p.removeUserDebugItem(self.nl_text_id)
        info = {'d': 0.9, 'f': 0, 'o': self._observation} 
        return (np.asarray(self.flatten_obs(self._observation.copy()), dtype="float32"), info)

    def flatten_obs(self, obs_dict):
        """ Returns the input obs dict as flattened list """
        # if len(obs["additional_obs"].keys()) != 0 and not self.dataset:
        #     obs["additional_obs"] = [p for sublist in list(obs["additional_obs"].values()) for p in sublist]
        # if not self.dataset:
        #     obs = np.asarray([p for sublist in list(obs.values()) for p in sublist])
        flattened_obs = np.zeros(22)
        for i, (key, value) in enumerate(obs_dict.items()):
            if key == "gripper_state":
                flattened_obs[i*7] = value
            else:
                flattened_obs[i*7:(i+1)*7] = value
        return flattened_obs

    def _set_cameras(self):
        """
        Add cameras to the environment
        """
        camera_args = self.workspace_dict[self.workspace]['camera']
        for cam_idx in range(len(camera_args['position'])):
            self.add_camera(position=camera_args['position'][cam_idx], target_position=camera_args['target'][cam_idx],
                            distance=0.001, is_absolute_position=True)

    def get_observation(self):
        """
        Get observation data from the environment. Probably should be moved to task.py

        Returns:
            :return observation: (array) Represented position of task relevant objects
        """
        ## !!!! This is a provisory solution until self.task.current_task.reward.get_relevant_entities() returns the relevant objects.
        ## Now we just take the position and orientation for every object in the scene including gripper, but that is legit in very few cases
        obs = dict()
        relevant_entities = self.reward.get_relevant_entities()
        if len(relevant_entities) == 0:
            warnings.warn(f"PRAG action {self.reward} is missing implementation of get_relevant_entities()! I am setting the gripper 6D and random object 6D as inputs to neural network.", category=UserWarning)
            relevant_entities = self.task.scene_objects[:2]
        #TODO: After retrieving relevant entities, decide which position is "actual_state" and which "goal_state"
        for name, obj in relevant_entities:
            obs[name] = list(obj.get_position()) + list(obj.get_orientation())
        if "gripper" in self.robot_action:
            obs["gripper_state"] = int(self.robot.gripper_active)
        self.observation = obs
        print("observation", self.observation)
        print("Observation dimension:", self.obsdim)
        if self.dataset:
            raise NotImplemented # @TODO one day
        return self.observation


    def step(self, action):
        """
        Environment step in simulation

        Parameters:
            :param action: (list) Action data to send to robot to perform movement in this step
        Returns:
            :return self._observation: (list) Observation data from the environment after this step
            :return reward: (float) Reward value assigned to this step
            :return done: (bool) Whether this stop is episode's final
            :return info: (dict) Additional information about step
        """
        self._apply_action_robot(action)
        if self.has_distractor: [self.dist.execute_distractor_step(d) for d in self.distractors["list"]]
        self._observation = self.get_observation()
        if self.dataset:
            reward, terminated, truncated, info = 0, False, False, {}
        else:
            reward = self.compute_reward()  # this uses rddl protoaction, no arguments needed
            self.episode_reward += reward
            done = self.episode_over #@TODO replace with actual is_done value from RDDL
            info = {'d': 0.9, 'f': int(self.episode_failed),
                    'o': self._observation} # @TODOreplace 'd' with actual distance values obtained from rddl or make own implementation
        if done is True: self.successful_finish(info)
        if self.task.subtask_over:
            self.reset(only_subtask=True)
        # return self._observation, reward, done, truncated, info
        truncated = False #not sure when to use this
        return np.asarray(self.flatten_obs(self._observation.copy()), dtype="float32"), reward, done, truncated,info


    def get_linkstates_unpacked(self):
        o = []
        [[o.append(x) for x in z] for z in self.robot.observe_all_links()]
        return o


    def check_obs_template(self):
        """
        @TODO Add smart and variable observation space constructor based on config? 

        Returns:
            :return obsdim: (int) Dimensionality of observation
        """
        obsdim = 21 # 7 values for each potential relevant entity - gripper, object, goal. Some tasks only use 2 of those
        if "gripper" in self.robot_action:
            obsdim += 1 # binary value for gripper close or open
        return obsdim

    def successful_finish(self, info):
        """
        End the episode and print summary
        Parameters:
            :param info: (dict) logged information about training
        """
        self.episode_final_reward.append(self.episode_reward)
        self.episode_number += 1
        self._print_episode_summary(info)

    def _apply_action_robot(self, action):
        """
        Apply desired action to robot in simulation

        Parameters:
            :param action: (list) Action data returned by trained model
        """
        for i in range(self.action_repeat):
            objects = self.env_objects
            self.robot.apply_action(action, env_objects=objects)
            if hasattr(self, "human"):
                self.human.point_finger_at(position=self.task_objects["goal_state"].get_position())
            self.p.stepSimulation()
        self.episode_steps += 1
        
    def choose_goal_object_by_human_with_keys(self, objects: List[EnvObject]) -> EnvObject:
        self.text_id = self.p.addUserDebugText("Point the human's finger via arrow keys at the goal object and press enter", [1, 0, 0.5], textSize=1)
        move_factor = 10  # times 1 cm

        while True:
            key_press = self.p.getKeyboardEvents()
            key_pressed = False

            if self.p.B3G_LEFT_ARROW in key_press.keys() and key_press[self.p.B3G_LEFT_ARROW] == 3:
                self.human.point_finger_at(move_factor * np.array([0.01, 0, 0]), relative=True)
                key_pressed = True
            if self.p.B3G_RIGHT_ARROW in key_press.keys() and key_press[self.p.B3G_RIGHT_ARROW] == 3:
                self.human.point_finger_at(move_factor * np.array([-0.01, 0, 0]), relative=True)
                key_pressed = True
            if self.p.B3G_DOWN_ARROW in key_press.keys() and key_press[self.p.B3G_DOWN_ARROW] == 3:
                self.human.point_finger_at(move_factor * np.array([0, 0, -0.01]), relative=True)
                key_pressed = True
            if self.p.B3G_UP_ARROW in key_press.keys() and key_press[self.p.B3G_UP_ARROW] == 3:
                self.human.point_finger_at(move_factor * np.array([0, 0, 0.01]), relative=True)
                key_pressed = True
            if self.p.B3G_RETURN in key_press.keys() and key_press[self.p.B3G_RETURN] == 3:
                self.p.removeUserDebugItem(self.text_id)
                return self.human.find_object_human_is_pointing_at(objects=objects)

            if key_pressed:
                self.p.stepSimulation()

    def draw_bounding_boxes(self):
        """
        Show bounding box of object in the scene in GUI
        """
        for object in self.env_objects:
            object.draw_bounding_box()

    def highlight_active_object(self, env_o, obj_role):
        if obj_role == "goal":
            env_o.set_color(cs.name_to_rgba("red"))
        elif obj_role == "init":
            env_o.set_color(cs.name_to_rgba("green"))
        elif obj_role == "done":
            env_o.set_color(cs.name_to_rgba("blue"))
        else:
            env_o.set_color(cs.name_to_rgba("gray"))


    def get_random_color(self):
        return cs.draw_random_rgba()

    def get_task_objects(self, with_none=False) -> List[EnvObject]:
        objects = [self.task_objects["actual_state"], self.task_objects["goal_state"]]
        if "distractor" in self.task_objects:
            objects += self.task_objects["distractor"]
        return [o for o in objects if isinstance(o, EnvObject)] if not with_none else [o if isinstance(o, EnvObject) else None for o in objects]

    def set_current_subtask_goal(self, goal) -> None:
        self.task_objects["actual_state"] = goal

    def network_control(self):
        return self.unwrapped.reward.network_switch_control(self.observation["task_objects"])

    def get_actions(self, owner, observation):
        model = self.models_link[owner]
        with th.no_grad():
            obs = obs_as_tensor(observation, "cpu")
            obs = th.unsqueeze(obs, 0)
            action, value, log_prob = model.policy(obs)
        return action, value, log_prob

    def set_models(self, models):
        self.models_link = models



