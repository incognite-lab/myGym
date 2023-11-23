import copy
from typing import List

from myGym.envs import robot, env_object
from myGym.envs import task as t
from myGym.envs import distractor as d
from myGym.envs.base_env import CameraEnv
from collections import ChainMap

from myGym.envs.env_object import EnvObject
from myGym.envs.rewards import *
import numpy as np
from itertools import chain
from gym import spaces
import random

from myGym.utils.helpers import get_workspace_dict
import pkg_resources
from myGym.envs.human import Human
import myGym.utils.colors as cs
from myGym.envs.vision_module import get_module_type
from myGym.envs.natural_language import NaturalLanguage

currentdir = pkg_resources.resource_filename("myGym", "envs")

# used to exclude these colors for other objects, in the order of goal, init, done, else
COLORS_RESERVED_FOR_HIGHLIGHTING = ["dark green", "green", "blue", "gray"]


class GymEnv(CameraEnv):
    """
    Environment class for particular environment based on myGym basic environment classes

    Parameters:
        :param task_objects: (list of strings) Objects that are relevant for performing the task
        :param observation: (dict) Template dictionary of what should be part of the network observation
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

    def __init__(self,
                 task_objects,
                 observation,
                 workspace="table",
                 dimension_velocity=0.05,
                 used_objects=None,
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
                 reward='distance',
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
        self.used_objects           = used_objects
        self.action_repeat          = action_repeat
        self.color_dict             = color_dict
        self.task_type              = task_type
        self.task_objects_dict     = task_objects
        self.task_objects = []
        self.env_objects = []
        self.vae_path = vae_path
        self.yolact_path = yolact_path
        self.yolact_config = yolact_config
        self.has_distractor         = distractors["list"] != None
        self.distractors            = distractors
        self.distance_type          = distance_type
        self.objects_area_borders = None
        self.reward = reward
        self.dist = d.DistractorModule(distractors["moveable"], distractors["movement_endpoints"],
                                       distractors["constant_speed"], distractors["movement_dims"], env=self)
        self.dataset   = dataset
        self.obs_space = obs_space
        self.visualize = visualize
        self.visgym    = visgym
        self.logdir    = logdir
        self.workspace_dict = get_workspace_dict()
        if not hasattr(self, "task"):
          self.task = None

        self.reach_gesture = False
        if self.task_type == "reach_gesture":
            if self.workspace != "collabtable":
                exc = f"Expected collabtable workspace when the reach_gesture task is passed, got {self.workspace} instead"
                raise Exception(exc)

            self.reach_gesture = True
            self.task_type = "reach"

        self.nl = NaturalLanguage(self)
        self.nl_mode = natural_language
        self.nl_text_id = None
        self.training = training
        if self.nl_mode:
            if not isinstance(self.task_objects_dict, dict):
                exc = f"Expected task_objects to be of type {dict} instead of {type(self.task_objects_dict)}"
                raise Exception(exc)
            # # just some dummy settings so that _set_observation_space() doesn't throw exceptions at the beginning
            # self.num_networks = 3
        self.rng = np.random.default_rng(seed=0)
        self.task_objects_were_given_as_list = isinstance(self.task_objects_dict, list)
        self.n_subtasks = len(self.task_objects_dict) if self.task_objects_were_given_as_list else 1
        if self.reach_gesture and not self.nl_mode:
            raise Exception("Reach gesture task can't be started without natural language mode")

        super(GymEnv, self).__init__(active_cameras=active_cameras, **kwargs)

    def _init_task_and_reward(self):
        if self.reward == 'distractor':
            self.has_distractor = True
            self.distractor = ['bus'] if not self.distractors["list"] else self.distractors["list"]
        reward_classes = {
            "1-network": {"distance": DistanceReward, "complex_distance": ComplexDistanceReward, "sparse": SparseReward,
                          "distractor": VectorReward, "poke": PokeReachReward, "push": PushReward, "switch": SwitchReward,
                          "btn": ButtonReward, "turn": TurnReward, "pnp": SingleStagePnP},
            "2-network": {"poke": DualPoke, "pnp": TwoStagePnP, "pnpbgrip": TwoStagePnPBgrip, "push": TwoStagePushReward},
            "3-network": {"pnp": ThreeStagePnP, "pnprot": ThreeStagePnPRot, "pnpswipe": ThreeStageSwipe, "FMR": FaMaR,"FROM": FaROaM,  "FMOR": FaMOaR, "FMOT": FaMOaT, "FROT": FaROaT,
                          "pnpswiperot": ThreeStageSwipeRot},
            "4-network": {"pnp": FourStagePnP, "pnprot": FourStagePnPRot, "FMLFR": FaMaLaFaR}}
    
        scheme = "{}-network".format(str(self.num_networks))
        assert self.reward in reward_classes[scheme].keys(), "Failed to find the right reward class. Check reward_classes in gym_env.py"
        self.task = t.TaskModule(task_type=self.task_type,
                                 observation=self.obs_type,
                                 vae_path=self.vae_path,
                                 yolact_path=self.yolact_path,
                                 yolact_config=self.yolact_config,
                                 distance_type=self.distance_type,
                                 number_tasks=len(self.task_objects_dict),
                                 env=self)
        self.reward = reward_classes[scheme][self.reward](env=self, task=self.task)

    def _setup_scene(self):
        """
        Set-up environment scene. Load static objects, apply textures. Load robot.
        """
        self._add_scene_object_uid(self._load_urdf(path="rooms/plane.urdf"), "floor")
        if self.visgym:
            self._add_scene_object_uid(self._load_urdf(path="rooms/room.urdf"), "gym")
            #self._change_texture("gym", self._load_texture("verticalmaze.jpg"))
            [self._add_scene_object_uid(self._load_urdf(path="rooms/visual/" + self.workspace_dict[w]['urdf']), w)
             for w in self.workspace_dict if w != self.workspace]
        self._add_scene_object_uid(
            self._load_urdf(path="rooms/collision/" + self.workspace_dict[self.workspace]['urdf']), self.workspace)
        ws_texture = self.workspace_dict[self.workspace]['texture'] if get_module_type(
            self.obs_type) != "vae" else "grey.png"
        if ws_texture: self._change_texture(self.workspace, self._load_texture(ws_texture))
        self._change_texture("floor", self._load_texture("parquet1.jpg"))
        self.objects_area_borders = self.workspace_dict[self.workspace]['borders']
        kwargs = {"position": self.workspace_dict[self.workspace]['robot']['position'],
                  "orientation": self.workspace_dict[self.workspace]['robot']['orientation'],
                  "init_joint_poses": self.robot_init_joint_poses, "max_velocity": self.max_velocity,
                  "max_force": self.max_force, "dimension_velocity": self.dimension_velocity,
                  "pybullet_client": self.p}
        self.robot = robot.Robot(self.robot_type, robot_action=self.robot_action, task_type=self.task_type, **kwargs)
        if self.workspace == 'collabtable': self.human = Human(model_name='human', pybullet_client=self.p)

    def _load_urdf(self, path, fixedbase=True, maxcoords=True):
        transform = self.workspace_dict[self.workspace]['transform']
        return self.p.loadURDF(pkg_resources.resource_filename("myGym", os.path.join("envs", path)),
                               transform['position'], self.p.getQuaternionFromEuler(transform['orientation']),
                               useFixedBase=fixedbase,
                               useMaximalCoordinates=maxcoords)

    def _change_texture(self, name, texture_id):
        self.p.changeVisualShape(self.get_scene_object_uid_by_name(name), -1,
                                 rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id)

    def _load_texture(self, name):
        return self.p.loadTexture(pkg_resources.resource_filename("myGym", "/envs/textures/{}".format(name)))

    def _set_observation_space(self):
        """
        Set observation space type, dimensions and range
        """
        self._init_task_and_reward()
        if self.obs_space == "dict":
            goaldim = int(self.task.obsdim / 2) if self.task.obsdim % 2 == 0 else int(self.task.obsdim / 3)
            self.observation_space = spaces.Dict(
                {"observation": spaces.Box(low=-10, high=10, shape=(self.task.obsdim,)),
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
        if "step" in self.robot_action:
            self.action_low = np.array([-1] * action_dim)
            self.action_high = np.array([1] * action_dim)
            # if "gripper" in self.robot_action:
            #    self.action_low = np.insert(self.action_low, action_dim, self.robot.gjoints_limits[0][1])
            #    self.action_high = np.insert(self.action_high, action_dim,self.robot.gjoints_limits[1][1])


        elif "absolute" in self.robot_action:
            if any(isinstance(i, list) for i in self.objects_area_borders):
                borders_max = np.max(self.objects_area_borders, 0)
                borders_min = np.min(self.objects_area_borders, 0)
                self.action_low = np.array(borders_min[0:7:2])
                self.action_high = np.array(borders_max[1:7:2])
            else:
                self.action_low = np.array(self.objects_area_borders[0:7:2])
                self.action_high = np.array(self.objects_area_borders[1:7:2])


        elif "joints" in self.robot_action:
            self.action_low = np.array(self.robot.joints_limits[0])
            self.action_high = np.array(self.robot.joints_limits[1])

        if "gripper" in self.robot_action:
            self.action_low = np.append(self.action_low, np.array(self.robot.gjoints_limits[0]))
            self.action_high = np.append(self.action_high, np.array(self.robot.gjoints_limits[1]))

        self.action_space = spaces.Box(self.action_low, self.action_high)

    def _rescale_action(self, action):
        """
        Rescale action returned by trained model to fit environment action space

        Parameters:
            :param action: (list) Action data returned by trained model
        Returns:
            :return rescaled_action: (list) Rescaled action to environment action space
        """
        return [(sub + 1) * (h - l) / 2 + l for sub, l, h in zip(action, self.action_low, self.action_high)]

    def reset(self, random_pos=True, hard=False, random_robot=False, only_subtask=False):
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
        if not only_subtask:
            self.robot.reset(random_robot=random_robot)
            super().reset(hard=hard)

            if not self.nl_mode:
                other_objects = []
                if self.task_objects_were_given_as_list:
                    task_objects_dict = copy.deepcopy(self.task_objects_dict)
                else:
                    if not self.reach_gesture:
                        init = self.rng.choice(self.task_objects_dict["init"])
                        goal = self.rng.choice(self.task_objects_dict["goal"])
                        objects = self.task_objects_dict["init"] + self.task_objects_dict["goal"]
                        task_objects_dict = [{"init": init, "goal": goal}]
                        other_objects = self._randomly_place_objects({"obj_list": [o for o in objects if o != init and o != goal]})
                    else:
                        goal = self.rng.choice(self.task_objects_dict["goal"])
                        task_objects_dict = [{"init": {"obj_name":"null"}, "goal": goal}]
                        other_objects = self._randomly_place_objects({"obj_list": [o for o in self.task_objects_dict["goal"] if o != goal]})

                all_subtask_objects = [x for i, x in enumerate(task_objects_dict) if i != self.task.current_task]
                subtasks_processed = [list(x.values()) for x in all_subtask_objects]
                subtask_objects = self._randomly_place_objects({"obj_list": list(chain.from_iterable(subtasks_processed))})
                self.env_objects = {"env_objects": self._randomly_place_objects(self.used_objects)}
                if self.task_objects_were_given_as_list:
                    self.env_objects["env_objects"] += other_objects
                self.task_objects = self._randomly_place_objects(task_objects_dict[self.task.current_task])
                self.task_objects = dict(ChainMap(*self.task_objects))
                if subtask_objects:
                    self.task_objects["distractor"] = subtask_objects

                self.nl.get_venv().set_objects(task_objects=self.task_objects)
            else:
                init_objects = []
                if not self.reach_gesture:
                    init_objects = self._randomly_place_objects({"obj_list": self.task_objects_dict["init"]})
                    for i, c in enumerate(cs.draw_random_rgba(size=len(init_objects), excluding=COLORS_RESERVED_FOR_HIGHLIGHTING)):
                        init_objects[i].set_color(c)
                goal_objects = self._randomly_place_objects({"obj_list": self.task_objects_dict["goal"]})
                for i, c in enumerate(cs.draw_random_rgba(size=len(goal_objects), transparent=self.task_type != "reach", excluding=COLORS_RESERVED_FOR_HIGHLIGHTING)):
                    goal_objects[i].set_color(c)

                if self.training or (not self.training and self.reach_gesture):
                    # setting the objects and generating a description based on them
                    self.nl.get_venv().set_objects(init_goal_objects=(init_objects, goal_objects))
                    self.nl.generate_subtask_with_random_description()

                    # resetting the objects to remove the knowledge about whether an object is an init or a goal
                    self.nl.get_venv().set_objects(all_objects=init_objects + goal_objects)
                    self.task_type, self.reward, self.num_networks, init, goal = self.nl.extract_subtask_info_from_description(self.nl.get_previously_generated_subtask_description())
                else:
                    success = False
                    i = 0

                    while (not success):
                        try:
                            if i > 0:
                                print("Unknown task description format. Actual format is very strict. "
                                      "All articles must be included. Examples of valid subtask descriptions in general:")
                                print("\"reach the cyan cube\"")
                                print("\"reach the transparent pink cube left to the gray cube\"")
                                print("\"pick the orange cube and place it to the same position as the pink cube\"")
                                print("Pay attention to the fact that colors, task and objects in your case can be different!")
                                print("To leave the program use Ctrl + Z!")
                            self.nl.set_current_subtask_description(input("Enter a subtask description in the natural language based on what you see:"))
                            # resetting the objects to remove the knowledge about whether an object is an init or a goal
                            self.nl.get_venv().set_objects(all_objects=init_objects + goal_objects)
                            self.task_type, self.reward, self.num_networks, init, goal = self.nl.extract_subtask_info_from_description(self.nl.get_previously_generated_subtask_description())
                            success = True
                            break
                        except:
                            pass
                        i += 1

                self.task_objects = {"actual_state": init if init is not None else self.robot, "goal_state": goal}
                other_objects = [o for o in init_objects + goal_objects if o != init and o != goal]
                self.env_objects = {"env_objects": other_objects + self._randomly_place_objects(self.used_objects)}

                # will set the task and the reward
                self._set_observation_space()
        if only_subtask:
            if self.task.current_task < (len(self.task_objects_dict)) and not self.nl_mode:
                self.shift_next_subtask()
        if self.has_distractor:
            distrs = []
            if self.distractors["list"]:
                for distractor in self.distractors["list"]:
                    distrs.append(
                        self.dist.place_distractor(distractor, self.p, self.task_objects["goal_state"].get_position()))
            if self.task_objects["distractor"]:
                self.task_objects["distractor"].extend(distrs)
            else:
                self.task_objects["distractor"] = distrs
        self.env_objects = {**self.task_objects, **self.env_objects}
        self.task.reset_task()
        self.reward.reset()
        self.p.stepSimulation()
        self._observation = self.get_observation()

        if self.gui_on and self.nl_mode:
            if self.reach_gesture:
                self.nl.set_current_subtask_description("reach there")
            self.nl_text_id = self.p.addUserDebugText(self.nl.get_previously_generated_subtask_description(), [2, 0, 1], textSize=1)
            if only_subtask and self.nl_text_id is not None:
                self.p.removeUserDebugItem(self.nl_text_id)

        return self.flatten_obs(self._observation.copy())

    def shift_next_subtask(self):
        # put current init and goal back in env_objects
        self.env_objects["distractor"].extend([self.env_objects["actual_state"], self.env_objects["goal_state"]])
        # set the next subtask objects as the actual and goal state and remove them from env_objects
        self.env_objects["actual_state"] = self.env_objects["distractor"][0]
        self.env_objects["goal_state"] = self.env_objects["distractor"][1]
        del self.env_objects["distractor"][:2]
        self.task_objects["distractor"] = self.env_objects["distractor"].copy()
        # copy the state to task_objects and change colors
        self.task_objects["actual_state"] = self.env_objects["actual_state"]
        self.task_objects["goal_state"] = self.env_objects["goal_state"]
        self.highlight_active_object(self.env_objects["actual_state"], "init")
        self.highlight_active_object(self.env_objects["goal_state"], "goal")
        for o in self.env_objects["distractor"][-2:]:
            self.highlight_active_object(o, "done")

    def flatten_obs(self, obs):
        """ Returns the input obs dict as flattened list 
        if len(obs["additional_obs"].keys()) != 0 and not self.dataset:
            obs["additional_obs"] = [p for sublist in list(obs["additional_obs"].values()) for p in sublist]
        if not self.dataset:
            obs = np.asarray([p for sublist in list(obs.values()) for p in sublist])"""
        return obs

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
        Get observation data from the environment

        Returns:
            :return observation: (array) Represented position of task relevant objects
        """
        self.observation["task_objects"] = self.task.get_observation()
        if self.dataset:
            self.observation["camera_data"] = self.render(mode="rgb_array")
            self.observation["objects"] = self.env_objects
            self.observation["additional_obs"] = {}
            return self.observation
        return self.observation["task_objects"]

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
            reward, done, info = 0, False, {}
        else:
            reward = self.reward.compute(observation=self._observation)
            self.episode_reward += reward
            # self.task.check_goal()
            done = self.episode_over
            info = {'d': self.task.last_distance / self.task.init_distance, 'f': int(self.episode_failed),
                    'o': self._observation}
        if done: self.successful_finish(info)
        if self.task.subtask_over:
            self.reset(only_subtask=True)
        # return self._observation, reward, done, info
        return self.flatten_obs(self._observation.copy()), reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # @TODO: Reward computation for HER, argument for .compute()
        reward = self.reward.compute(np.append(achieved_goal, desired_goal))
        return reward

    def successful_finish(self, info):
        """
        End the episode and print summary
        Parameters:
            :param info: (dict) logged information about training
        """
        self.episode_final_reward.append(self.episode_reward)
        self.episode_final_distance.append(self.task.last_distance / self.task.init_distance)
        self.episode_number += 1
        self._print_episode_summary(info)

    def _apply_action_robot(self, action):
        """
        Apply desired action to robot in simulation

        Parameters:
            :param action: (list) Action data returned by trained model
        """
        use_magnet = self.reward.get_magnetization_status()
        for i in range(self.action_repeat):
            objects = self.env_objects
            self.robot.apply_action(action, env_objects=objects)
            if hasattr(self, "human"):
                self.human.point_finger_at(position=self.task_objects["goal_state"].get_position())
            self.p.stepSimulation()
        # print(f"Substeps:{i}")
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

    def _place_object(self, obj_info):
        fixed = True if obj_info["fixed"] == 1 else False
        pos = env_object.EnvObject.get_random_object_position(obj_info["sampling_area"])
        orn = env_object.EnvObject.get_random_z_rotation() if obj_info["rand_rot"] == 1 else [0, 0, 0, 1]
        object = env_object.EnvObject(obj_info["urdf"], pos, orn, pybullet_client=self.p, fixed=fixed)
        if self.color_dict: object.set_color(self.color_of_object(object))
        return object

    def _randomly_place_objects(self, object_dict):
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
        if not "init" in object_dict.keys():  # solves used_objects
            for idx, o in enumerate(object_dict["obj_list"]):
                  if o["obj_name"] != "null":
                      urdf = self._get_urdf_filename(o["obj_name"])
                      if urdf:
                        object_dict["obj_list"][idx]["urdf"] = urdf
                      else:
                        del object_dict["obj_list"][idx]
            if "num_range" in object_dict.keys():
                for x in range(random.randint(object_dict["num_range"][0], object_dict["num_range"][1])):
                    env_o = self._place_object(random.choice(object_dict["obj_list"]))
                    self.highlight_active_object(env_o, "other")
                    env_objects.append(env_o)
            else:
                for o in object_dict["obj_list"]:
                    if o["obj_name"] != "null":
                        env_o = self._place_object(o)
                        self.highlight_active_object(env_o, "other")
                        env_objects.append(env_o)
        else:  # solves task_objects
            for o in ['init', 'goal']:
                d = object_dict[o]
                if d["obj_name"] != "null":
                    d["urdf"] = self._get_urdf_filename(d["obj_name"])
                    n = "actual_state" if o == "init" else "goal_state"
                    env_o = self._place_object(d)
                    self.highlight_active_object(env_o, o)
                    env_objects.append({n: env_o})
                elif d["obj_name"] == "null" and o == "init":
                    env_objects.append({"actual_state": self.robot})
        return env_objects

    def highlight_active_object(self, env_o, obj_role):
        if obj_role == "goal":
            env_o.set_color(cs.name_to_rgba("transparent green"))
        elif obj_role == "init":
            env_o.set_color(cs.name_to_rgba("green"))
        elif obj_role == "done":
            env_o.set_color(cs.name_to_rgba("blue"))
        else:
            env_o.set_color(cs.name_to_rgba("gray"))

    def color_of_object(self, object):
        """
        Set object's color

        Parameters:
            :param object: (object) Object
        Returns:
            :return color: (list) RGB color
        """
        if object.name not in self.color_dict:
            return cs.draw_random_rgba()
        else:
            color_name = random.sample(self.color_dict[object.name], 1)[0]
            color = cs.name_to_rgba(color_name)
        return color

    def get_task_objects(self, with_none=False) -> List[EnvObject]:
        objects = [self.task_objects["actual_state"], self.task_objects["goal_state"]]
        if "distractor" in self.task_objects:
            objects += self.task_objects["distractor"]
        return [o for o in objects if isinstance(o, EnvObject)] if not with_none else [o if isinstance(o, EnvObject) else None for o in objects]

    def set_current_subtask_goal(self, goal) -> None:
        self.task_objects["actual_state"] = goal
