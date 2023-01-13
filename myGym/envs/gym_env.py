from myGym.envs import robot, env_object
from myGym.envs import task as t
from myGym.envs import distractor as d
from myGym.envs.base_env import CameraEnv
from collections import ChainMap
from myGym.envs.rewards import *
import numpy as np
from itertools import chain
from gym import spaces
import random
from myGym.utils.helpers import get_workspace_dict
import pkg_resources
currentdir = pkg_resources.resource_filename("myGym", "envs")


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
                 **kwargs
                 ):

        self.workspace              = workspace
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
        self.has_distractor         = distractors["list"] != None
        self.distractors            = distractors
        self.distance_type          = distance_type
        self.objects_area_borders = None

        self.task = t.TaskModule(task_type=self.task_type,
                                 observation=observation,
                                 task_objects=self.task_objects,
                                 vae_path=vae_path,
                                 yolact_path=yolact_path,
                                 yolact_config=yolact_config,
                                 distance_type=self.distance_type,
                                 number_tasks=len(task_objects),
                                 env=self)

        self.dist = d.DistractorModule(distractors["moveable"], distractors["movement_endpoints"],
                                       distractors["constant_speed"], distractors["movement_dims"], env=self)

        if reward == 'distractor':
            self.has_distractor = True
            self.distractor = ['bus'] if not self.distractors["list"] else self.distractors["list"]

        reward_classes = {"1-network":   {"distance": DistanceReward, "complex_distance": ComplexDistanceReward, "sparse": SparseReward,
                                              "distractor": VectorReward, "poke": PokeReachReward, "switch": SwitchReward,
                                              "btn": ButtonReward, "turn": TurnReward, "pnp":SingleStagePnP},
                          "2-network":     {"poke": DualPoke, "pnp":TwoStagePnP,"pnpbgrip":TwoStagePnPBgrip},
                          "3-network":     {"pnp":ThreeStagePnP, "pnprot":ThreeStagePnPRot, "pnpswipe":ThreeStageSwipe, "pnpswiperot":ThreeStageSwipeRot},
                          "4-network":     {"pnp":FourStagePnP}}
        scheme = "{}-network".format(str(self.num_networks))
        assert reward in reward_classes[scheme].keys(), "Failed to find the right reward class. Check reward_classes in gym_env.py"
        self.reward = reward_classes[scheme][reward](env=self, task=self.task)
        self.dataset   = dataset
        self.obs_space = obs_space
        self.visualize = visualize
        self.visgym    = visgym
        self.logdir    = logdir
        self.workspace_dict = get_workspace_dict()
        super(GymEnv, self).__init__(active_cameras=active_cameras, **kwargs)

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
        self._add_scene_object_uid(self._load_urdf(path="rooms/collision/"+self.workspace_dict[self.workspace]['urdf']), self.workspace)
        ws_texture = self.workspace_dict[self.workspace]['texture'] if self.task.vision_src != "vae" else "grey.png"
        if ws_texture: self._change_texture(self.workspace, self._load_texture(ws_texture))
        self._change_texture("floor", self._load_texture("parquet1.jpg"))
        self.objects_area_borders = self.workspace_dict[self.workspace]['borders']
        kwargs = {"position": self.workspace_dict[self.workspace]['robot']['position'],
                  "orientation": self.workspace_dict[self.workspace]['robot']['orientation'],
                  "init_joint_poses":self.robot_init_joint_poses, "max_velocity":self.max_velocity,
                    "max_force":self.max_force,"dimension_velocity":self.dimension_velocity,
                  "pybullet_client":self.p}
        self.robot = robot.Robot(self.robot_type, robot_action=self.robot_action,task_type=self.task_type, **kwargs)
        if self.workspace == 'collabtable':  self.human = robot.Robot('human', robot_action='joints', **kwargs)

    def _load_urdf(self, path, fixedbase=True, maxcoords=True):
        transform = self.workspace_dict[self.workspace]['transform']
        return self.p.loadURDF(pkg_resources.resource_filename("myGym", os.path.join("envs", path)),
            transform['position'], self.p.getQuaternionFromEuler(transform['orientation']), useFixedBase=fixedbase,
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
        if "step" in self.robot_action:
            self.action_low = np.array([-1] * action_dim)
            self.action_high = np.array([1] * action_dim)
            #if "gripper" in self.robot_action:
            #    self.action_low = np.insert(self.action_low, action_dim, self.robot.gjoints_limits[0][1])
            #    self.action_high = np.insert(self.action_high, action_dim,self.robot.gjoints_limits[1][1])
                

        elif "absolute" in self.robot_action:
            if any(isinstance(i, list) for i in self.objects_area_borders):
                borders_max = np.max(self.objects_area_borders,0)
                borders_min = np.min(self.objects_area_borders,0)
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
            all_subtask_objects = [x for i,x in enumerate(self.task_objects_dict) if i!=self.task.current_task]
            subtasks_processed = [list(x.values()) for x in all_subtask_objects]
            subtask_objects = self._randomly_place_objects({"obj_list": list(chain.from_iterable(subtasks_processed))})
            self.env_objects = {"env_objects": self._randomly_place_objects(self.used_objects)}
            self.task_objects = self._randomly_place_objects(self.task_objects_dict[self.task.current_task])
            self.task_objects = dict(ChainMap(*self.task_objects))
            if subtask_objects:
                self.task_objects["distractor"] = subtask_objects
        if only_subtask:
            if self.task.current_task < (len(self.task_objects_dict)):
                self.shift_next_subtask()
        if self.has_distractor:
            distrs = []
            if self.distractors["list"]:
                for distractor in self.distractors["list"]:
                    distrs.append(self.dist.place_distractor(distractor, self.p, self.task_objects["goal_state"].get_position()))
            if self.task_objects["distractor"]:
                self.task_objects["distractor"].extend(distrs)
            else:
                self.task_objects["distractor"] = distrs
        self.env_objects = {**self.task_objects, **self.env_objects}
        self.task.reset_task()
        self.reward.reset()
        self.p.stepSimulation()
        self._observation = self.get_observation()
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
        """ Returns the input obs dict as flattened list """
        if len(obs["additional_obs"].keys()) != 0 and not self.dataset:
           obs["additional_obs"] = [p for sublist in list(obs["additional_obs"].values()) for p in sublist]
        if not self.dataset:
            obs = np.asarray([p for sublist in list(obs.values()) for p in sublist])
        return obs

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
        if self.dataset: reward, done, info = 0, False, {}
        else:
            reward = self.reward.compute(observation=self._observation)
            self.episode_reward += reward
            #self.task.check_goal()
            done = self.episode_over
            info = {'d': self.task.last_distance / self.task.init_distance, 'f': int(self.episode_failed), 'o': self._observation}
        if done: self.successful_finish(info)
        if self.task.subtask_over: 
            self.reset(only_subtask=True)
        #return self._observation, reward, done, info
        return self.flatten_obs(self._observation.copy()), reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        #@TODO: Reward computation for HER, argument for .compute()
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
        for i in range(self.action_repeat):
            objects = self.env_objects
            self.robot.apply_action(action, env_objects=objects)
            if hasattr(self, 'human'):
                self.human.apply_action(np.random.uniform(self.human.joints_limits[0], self.human.joints_limits[1]))
            self.p.stepSimulation()
        #print(f"Substeps:{i}")
        self.episode_steps += 1

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
                    env_o = self._place_object(o)
                    self.highlight_active_object(env_o, "other")
                    env_objects.append(env_o)
        else:  # solves task_objects
            for o in ['init','goal']:
                d = object_dict[o]
                if d["obj_name"] != "null":
                    d["urdf"] = self._get_urdf_filename(d["obj_name"])
                    n = "actual_state" if o == "init" else "goal_state"
                    env_o = self._place_object(d)
                    self.highlight_active_object(env_o, o)
                    env_objects.append({n:env_o})
                elif d["obj_name"] == "null" and o == "init":
                    env_objects.append({"actual_state":self.robot})
        return env_objects

    def highlight_active_object(self, env_o, obj_role):
        if obj_role == "goal":
            env_o.set_color([0, 0.4, 0, 0.5])
        elif obj_role == "init":
            env_o.set_color([0, 0.8, 0, 1])
        elif obj_role == "done":
            env_o.set_color([0.5, 0.8, 1, 1])
        else:
            env_o.set_color([0.2, 0.2, 0.2, 1])

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
            color = random.sample(self.color_dict[object.name], 1)[0]
            color = [x / 255 for x in color] if any([x>1 for x in color]) else color
            if len(color) < 4: color.append(1)
        return color
