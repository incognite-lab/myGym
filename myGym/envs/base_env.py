import pybullet_data
import os
import pybullet
import pybullet_utils.bullet_client as bc
import time
import numpy as np
from gym.utils import seeding
import gym
import inspect
from myGym.envs.camera import Camera
import pkg_resources
currentdir = pkg_resources.resource_filename("myGym", "envs")
repodir = pkg_resources.resource_filename("myGym", "./")


class BaseEnv(gym.Env):
    """
    The base class for environments without rendering

    Parameters:
        :param gui_on: (bool) Whether or not to use PyBullet built-in GUI
        :param objects_dir_path: (str) Path to directory with URDF files for objects
        :param max_steps: (int) The maximum number of actions per episode
        :param show_bounding_boxes_gui: (bool) Whether or not to show bounding boxes in GUI
        :param changing_light_gui: (bool) Whether or not to change light in GUI
        :param shadows_on_gui: (bool) Whether or not to show shadows in GUI
    """
    metadata = {'render.modes': [
        'human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 gui_on=True,
                 objects_dir_path=pkg_resources.resource_filename("myGym", "envs/"),
                 max_steps=1024,
                 show_bounding_boxes_gui=False,
                 changing_light_gui=False,
                 shadows_on_gui=True
                 ):
        self.gui_on = gui_on
        self.max_steps = max_steps
        self.show_bounding_boxes_gui = show_bounding_boxes_gui
        self.changing_light_gui = changing_light_gui
        self.shadows_on_gui = shadows_on_gui

        # Set episode information
        self.episode_start_time = None
        self.episode_over = False
        self.episode_failed = False
        self.episode_reward = 0.0
        self.episode_final_reward = []
        self.episode_final_distance = []
        self.episode_number = 0
        self.episode_steps = 0
        self.episode_max_time = 300
        self.episode_info = ""

        # Set general params
        self.time_step = 1. / 240.
        self.urdf_root = pybullet_data.getDataPath()
        self.observation = {}

        # Set objects information
        self.objects_dir_path = objects_dir_path
        self.env_objects = {}
        self.scene_objects_uids = {}
        self.all_objects_filenames = self._get_all_urdf_filenames(self.objects_dir_path)

        # Set GUI
        self._connect_to_physics_server()

        # Set env params and load models
        self._set_physics()
        self._setup_scene()
        self._set_observation_space()
        self._set_action_space()

    def _connect_to_physics_server(self):
        """
        Connect to the PyBullet physics server in SHARED_MEMORY, GUI or DIRECT mode
        """
        if self.gui_on:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
            # if (self.p < 0):
            #     self.p = bc.BulletClient(connection_mode=p.GUI)
            self._set_gui_mode()
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

    def _set_gui_mode(self):
        """
        Set GUI parameters: camera, shadows, extra elements
        """
        self.p.resetDebugVisualizerCamera(1.5, 225, -20, [0.0, 0.4, 0.2])
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, self.shadows_on_gui)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)

    def _set_physics(self):
        """
        Set physics engine parameters
        """
        self.p.setGravity(0, 0, -9.81)
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0.001, numSolverIterations=150, numSubSteps=20, useSplitImpulse=1, collisionFilterMode=1, constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG, globalCFM=0.000001, contactBreakingThreshold=0.001)
        self.p.setTimeStep(self.time_step)
        self.p.setRealTimeSimulation(0)
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        #print(self.p.getPhysicsEngineParameters())

    def _setup_scene(self):
        """
        Set up scene elements (furniture, objects, robots)
        """
        raise NotImplementedError

    def _set_observation_space(self):
        """
        Set limits of observations
        """
        raise NotImplementedError

    def _set_action_space(self):
        """
        Set limits of actions
        """
        raise NotImplementedError

    def _get_observation(self):
        """
        Get info about the state of the environment

        Returns:
            :return observation: (object) Observation of the environment
        """
        raise NotImplementedError

    def step(self, action):
        """
        Apply action on the environment

        Parameters:
            :param action: (object) An action provided by the agent
        Returns:
            :return observation: (object)
            :return reward: (float)
            :return done: (bool):
            :return info: (dict):
        """
        raise NotImplementedError

    def _add_scene_object_uid(self, scene_object_uid, name):
        """
        Call this method in order to enable texturization of object

        Parameters:
            :param scene_object: (int)
        """
        self.scene_objects_uids[scene_object_uid] = name

    def get_scene_object_uid_by_name(self, name):
        for uid, object_name in self.scene_objects_uids.items():
            if name == object_name:
                return uid
        return None

    def seed(self, seed=None):
        """
        Set the seed for this env's random number generator(s)
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def hard_reset(self):
        """
        Full reset of the simulation. Delete and load again all objects and reset physics.
        """
        self.p.resetSimulation()
        self.p.disconnect()
        self._connect_to_physics_server()
        self.scene_objects_uids = {}
        #self.episode_number = 0
        self._set_physics()
        self._setup_scene()

    def _restart_episode(self):
        """
        Reset episode information and delete all objects
        """
        self.p.removeAllUserDebugItems()
        self.episode_start_time = time.time()
        self.episode_over = False
        self.episode_failed = False
        self.episode_reward = 0.0
        self.episode_steps = 0

    def reset(self, hard=False):
        """
        Reset the state of the environment
        """
        if hard:
          self.hard_reset()
        else:
          self._remove_all_objects()

        self._restart_episode()

    def _draw_bounding_boxes(self):
        """
        Show bounding boxes in tne PyBullet GUI
        """
        for object in self.env_objects:
            object.draw_bounding_box()

    def _compute_reward(self):
        """
        Compute reward for the agent
        """
        return NotImplementedError

    def _print_episode_summary(self, info_dict={}):
        """
        Show an extra information about the episode

        Parameters:
            :param info_dict: (dict) Extra info
        """
        if self.episode_failed:
            episode_status = "FAILURE"
        else:
            episode_status = "SUCCESS"

        print("#---------Episode-Summary---------#")
        print("Episode number: " + str(self.episode_number))
        print("Episode's number of steps: " + str(self.episode_steps))
        print("Episode status: " + episode_status)
        print("Episode info: " + self.episode_info)
        print("Episode reward: " + str(self.episode_reward))
        if hasattr(self.reward, "network_rewards"):
                [print("Reward network {}: {}".format(i, x)) for i, x in enumerate(self.reward.network_rewards)]
        print("Last step reward: " + str(self.reward.rewards_history[-1]))
        print("#---------------------------------#")

        #for key, value in info_dict.items():
        #    print(key + ": " + str(value))

    def _get_urdf_filename(self, obj_name):
        """
        Return a matching URDF from directory with objects URDFs
        Parameters:
            :param obj_name: (string) Name of the object
        Returns:
            :return urdf: (string)
        """
        if "virtual" in obj_name:
            return "virtual.urdf"
        for file in self.all_objects_filenames:
            if '/' + obj_name + '.' in file:
                return file
        if self.dataset:
            print("Did not find an urdf for {}, if it is a robot, it is OK".format(obj_name))
            return None
        else:
            raise Exception('Could not match the object name {} with its urdf path'.format(obj_name))

    def _get_random_urdf_filenames(self, n, used_objects=None):
        """
        Sample random URDF files from directory with objects URDFs

        Parameters:
            :param n: (int) Number of URDF's
            :param used_objects: (list) Specified subset of objects
        Returns:
            :return selected_objects_filenames: (list)
        """
        all_objects_filenames = self.all_objects_filenames
        if used_objects:
            all_objects_filenames = []
            for object_name in used_objects:
                if "null" in object_name:
                    all_objects_filenames.append("goal")
                urdf = self._get_urdf_filename(object_name)
                if urdf:
                    all_objects_filenames.append(urdf)
        assert all_objects_filenames is not [], "Could not find any urdf among the objects: {}".format(used_objects)

        selected_objects_filenames = []
        if (n <= len(all_objects_filenames)):
            selected_objects = np.random.choice(
                np.arange(len(all_objects_filenames)), n, replace=True)
        else:
            selected_objects = list(np.arange(len(all_objects_filenames)))
            remain = n - len(all_objects_filenames)
            selected_objects += list(np.random.choice(
                np.arange(len(all_objects_filenames)), remain))
        for object_id in selected_objects:
            selected_objects_filenames.append(all_objects_filenames[object_id])
        return selected_objects_filenames

    def _get_all_urdf_filenames(self, dir):
        """
        Get all URDF filenames from directory

        Parameters:
            :param dir: (int) Number of URDFs
        Returns:
            :return filenames: (list)
        """
        list_all = []
        for (dirpath, dirnames, filenames) in os.walk(self.objects_dir_path):
            if '_old' not in dirpath and 'urdf' in dirpath:
                list_all += [os.path.join(dirpath, file) for file in filenames]
        return list_all

    def _remove_object(self, obj):
        """
        Totally remove object from the simulation

        Parameters:
            :param object: (EnvObject) Object to remove
        """
        assert hasattr(obj, 'get_uid'), "Trying to remove something else than EnvObject"
        self.p.removeBody(obj.get_uid())


    def _remove_all_objects(self):
        """
        Remove all objects from simulation (not scene objects or robots)
        """
        env_objects_copy = self.env_objects.copy()
        for key, o in env_objects_copy.items():
            if isinstance(o, list):
                for i in o:
                  if o not in [self.robot, []]:
                    self._remove_object(i)
            else:
              if o != self.robot:
                self._remove_object(o)

    def get_texturizable_objects_uids(self):
        """
        Get all objects in the environment, on which textures can be applied
        
        Returns:
            :return texturizable_objects_uids: (list)
        """
        uids = []
        for key, val in self.env_objects.items():
            if hasattr(val, "get_uid"):
                uids.append(val.get_uid())
            elif isinstance(val, list):
               for item in val:
                  if hasattr(item, "get_uid"):
                    uids.append(item.get_uid())
        return uids + list(self.scene_objects_uids.keys())

    def get_colorizable_objects_uids(self):
        """
        Get all objects in the environment, which color can be changed

        Returns:
            :return colorizable_objects_uids: (list)
        """
        return [object.get_uid() for object in self.env_objects] + list(self.scene_objects_uids.keys())

    def __del__(self):
        """
        Disconnect from the physics server
        """
        self.p.disconnect()


class CameraEnv(BaseEnv):
    """
    The class for environments with rendering

    Parameters:
        :param camera_resolution: (list) The number of pixels in image (WxH)
        :param shadows_on: (bool) Whether or not to use shadows while rendering, only applies to ER_TINY_RENDERER
        :param render_on: (bool) Turn on rendering
        :param renderer: (int) self.p.ER_TINY_RENDERER (CPU) or self.p.ER_BULLET_HARDWARE_OPENGL (GPU)
        :param active_cameras: (list) Set 1 at a position(=camera number) to save images from this camera
    """
    def __init__(self, camera_resolution=[640, 480], shadows_on=True,
                 render_on=True, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                 active_cameras=None, **kwargs):

        super(CameraEnv, self).__init__(**kwargs)

        self.camera_resolution = camera_resolution
        self.shadows_on = shadows_on
        self.render_on = render_on
        self.renderer = renderer
        self.active_cameras = active_cameras
        self.cameras = []

        self.set_light()
        self._set_cameras()

    def set_light(self, light_direction=[1, 1, 1], light_color=[0.1, 0.1, 0.1],
                  light_distance=1., light_ambient=1., light_diffuse=1.,
                  light_specular=1.):
        """
        Set light parameters for rendering, doesn't affect PyBullet GUI. Appart from light_direction, all parameters only apply to ER_TINY_RENDERER.

        Parameters:
            :param light_direction: (list) Specifies the world position of the light source
            :param light_color: (list) Directional light color in RGB in range 0..1
            :param light_distance: (float) Distance of the light along the normalized light_direction
            :param light_ambient: (float) Light ambient coefficient in range 0..1
            :param light_diffuse: (float) Light diffuse coefficient in range 0..1
            :param light_specular: (float) Light specular coefficient in range 0..1
        """
        self.light_direction = light_direction
        self.light_color = light_color
        self.light_distance = light_distance
        self.light_ambient = light_ambient
        self.light_diffuse = light_diffuse
        self.light_specular = light_specular

    def get_render_parameters(self):
        """
        Return environment parameters for rendering, initially is intended to
        use by cameras

        Returns:
            :return render_parameters: (dict) Render parameters
        """
        return {
            "width": self.camera_resolution[0],
            "height": self.camera_resolution[1],
            "lightDirection": self.light_direction,
            "lightColor": self.light_color,
            "lightDistance": self.light_distance,
            "shadow": 1 if self.shadows_on else 0,
            "lightAmbientCoeff": self.light_ambient,
            "lightDiffuseCoeff": self.light_diffuse,
            "lightSpecularCoeff": self.light_specular,
            "renderer": self.renderer
        }

    def _set_cameras(self):
        """
        Set cameras available to use for rendering
        """
        raise NotImplementedError

    def get_cameras(self):
        return self.cameras

    def add_camera(self, **kwargs):
        """
        Add new camera to the environment

        Parameters:
            :param position: (list) Eye position in Cartesian world coordinates
            :prarm target_position: (list) Position of the target point
            :param up_vector: (list) Up vector of the camera
            :param up_axis_index: (int) Either 1 for Y or 2 for Z axis up
            :param yaw: (float) Yaw angle in degrees left/right around up-axis
            :param pitch: (float) Pitch in degrees up/down
            :param roll: (float) Roll in degrees around forward vector
            :param distance: (float) Distance from eye to focus point
            :param field_of_view: (float) Field of view
            :param near_plane_distance: (float) Near plane distance
            :param far_plane_distance: (float) Far plane distance
        """
        self.cameras.append(Camera(env=self, **kwargs))

    def set_active_cameras(self, active_cameras):

        if (len(active_cameras) == len(self.cameras)):
            self.active_cameras = active_cameras

    def change_current_camera(self, camera_num):
        print("Change camera to " + str(self.current_camera))
        self.current_camera = camera_num

    def render(self, mode="rgb_array", camera_id=None):
        """
        Get image (image, depth, segmentation_mask) from camera or active cameras

        Parameters:
            :param mode: (str) rgb_array to return RGB image
            :param camera_id: (int) Get image from specified camera
        Returns:
            :return camera_data: (dict) Key: camera_id, Value: info from camera
        """
        if mode != "rgb_array":
            return np.array([])
        camera_data = {}
        if self.render_on:
            if camera_id is not None:
                camera_data[camera_id] = self.cameras[camera_id].render()
            else:
                for camera_num in range(len(self.active_cameras)):
                    if self.active_cameras[camera_num]:
                        camera_data[camera_num] = self.cameras[camera_num].render()
        return camera_data

    def project_point_to_camera_image(self, point, camera_id):
        """
        Project 3D point in Cartesian world coordinates to 2D point in pixel space

        Parameters:
            :param point: (list) 3D point in Cartesian world coordinates
            :param camera_id: (int) Index of camera to project on

        Returns:
            :return 2d_point: (list) 2D coordinates of point on imageg
        """
        return self.cameras[camera_id].project_point_to_image(point)

    def get_camera_opencv_matrix_values(self, camera_id):
        """
        Compute values of OpenCV matrix

        Parameters:
            :param camera_id: (int) Index of camera to get matrix from
        Returns:
            :return values: (dict) fx, fy, cx, cy values
        """
        return self.cameras[camera_id].get_opencv_camera_matrix_values()
