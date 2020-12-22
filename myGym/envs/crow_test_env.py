from . import env_object
from .base_env import CameraEnv
import pybullet as p
import time
import numpy as np
from gym import spaces
import os
import pkg_resources
import inspect
currentdir = pkg_resources.resource_filename("myGym", "envs")
repodir = pkg_resources.resource_filename("myGym", "./")

class CrowTestEnv(CameraEnv):

    def __init__(self,
                 used_objects={},
                 **kwargs
                 ):
        super(CrowTestEnv, self).__init__(**kwargs)
        self.used_objects = used_objects
        self.object_init_position = [0, 0, 0.7]
        self.current_object = 0
        self.object_speed = 0.02
        self.ang_speed = 0.03
        self.current_camera = 5
        self.set_control_keys()
        p.resetDebugVisualizerCamera(cameraDistance=0.001, cameraYaw=180,
                                     cameraPitch=-89, cameraTargetPosition=[0.0, 0.0, 2.105])

    def _setup_scene(self):
        self._add_scene_object_uid(p.loadURDF("envs/rooms/plane_with_restitution.urdf", [0, 0, 0]), "floor")
        self._add_scene_object_uid(p.loadURDF("envs/rooms/table.urdf", [0, 0, 0]), "table")

    def _set_observation_space(self):
        observationDim = 6
        observation_high = np.array([100] * observationDim)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high)

    def _set_action_space(self):
        action_dim = 8
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

    def reset(self):
        self._restart_episode()

        self.env_objects = self.init_object(self.object_init_position)

        p.stepSimulation()
        self._observation = self.get_observation()
        return np.array(self._observation)

    def _set_cameras(self):
        """
        sets cameras available in self.cameras
        @param resolution : [width x height] in pix of the render, camera resolution.
        """

        self.add_camera(target_position=[-0.250, 0.715, 1.99], distance=0.001,
                        roll=0, pitch=-20, yaw=90)
        self.add_camera(target_position=[-0.554, 0.945, 2.040], distance=0.001,
                        roll=0, pitch=-30, yaw=180)
        self.add_camera(target_position=[-0.554, 0.455, 2.040], distance=0.001,
                        roll=0, pitch=-30, yaw=0)
        self.add_camera(target_position=[-0.17, 0.720, 2.045], distance=0.001,
                        roll=0, pitch=-90, yaw=-90)
        self.add_camera(target_position=[0.57, 0.720, 2.045], distance=0.001,
                        roll=0, pitch=-20, yaw=90)
        self.add_camera(target_position=[0, 0, 2.105], distance=0.001,
                        roll=0, pitch=-90, yaw=180)
        self.add_camera(target_position=[-0.51, 0.035, 1.345], distance=0.001,
                        roll=0, pitch=0, yaw=-45)
        self.add_camera(target_position=[-0.49, 1.42, 1.345], distance=0.001,
                        roll=0, pitch=0, yaw=-135)

    def get_observation(self):
        # Returns observation as a dictionary with information about robots and objects
        self.observation["objects"] = self.env_objects
        self.observation["camera_data"] = self.render(camera_id=self.current_camera)

        return self.observation

    def step(self, action):
        # Environment step

        if self.render_on:
            time.sleep(self.time_step)
        for obj in self.env_objects:
            obj.bounding_box = obj.get_bounding_box()
            obj.centroid = obj.get_centroid()
        if self.gui_on:
            self._check_keyboard_events()
        if self.gui_on and self.show_bounding_boxes_gui:
            self.draw_bounding_boxes()
        self.observation = self.get_observation()
        return self.observation, None, None, None

    def draw_bounding_boxes(self):
        for object in self.env_objects:
            object.draw_bounding_box()

    def change_object(self, offset=1):
        object = self.env_objects[self.current_object]
        pos = object.get_position()
        orn = object.get_orientation()
        color = object.get_color_rgba()
        p.removeBody(self.env_objects[self.current_object].uid)
        self.current_object += offset
        if self.current_object < 0:
            self.current_object = len(self.env_objects) - 1
        elif self.current_object > len(self.env_objects) - 1:
            self.current_object = self.current_object % len(self.env_objects)
        object = self.env_objects[self.current_object]
        object.set_init_position(pos)
        object.set_init_orientation(orn)
        object.load()
        object.set_color(color)
        p.resetBasePositionAndOrientation(object.uid, pos, orn)

    def set_control_keys(self):
        s = 0.3
        h = self.object_init_position[2]
        self.object_positions = {
          1: [s, -s, h], 2: [0, -s, h], 3: [-s, -s, h],
          4: [s, 0, h], 5: [0, 0, h], 6: [-s, 0, h],
          7: [s, s, h], 8: [0, s, h], 9: [-s, s, h],
        }
        speed = self.object_speed
        ang_speed = self.ang_speed

        self.key_actions = {
          p.B3G_LEFT_ARROW: lambda: self.env_objects[0].move([speed, 0, 0]),
          p.B3G_RIGHT_ARROW: lambda: self.env_objects[0].move([-speed, 0, 0]),
          p.B3G_UP_ARROW: lambda: self.env_objects[0].move([0, -speed, 0]),
          p.B3G_DOWN_ARROW: lambda: self.env_objects[0].move([0, speed, 0]),
          p.B3G_SPACE: lambda: self.env_objects[0].move([0, 0, speed]),
          p.B3G_SHIFT: lambda: self.env_objects[0].move([0, 0, -speed]),
          117: lambda: self.env_objects[0].rotate_euler([ang_speed, 0, 0]), # u
          106: lambda: self.env_objects[0].rotate_euler([-ang_speed, 0, 0]), # j
          105: lambda: self.env_objects[0].rotate_euler([0, ang_speed, 0]), # i
          107: lambda: self.env_objects[0].rotate_euler([0, -ang_speed, 0]), # k
          111: lambda: self.env_objects[0].rotate_euler([0, 0, ang_speed]), # o
          108: lambda: self.env_objects[0].rotate_euler([0, 0, -ang_speed]), # l
        }

        self.key_discrete_actions = {
          99: lambda: self.env_objects[self.current_object].set_random_color(),
          45: lambda: self.change_object(offset=-1), # -
          61: lambda: self.change_object(offset=1), # +
          122: lambda: self.change_current_camera(offset=-1), # z
          120: lambda: self.change_current_camera(offset=1) # x
        }

        shift = 48
        position_actions = {key+shift: lambda key=key: self.env_objects[0].set_position(self.object_positions.get(key)) for key in range(1, 10)}
        self.key_actions.update(position_actions)

    def _check_keyboard_events(self):
        # Check pressed keys
        events = p.getKeyboardEvents()
        print(events)
        for key in self.key_actions:
            if key in events and events[key] & p.KEY_IS_DOWN:
                self.key_actions[key]()

        for key in self.key_discrete_actions:
            if key in events and events[key] & p.KEY_WAS_TRIGGERED:
                self.key_discrete_actions[key]()

    def change_current_camera(self, offset):
        cameras_num = len(self.cameras)
        current_camera = self.current_camera + offset
        if current_camera > cameras_num - 1:
            current_camera %= cameras_num
        elif current_camera < 0:
            current_camera = cameras_num - 1
        print("Change camera to " + str(current_camera))
        self.current_camera = current_camera

    def init_object(self, pos=[0, 0, 1], orn=[0, 0, 0, 1], random_color=True):
        env_objects = []
        init_color = env_object.EnvObject.get_random_color()
        for object_name in self.used_objects:
            if self.used_objects[object_name] > 0:
                object_filename = os.path.join(currentdir, self.objects_dir_path) + object_name + ".urdf"
                object = env_object.EnvObject(object_filename, pos, orn, fixed=True, pybullet_client=self.p)
                p.removeBody(object.uid)
                if random_color:
                    object.set_color(init_color)
                env_objects.append(object)
        if len(env_objects) > 0:
            env_objects[0].load()
        return env_objects
