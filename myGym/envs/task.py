from myGym.envs.vision_module import VisionModule
import matplotlib.pyplot as plt
import pybullet as p
import warnings
import time
import numpy as np
import pkg_resources
import cv2
import random
from scipy.spatial.distance import cityblock
import math
currentdir = pkg_resources.resource_filename("myGym", "envs")


class TaskModule():
    """
    Task module class for task management

    Parameters:
        :param task_type: (string) Type of learned task (reach, push, ...)
        :param num_subgoals: (int) Number of subgoals in task
        :param task_objects: (list of strings) Objects that are relevant for performing the task
        :param distance_type: (string) Way of calculating distances (euclidean, manhattan)
        :param logdir: (string) Directory for logging
        :param env: (object) Environment, where the training takes place
    """
    def __init__(self, task_type='reach', task_objects='cube_holes', num_subgoals=0, observation={},
                 vae_path=None, yolact_path=None, yolact_config=None, distance_type='euclidean',
                 logdir=currentdir, env=None):
        self.task_type = task_type
        self.current_task = 0
        self.distance_type = distance_type
        self.logdir = logdir
        self.task_objects_names = task_objects
        self.num_subgoals = num_subgoals
        self.env = env
        self.image = None
        self.depth = None
        self.last_distance = None
        self.init_distance = None
        self.current_norm_distance = None
        self.stored_observation = []
        self.fig = None
        self.threshold = 0.1 # distance threshold for successful task completion
        self.angle = None
        self.prev_angle = None
        self.pressed = None
        self.turned = None
        self.desired_angle = 57
        self.coefficient_kd = 0
        self.coefficient_kw = 0
        self.coefficient_ka = 0
        self.obs_template = observation
        self.vision_module = VisionModule(observation=observation, env=env, vae_path=vae_path, yolact_path=yolact_path, yolact_config=yolact_config)
        self.vision_src = self.vision_module.src
        self.obsdim = self.check_obs_template()

    def reset_task(self):
        """
        Reset task relevant data and statistics
        """
        self.last_distance = None
        self.init_distance = None
        self.current_norm_distance = None
        self.angle = None
        self.pressed = None
        self.turned = None
        self.vision_module.mask = {}
        self.vision_module.centroid = {}
        self.vision_module.centroid_transformed = {}
        self.env.task_objects["robot"] = self.env.robot
        if self.vision_src == "vae":
            self.generate_new_goal(self.env.objects_area_boarders, self.env.active_cameras)
        self.subgoals = [False]*self.num_subgoals #subgoal completed?
        if self.task_type == '2stepreach':
            self.obs_sub = [[0,2],[0,1]] #objects to have in observation for given subgoal
            self.sub_idx = 0

    def check_obs_template(self):
        """
        Checks if observations are set according to rules and computes observation dim

        Returns:
            :return obsdim: (int) Dimensionality of observation
        """
        t = self.obs_template
        assert "actual_state" and "goal_state" in t.keys(), \
            "Observation setup in config must contain actual_state and goal_state"
        if t["additional_obs"]:
            assert [x in ["joints_xyz", "joints_angles", "endeff_xyz", "endeff_6D", "touch", "distractor"] for x in
                    t["additional_obs"]], "Failed to parse some of the additional_obs in config"
        assert t["actual_state"] in ["endeff_xyz", "obj_xyz", "obj_6D", "vae", "yolact", "voxel", "dope"],\
            "failed to parse actual_state in Observation config"
        assert t["goal_state"] in ["obj_xyz", "obj_6D", "vae", "yolact", "voxel" or "dope"],\
            "failed to parse goal_state in Observation config"
        if "endeff" not in t["actual_state"]:
            assert t["actual_state"] == t["goal_state"], \
                "actual_state and goal_state in Observation must have the same format"
        else:
            if "endeff_xyz" or "endeff_6D" in t["additional_obs"]:
                warnings.warn("Observation config: endeff_xyz already in actual_state, no need to have it in additional_obs. Removing it")
                [self.obs_template["additional_obs"].remove(x) for x in t["additional_obs"] if "endeff" in x]
        obsdim = 0
        for x in [t["actual_state"], t["goal_state"]]:
            if x in ["endeff_xyz", "obj_xyz", "yolact", "voxel"]:
                obsdim += 3
            elif x in ["dope", "obj_6D"]:
                obsdim += 6
            else:
                obsdim += self.vision_module.obsdim
        for x in t["additional_obs"]:
            if x in ["joints_xyz", "joints_angles", "endeff_xyz", "distractor"]:
                obsdim += 3
            elif x == "endeff_6D":
                obsdim += 6
            else:
                obsdim += 1
        return obsdim

    def render_images(self):
        render_info = self.env.render(mode="rgb_array", camera_id=self.env.active_cameras)
        self.image = render_info[self.env.active_cameras]["image"]
        self.depth = render_info[self.env.active_cameras]["depth"]
        if self.env.visualize == 1 and self.vision_src != "vae":
            cv2.imshow("Vision input", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def visualize_vae(self, recons):
        imsize = self.vision_module.vae_imsize
        actual_img, goal_img = [(lambda a: cv2.resize(a[60:390, 160:480], (imsize, imsize)))(a) for a in
                                [self.image, self.goal_image]]
        images = []
        for idx, im in enumerate([actual_img, recons[0], goal_img, recons[1]]):
            im = cv2.copyMakeBorder(im, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.putText(im, ["actual", "actual rec", "goal", "goal rec"][idx], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 0, 0), 1, 0)
            images.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        fig = np.vstack((np.hstack((images[0], images[1])), np.hstack((images[2], images[3]))))
        cv2.imshow("Scene", fig)
        cv2.waitKey(1)

    def get_additional_obs(self, d, robot):
        info = d.copy()
        info["additional_obs"] = {}
        for key in d["additional_obs"]:
            if key == "joints_xyz":
                info["additional_obs"]["joints_xyz"] = self.env.robot.observe_all_links()
            elif key == "joints_angles":
                info["additional_obs"]["joints_angles"] = self.env.robot.get_joints_states()
            elif key == "endeff_xyz":
                info["additional_obs"]["endeff_xyz"] = self.vision_module.get_obj_position(robot, self.image, self.depth)[:3]
            elif key == "endeff_6D":
                info["additional_obs"]["endeff_6D"] = list(self.vision_module.get_obj_position(robot, self.image, self.depth)) \
                                                      + list(self.vision_module.get_obj_orientation(robot))
            elif key == "touch":
                info["additional_obs"]["touch"] = 1 if self.env.robot.touch_sensors_active() else 0
            elif key == "distractor":
                poses = [self.vision_module.get_obj_position(self.env.task_objects["distractor"][x],\
                                    self.image, self.depth) for x in range(len(self.env.task_objects["distractor"]))]
                info["additional_obs"]["distractor"] = [p for sublist in poses for p in sublist]
        return info

    def get_observation(self):
        """
        Get task relevant observation data based on reward signal source

        Returns:
            :return self._observation: (array) Task relevant observation data, positions of task objects 
        """
        info_dict = self.obs_template.copy()
        self.render_images() if "ground_truth" not in self.vision_src else None
        if self.vision_src == "vae":
            [info_dict["actual_state"], info_dict["goal_state"]], recons = (self.vision_module.encode_with_vae(
                imgs=[self.image, self.goal_image], task=self.task_type, decode=self.env.visualize))
            self.visualize_vae(recons) if self.env.visualize == 1 else None
        else:
            for key in ["actual_state", "goal_state"]:
                    if "endeff" in info_dict[key]:
                           xyz = self.vision_module.get_obj_position(self.env.task_objects["robot"], self.image, self.depth)
                    else:
                           xyz = self.vision_module.get_obj_position(self.env.task_objects[key],self.image,self.depth)
                    info_dict[key] = xyz
        self._observation = self.get_additional_obs(info_dict, self.env.task_objects["robot"])
        return self._observation

    def check_vision_failure(self):
        """
        Check if YOLACT vision model fails repeatedly during episode

        Returns:
            :return: (bool)
        """
        self.stored_observation.append(self._observation)
        if len(self.stored_observation) > 9:
            self.stored_observation.pop(0)
            if self.vision_src == "yolact": # Yolact assigns 10 to not detected objects
                if all(10 in obs for obs in self.stored_observation):
                    return True
        return False

    def check_time_exceeded(self):
        """
        Check if maximum episode time was exceeded

        Returns:
            :return: (bool)
        """
        if (time.time() - self.env.episode_start_time) > self.env.episode_max_time:
            self.env.episode_info = "Episode maximum time {} s exceeded".format(self.env.episode_max_time)
            return True
        return False

    def check_object_moved(self, object, threshold=0.3):
        """
        Check if object moved more than allowed threshold

        Parameters:
            :param object: (object) Object to check
            :param threshold: (float) Maximum allowed object movement
        Returns:
            :return: (bool)
        """
        if self.vision_src != "vae":
            object_position = object.get_position()
            pos_diff = np.array(object_position[:2]) - np.array(object.init_position[:2])
            distance = np.linalg.norm(pos_diff)
            if distance > threshold:
                self.env.episode_info = "The object has moved {:.2f} m, limit is {:.2f}".format(distance, threshold)
                return True
        return False

    def check_switch_threshold(self):
        self.angle = self.env.reward.get_angle()

        if abs(self.angle) >= 18:
            return True
        else:
            return False

    def check_press_threshold(self):
        self.pressed = self.env.reward.get_position()
        if self.pressed >= 1.71:
            return True
        else:
            return False

    def check_turn_threshold(self):
        self.turned = self.env.reward.get_angle()
        if self.turned >= self.desired_angle:
            return True
        elif self.turned <= -self.desired_angle:
            return -1
        else:
            return False

    # def check_distance_threshold(self, observation):
    #     """
    #     Check if the distance between relevant task objects is under threshold for successful task completion
    #
    #     Returns:
    #         :return: (bool)
    #     """
    #     observation = observation["observation"] if isinstance(observation, dict) else observation
    #     o1 = observation[0:int(len(observation[:-3])/2)] if self.reward_type == "2dvu" else observation[0:3]
    #     o2 = observation[int(len(observation[:-3])/2):-3]if self.reward_type == "2dvu" else observation[3:6]
    #     self.current_norm_distance = self.calc_distance(o1, o2)
    #     return self.current_norm_distance < self.threshold


    def check_poke_threshold(self, observation):
        """
        Check if the distance between relevant task objects is under threshold for successful task completion

        Returns:
            :return: (bool)
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        goal  = observation[0:3]
        poker = observation[3:6]
        self.current_norm_distance = self.calc_distance(goal, poker)
        return self.current_norm_distance < 0.1

    def check_reach_distance_threshold(self, observation):
        """
        Check if the distance between relevant task objects is under threshold for successful task completion
            Jonášova verze
        Returns:
            :return: (bool)
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        goal    = observation[0:3]
        gripper = self.env.reward.get_accurate_gripper_position(observation[3:6])
        self.current_norm_distance = self.calc_distance(goal, gripper)
        return self.current_norm_distance < self.threshold

    def check_distance_threshold(self, observation):
        """
        Check if the distance between relevant task objects is under threshold for successful task completion
            Jonášova verze
        Returns:
            :return: (bool)
        """
        observation = observation["observation"] if isinstance(observation, dict) else observation
        # goal is first in obs and griper is last (always)
        goal = observation[0:3]
        gripper = self.env.reward.get_accurate_gripper_position(observation[-3:])
        self.current_norm_distance = self.calc_distance(goal, gripper)
        return self.current_norm_distance < self.threshold

    def check_distractor_distance_threshold(self, goal, gripper):
        """
        Check if the distance between relevant task objects is under threshold for successful task completion

        Returns:
            :return: (bool)
        """
        self.current_norm_distance = self.calc_distance(goal, gripper)
        threshold = 0.1
        return self.current_norm_distance < threshold

    def check_points_distance_threshold(self): 
        if (self.task_type == 'pnp') and (self.env.robot_action != 'joints_gripper') and (len(self.env.robot.magnetized_objects) == 0):
            o1 = self.current_task_objects[0]
            o2 = self.current_task_objects[2]
        else:
            o1 = self.current_task_objects[0]
            o2 = self.current_task_objects[1]
        if o1 == self.env.robot:
            closest_points = self.env.p.getClosestPoints(o1.get_uid, o2.get_uid(), self.threshold, o1.end_effector_index, -1)
        elif o2 == self.env.robot:
            closest_points = self.env.p.getClosestPoints(o2.get_uid(), o1.get_uid(), self.threshold, o2.end_effector_index, -1)
        else:
            closest_points = self.env.p.getClosestPoints(o1.get_uid(), o2.get_uid(), self.threshold, -1, -1)
        if len(closest_points) > 0:
            return closest_points
        else:
            return False

    def check_goal(self):
        """
        Check if goal of the task was completed successfully
        """
        self.last_distance = self.current_norm_distance
        if self.init_distance is None:
            self.init_distance = self.current_norm_distance
        finished = None
        if self.task_type == 'reach':
            finished = self.check_distance_threshold(self._observation)
        if self.task_type == 'push' or self.task_type == 'throw' or self.task_type == 'pick_n_place':
            finished = self.check_points_distance_threshold()
        if self.task_type == 'poke':
            finished = self.check_poke_threshold(self._observation)
        if self.task_type == "switch":
            finished = self.check_switch_threshold()
        if self.task_type == "press":
            finished = self.check_press_threshold()
        if self.task_type == "turn":
            finished = self.check_turn_threshold()
        if self.task_type == 'pnp' and self.env.robot_action != 'joints_gripper' and finished:
            if len(self.env.robot.magnetized_objects) == 0:
                self.env.episode_over = False
                self.env.robot.magnetize_object(self.current_task_objects[0], finished)
            else:
                self.env.episode_over = True
                if self.env.episode_steps == 1:
                    self.env.episode_info = "Task completed in initial configuration"
                else:
                    self.env.episode_info = "Task completed successfully"
        elif (self.task_type == '2stepreach') and (False in self.subgoals) and finished:
                self.env.episode_info = "Subgoal {}/{} completed successfully".format(self.sub_idx+1, self.num_subgoals)
                self.subgoals[self.sub_idx] = True #current subgoal done
                self.env.episode_over = False #don't reset episode
                self.env.robot.magnetize_object(self.env.task_objects[self.obs_sub[self.sub_idx][0]], finished) #magnetize first object
                self.sub_idx += 1 #continue with next subgoal
                self.env.reward.reset() #reward reset
        elif finished:
            self.env.episode_over = True
            if self.env.episode_steps == 1:
                self.env.episode_info = "Task completed in initial configuration"
            else:
                self.env.episode_info = "Task completed successfully"
        if self.check_time_exceeded():
            self.env.episode_over = True
            self.env.episode_failed = True
        if self.env.episode_steps == self.env.max_steps:
            if self.task_type == "turn":
                self.env.episode_over = True
                self.env.episode_failed = True
                if self.desired_angle == self.desired_angle-int(self.env.reward.get_angle()):
                    self.env.episode_info = "Angle without change"
                else:
                    self.env.episode_info = f"Remaining angle: {int(self.desired_angle-self.env.reward.get_angle())}"
                if self.check_turn_threshold() == -1:
                    self.env.episode_over = True
                    self.env.episode_failed = True
                    self.env.episode_info = "Bad direction"

            else:
                self.env.episode_over = True
                self.env.episode_failed = True
                self.env.episode_info = "Max amount of steps reached"

        if "ground_truth" not in self.vision_src and (self.check_vision_failure()):
            self.stored_observation = []
            self.env.episode_over = True
            self.env.episode_failed = True
            self.env.episode_info = "Vision fails repeatedly"

    def calc_distance(self, obj1, obj2):
        """
        Calculate distance between two objects

        Parameters:
            :param obj1: (float array) First object position representation
            :param obj2: (float array) Second object position representation
        Returns: 
            :return dist: (float) Distance between 2 float arrays
        """
        if self.distance_type == "euclidean":
            dist = np.linalg.norm(np.asarray(obj1) - np.asarray(obj2))
        elif self.distance_type == "manhattan":
            dist = cityblock(obj1, obj2)
        return dist

    def calc_rotation_diff(self, obj1, obj2):
        """
        Calculate diffrence between orientation of two objects

        Parameters:
            :param obj1: (float array) First object orientation (Euler angles)
            :param obj2: (float array) Second object orientation (Euler angles)
        Returns: 
            :return diff: (float) Distance between 2 float arrays
        """
        if self.distance_type == "euclidean":
            diff = np.linalg.norm(np.asarray(obj1) - np.asarray(obj2))
        elif self.distance_type == "manhattan":
            diff = cityblock(obj1, obj2)
        return diff

    def generate_new_goal(self, object_area_borders, camera_id):
        """
        Generate an image of new goal for VEA vision model. This function is supposed to be called from env workspace.
        
        Parameters:
            :param object_area_borders: (list) Volume in space where task objects can be located
            :param camera_id: (int) ID of environment camera active for image rendering
        """
        if self.task_type == "push":
            random_pos = self.env.task_objects[0].get_random_object_position(object_area_borders)
            random_rot = self.env.task_objects[0].get_random_object_orientation()
            self.env.robot.reset_up()
            self.env.task_objects[0].set_position(random_pos)
            self.env.task_objects[0].set_orientation(random_rot)
            self.env.task_objects[1].set_position(random_pos)
            self.env.task_objects[1].set_orientation(random_rot)
            render_info = self.env.render(mode="rgb_array", camera_id = self.env.active_cameras)
            self.goal_image = render_info[self.env.active_cameras]["image"]
            random_pos = self.env.task_objects[0].get_random_object_position(object_area_borders)
            random_rot = self.env.task_objects[0].get_random_object_orientation()
            self.env.task_objects[0].set_position(random_pos)
            self.env.task_objects[0].set_orientation(random_rot)
        elif self.task_type == "reach":
            bounded_action = [random.uniform(-3,-2.4) for x in range(2)]
            action = [random.uniform(-2.9,2.9) for x in range(6)]
            self.env.robot.reset_joints(bounded_action + action)
            self.goal_image  = self.env.render(mode="rgb_array", camera_id=self.env.active_cameras)[self.env.active_cameras]['image']
            self.env.robot.reset_up()
            #self.goal_image = self.vision_module.vae_generate_sample()

