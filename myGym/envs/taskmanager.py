import pybullet as p
import warnings
import numpy as np
import pkg_resources


class TaskModule():
    """
    Task module class for task management

    Parameters:
        :param task_type: (string) Type of learned task (reach, push, ...)
        :param task_objects: (list of strings) Objects that are relevant for performing the task
        :param distance_type: (string) Way of calculating distances (euclidean, manhattan)
        :param logdir: (string) Directory for logging
        :param env: (object) Environment, where the training takes place
    """
    def __init__(self, env=None, number_tasks=None):
        self.env = env
        self.number_tasks = number_tasks
        self.subtask_over = False
        self.current_task = None

    def get_world_state(self):
        """
        Get all objects in the scene including scene objects, interactive objects and robot

        Returns:
            :return observation: (dict) Task relevant observation data, positions of task objects 
        """
        pass


    def check_goal(self):
        """
        Check if goal of the task was completed successfully
        """
        pass

    def check_end_episode(self):
        """
        Check if episode should finish based on fulfilled goal or exceeded number of steps
        """
        pass



    def reset_task(self):
        """
        Start a new episode with a new setup
        """
        pass