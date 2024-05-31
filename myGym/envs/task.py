import pybullet as p
import random
import myGym.envs.scene_objects
from rddl.rddl_sampler import RDDLWorld
import pkg_resources, os

class TaskModule():
    """
    Task module class for task management

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task_num_range: (list of ints) range of allowed number of subtasks
        :param allowed_objects: (list of strings) names of allowed object classes
        :param allowed_predicates: (list of strings) names of allowed predicates

    """
    def __init__(self, env=None, num_task_range=[], allowed_protoactions=[], allowed_objects=[], allowed_predicates=[], pybullet_client=None):
        self.env = env
        self.p = pybullet_client
        self.number_tasks = num_task_range
        self.allowed_protoactions = allowed_protoactions
        self.allowed_objects = allowed_objects
        self.allowed_predicates = allowed_predicates
        self.current_task_length = None
        self.current_task_sequence = None
        self.subtask_over = False
        self.current_task = None
        self.scene_entities = []
        self.scene_objects = []
        self.rddl_world = RDDLWorld()

    def sample_num_subtasks(self):
        self.current_task_length = random.randint(*self.number_tasks)
        return self.current_task_length
    
    def create_new_task_sequence(self):
        n_samples = self.sample_num_subtasks()
        task_sequence = self.rddl_world.sample_generator(n_samples)
        self.current_task_sequence = task_sequence
        print("Generated a new action sequence")

    def get_next_task(self):
        if self.current_task_sequence == None:
            self.create_new_task_sequence()
        self.current_task = next(self.current_task_sequence)
        print(f"Current task: {self.current_task}")
        print("Desired world state after action:")
        self.rddl_world.show_world_state()

    def build_scene_for_task(self):
        if self.current_task is None:
            self.get_next_task()
        scene_entities = self.rddl_world.get_created_variables()
        for entity in scene_entities:
            if "Gripper" in str(entity.type):
                pass
            else:
                pos = entity.type.get_random_object_position(self.env.reachable_borders)
                orn =  entity.type.get_random_z_rotation()
                kw = {"env": self.env, "position":pos, "orientation":orn, "pybullet_client":self.p, "fixed":False, "observation":self.env.vision_source,
                      "vae_path":self.env.vae_path, "yolact_path":self.env.yolact_path, "yolact_config":self.env.yolact_config}
                o = entity.type(**kw)
                entity.bind(o)
                if o.rgba is None:
                    o.set_color(self.env.get_random_color())
                else:
                    o.set_color(o.get_color_rgba())
                self.scene_objects.append(o)
                self.scene_entities.append(entity)


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
        self.current_task_length = None
        self.current_task_sequence = None
        self.subtask_over = False
        self.current_task = None