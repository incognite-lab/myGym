import pybullet as p
import random
import myGym.envs.scene_objects
from rddl.rddl_sampler import RDDLWorld

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
        self.rddl_world = RDDLWorld()

    def sample_num_subtasks(self):
        self.current_task_length = random.randint(*self.number_tasks)
        return self.current_task_length
    
    def create_new_task_sequence(self):
        n_samples = self.sample_num_subtasks()
        task_sequence = self.rddl_world.sample_generator(n_samples)
        self.current_task_sequence = task_sequence

    def get_next_task(self):
        if self.current_task_sequence == None:
            self.create_new_task_sequence()
        self.current_task = next(self.current_task_sequence)

        # while True:
        #     try:
        #         action = next(gen)
        #     except StopIteration:
        #         break
        #     print(f"Generated action: {action}")
        #     print("World state after action:")
        #     rddl_world.show_world_state()
        #     print("")
        #     actions.append(action)

        # variables = rddl_world.get_created_variables()

        # str_actions = '\n\t'.join([repr(a) for a in actions])
        # print(f"Actions:\n\t{str_actions}")
        # str_variables = '\n\t'.join([repr(v) for v in variables])
        # print(f"Variables:\n\t{str_variables}")
        # print("> Initial state:")
        # rddl_world.show_initial_world_state()
        # print("> Goal state:")
        # rddl_world.show_goal_world_state()

    def build_scene_for_task(self):
        if self.current_task is None:
            self.get_next_task()
        scene_entities = self.rddl_world.get_created_variables()
        for entity in scene_entities:
            if "Gripper" in str(entity.type):
                pass
            else:
                entity.type(env=self.env, pybullet_client=self.p)

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