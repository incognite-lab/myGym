import random
import myGym.envs.scene_objects # allows binding with rddl
from myGym.envs.scene_objects import TiagoGripper
from rddl.rddl_sampler import RDDLWorld
from rddl.task import RDDLTask
from rddl.entities import Gripper, ObjectEntity

class TaskModule():
    """
    Task module class for task management with RDDL

    Parameters:
        :param env: (object) Environment, where the training takes place
        :param task_num_range: (list of ints) range of allowed number of subtasks
        :param allowed_objects: (list of strings) names of allowed object classes
        :param allowed_predicates: (list of strings) names of allowed predicates
        :param pybullet_client: (object) pybullet instance
    """
    def __init__(self, env=None, num_task_range=[], allowed_protoactions=[], allowed_objects=[], allowed_predicates=[], pybullet_client=None):
        self.env = env
        self.p = pybullet_client
        self.number_tasks_range = num_task_range
        self.allowed_protoactions = allowed_protoactions
        self.allowed_objects = allowed_objects
        self.allowed_predicates = allowed_predicates
        self.current_task_length = None
        self.current_task_sequence = None
        self.subtask_over = False
        self.rddl_robot = None
        self.current_task = None
        self.current_action = None
        self.scene_entities = []
        self.scene_objects = []
        self.rddl_world = RDDLWorld(allowed_actions=self.cfg_strings_to_classes(self.allowed_protoactions), 
                                    allowed_entities=self.cfg_strings_to_classes(self.allowed_objects) + [TiagoGripper], 
                                    allowed_initial_actions=self.cfg_strings_to_classes(self.allowed_protoactions))
        #self.rddl_task = RDDLTask()

    def sample_num_subtasks(self):
        '''Whenever a new sequence of actions is desired, this function chooses a random
          length of the sequence based on the range from config. '''
        self.current_task_length = random.randint(*self.number_tasks_range)
        return self.current_task_length
    
    def cfg_strings_to_classes(self, cfg:list):
        """Looks up the classes corresponding to names in cfg and returns a list of them.
        Args:
            cfg (list): list of strings with names of entities, e.g., ["Approach", "Drop"] or ["Banana", "Tuna"]
        Raises:
            Exception: When the class is not found by rddl, that usually means it is not defined in scene_objects.py
        Returns:
            _type_: list of classes
        """
        list_of_classes = []
        for n in cfg:
            n_fixed = n[0].upper() + n[1:].lower()
            try:
                cl = getattr(myGym.envs.scene_objects, n_fixed)
                list_of_classes.append(cl)
            except:
                raise Exception("Entity {} not found by RDDL! Check scene_objects.py for this class".format(n_fixed))
        return list_of_classes

    def create_new_task_sequence(self):
        '''Calls rddl to make a new sequence of actions (subtasks).'''
        n_samples = self.sample_num_subtasks()
        # task_sequence = self.rddl_world.sample_generator(n_samples)
        self.rddl_task = self.rddl_world.sample_world(n_samples)
        self.current_task_sequence = self.rddl_task.get_generator()
        print("Generated a new action sequence")

    def get_next_task(self):
        '''When the previous action (subtask) is finished, this function will jump to the next one.'''
        if self.current_task_sequence is None:
            # if there is no action sequence to follow, first make a new one
            self.create_new_task_sequence()
        self.current_action = next(self.current_task_sequence)
        print(f"Current task: {self.current_action}")
        print("Desired world state after action:")
        self.rddl_task.show_current_state()

    def build_scene_for_task_sequence(self):
        '''After a new sequence of actions (subtasks) is created, this function will create the physical scene according to the
        symbolic template. '''
        if self.current_action is None:
            # if there is no action sequence, first make a new one
            self.get_next_task()
        # scene_entities = self.rddl_world.get_created_variables()
        scene_entities = self.rddl_task.gather_objects()
        for entity in scene_entities:
            if issubclass(entity.type, Gripper) and not entity.is_bound():
                robot = entity.type(self.env.robot_type, robot_action=self.env.robot_action, task_type=self.env.task_type, **self.env.robot_kwargs)
                entity.bind(robot)
                self.rddl_robot = entity
                self.scene_objects.append(robot)
            elif not entity.is_bound():
                pos = entity.type.get_random_object_position(self.env.reachable_borders) # @TODO needs to be constrained by predicates
                orn =  entity.type.get_random_z_rotation() #@TODO needs to be constrained by predicates

                kw = {"env": self.env, "position":pos, "orientation":orn, "pybullet_client":self.p, "fixed":False, "observation":self.env.vision_source,
                    "vae_path":self.env.vae_path, "yolact_path":self.env.yolact_path, "yolact_config":self.env.yolact_config}
                self.spawn_object_for_rddl(entity, **kw)

                

    def spawn_object_for_rddl(self, rddl_entity, env, position, orientation, pybullet_client, 
                              fixed=False, observation="ground_truth", vae_path=None, yolact_path=None, yolact_config=None):
        """Initializes EnvObject and adds it to the scene. The object will be bound to the symbolic entity provided as rddl_entity.

        Args:
            rddl_entity (Variable): unbound rddl entity to bind with
            env (class instance): env instance
            position (list): position of the object [x, y, z]
            orientation (tuple): orientation of the object in quaternions (x1, x2, x3, x4)
            pybullet_client (object): instance of pybullet simulator
            fixed (bool): whether object has fixed base or not
            observation (str): vision source ("ground_truth", "yolact", "vae")
            vae_path (str): path to vae model, if used
            yolact_path (str): path to yolact model, if used
            yolact_config (str): path to saved yolact config
        """
        kw = {"env": env, "position":position, "orientation":orientation, "pybullet_client":pybullet_client, "fixed":fixed, 
              "observation":observation, "vae_path":vae_path, "yolact_path":yolact_path, "yolact_config":yolact_config}
        o = rddl_entity.type(**kw) # initialize EnvObject and add to scene
        rddl_entity.bind(o) # bind spawned object to the symbolic one
        if o.rgba is None: # if no colour bound to object, assign random rgb
            o.set_color(self.env.get_random_color())
        else:
            o.set_color(o.get_color_rgba()) # assign correct colour to object as defined in scene_objects.py
        self.scene_objects.append(o)
        self.scene_entities.append(rddl_entity)
        

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
        self.current_action = None