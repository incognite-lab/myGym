import random
import myGym.envs.scene_objects # allows binding with rddl
from myGym.envs.scene_objects import TiagoGripper
from rddl.rddl_sampler import RDDLWorld
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
        self.scene_entities = []
        self.scene_objects = []
        self.rddl_world = RDDLWorld(allowed_actions=self.cfg_strings_to_classes(self.allowed_protoactions),
                                    allowed_entities=self.cfg_strings_to_classes(self.allowed_objects) + [TiagoGripper],
                                    allowed_initial_actions=self.cfg_strings_to_classes(self.allowed_protoactions))

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
        task_sequence = self.rddl_world.sample_generator(n_samples)
        self.current_task_sequence = task_sequence
        print("Generated a new action sequence")

    def get_next_task(self):
        '''When the previous action (subtask) is finished, this function will jump to the next one.'''
        if self.current_task_sequence == None:
            # if there is no action sequence to follow, first make a new one
            self.create_new_task_sequence()
        self.current_task = next(self.current_task_sequence)
        print(f"Current task: {self.current_task}")
        print("Desired world state after action:")
        self.rddl_world.show_world_state()

    def build_scene_for_task_sequence(self):
        '''After a new sequence of actions (subtasks) is created, this function will create the physical scene according to the
        symbolic template. '''
        if self.current_task == None:
            # if there is no action sequence, first make a new one
            self.get_next_task()
        scene_entities = self.rddl_world.get_created_variables()
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

    def check_vision_failure(self):
        """
        Check if YOLACT vision model fails repeatedly during episode

        Returns:
            :return: (bool)
        """
        self.stored_observation.append(self._observation["actual_state"])
        self.stored_observation.append(self._observation["goal_state"])
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

    def check_episode_steps(self):
        """
        Check if maximum episode steps was exceeded

        Returns:
            :return: (bool)
        """
        if self.env.episode_steps == self.env.max_episode_steps:
            self.end_episode_fail("Max amount of steps reached")
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

    def check_turn_threshold(self, desired_angle=57):
        turned = self.env.reward.get_angle()
        if turned >= desired_angle:
            return True
        elif turned <= - desired_angle:
            return -1
        return False

    def check_distance_threshold(self, observation, threshold=0.1):
        """
        Check if the distance between relevant task objects is under threshold for successful task completion
        Returns:
            :return: (bool)
        """
        self.current_norm_distance = self.calc_distance(observation["goal_state"], observation["actual_state"])
        return self.current_norm_distance < threshold

    def check_distrot_threshold(self, observation, threshold=0.1):
        """
        Check if the distance between relevant task objects is under threshold for successful task completion
        Returns:
            :return: (bool)
        """
        self.current_norm_distance = self.calc_distance(observation["goal_state"], observation["actual_state"])
        self.current_norm_rotation = self.calc_rot_quat(observation["goal_state"], observation["actual_state"])

        if self.current_norm_distance < threshold and self.current_norm_rotation < threshold:
            return True
        return False


    def get_dice_value(self, quaternion):
        def noramalize(q):
            return q/np.linalg.norm(q)

        faces = np.array([
            [0,0,1],
            [0,1,0],
            [1,0,0],
            [0,0,-1],
            [0,-1,0],
            [-1,0,0],
        ])

        rot_mtx = Rotation.from_quat(noramalize(quaternion)).as_matrix()

        rotated_faces = np.dot(rot_mtx, faces.T).T
        top_face_index = np.argmax(rotated_faces[:,2])

        #face_nums = [2, 5, 1, 4, 6, 3] #states that first face has number 2 on it, second 5 and so on...
        #return face_nums[top_face_index]
        return top_face_index+1

    def check_dice_moving(self, observation, threshold=0.1):
        def calc_still(o1, o2):
            result = 0
            for i in range(len(o1)):
                result+= np.power(o1[i]-o2[i],2)
            result = np.sqrt(result)

            return result < 0.000005
        #print(observation["goal_state"])
        if len(observation)<3:
            print("Invalid",observation)
        x = np.array(observation["actual_state"][3:])

        #print(observation)

        if not self.check_distance_threshold(self._observation) and self.env.episode_steps > 25:
            if calc_still(observation["actual_state"], self.stored_observation):
                if (self.stored_observation == observation["actual_state"]):
                    return 0
                else:
                    if self.writebool:
                        print(self.get_dice_value(x))
                        print(observation)
                        self.writebool = False
                    if self.get_dice_value(x) == 6:
                        return 2
                    return 1

            else:
                self.stored_observation = observation["actual_state"]
                return 0
        else:
            self.stored_observation = observation["actual_state"]
            self.writebool = True
            #print(self.get_dice_value(x))
            return 0

    def check_points_distance_threshold(self, threshold=0.1):
        o1 = self.env.task_objects["actual_state"]
        if (self.task_type == 'pnp') and (self.env.robot_action != 'joints_gripper') and (len(self.env.robot.magnetized_objects) == 0):
            o2 = self.env.robot
            closest_points = self.env.p.getClosestPoints(o2.get_uid(), o1.get_uid(), threshold,
                                                         o2.end_effector_index, -1)
        else:
            o2 = self.env.task_objects["goal_state"]
            idx = -1 if o1 != self.env.robot else self.env.robot.end_effector_index
            closest_points = self.env.p.getClosestPoints(o1.get_uid(), o2.get_uid(), threshold, idx, -1)
        return closest_points if len(closest_points) > 0 else False

    def drop_magnetic(self):
        """
        Release the object if required point was reached and controls if task was compleated.
        Returns:
            :return: (bool)
        """
        if self.env.reward.point_was_reached:
            if not self.env.reward.was_dropped:
                self.env.episode_over = False
                self.env.robot.release_all_objects()
                self.env.task.subtask_over = True
                self.current_task = 0
                self.env.reward.was_dropped = True
        # print("drop episode", self.env.reward.drop_episode, "episode steps", self.env.episode_steps)
        if self.env.reward.drop_episode and self.env.reward.drop_episode + 35 < self.env.episode_steps:
            self.end_episode_success()
            return True
        else:
            return False

    def check_goal(self):
        """
        Check if goal of the task was completed successfully
        """

        finished = None
        if self.task_type in ['reach', 'poke', 'pnp', 'pnpbgrip', 'FMOT', 'FROM', 'FROT', 'FMOM', 'FM','F','A','AG','AGM','AGMD','AGMDW']: #all tasks ending with R (FMR) have to have distrot checker
            finished = self.check_distance_threshold(self._observation)
        if self.task_type in ['pnprot','pnpswipe','FMR', 'FMOR', 'FMLFR', 'compositional']:
            finished = self.check_distrot_threshold(self._observation)
        if self.task_type in ["dropmag"]: #FMOT should be compositional
            self.check_distance_threshold(self._observation)
            finished = self.drop_magnetic()
        if self.task_type in ['push', 'throw']:
            self.check_distance_threshold(self._observation)
            finished = self.check_points_distance_threshold()
        if self.task_type == "switch":
            self.check_distance_threshold(self._observation)
            finished = abs(self.env.reward.get_angle()) >= 18
        if self.task_type == "press":
            self.check_distance_threshold(self._observation)
            finished = self.env.reward.get_angle() >= 1.71
        if self.task_type == "dice_throw":
            finished = self.check_dice_moving(self._observation)

        if self.task_type == "turn":
            self.check_distance_threshold(self._observation)
            finished = self.check_turn_threshold()
        self.last_distance = self.current_norm_distance
        if self.init_distance is None:
            self.init_distance = self.current_norm_distance
        #if self.task_type == 'pnp' and self.env.robot_action != 'joints_gripper' and finished:
        #    if len(self.env.robot.magnetized_objects) == 0 and self.env.episode_steps > 5:
        #        self.end_episode_success()
        #    else:
        #        self.env.episode_over = False
        if finished:
            if self.task_type == "dice_throw":

                if finished == 1:
                    self.end_episode_fail("Finished with wrong dice result thrown")
                return finished
            self.end_episode_success()
        if self.check_time_exceeded() or self.env.episode_steps == self.env.max_episode_steps:
            self.end_episode_fail("Max amount of steps reached")
        if "ground_truth" not in self.vision_src and (self.check_vision_failure()):
            self.stored_observation = []
            self.end_episode_fail("Vision fails repeatedly")

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