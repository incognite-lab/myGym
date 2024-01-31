import copy
import itertools
import re
from enum import Enum, auto
from typing import Tuple, List

import numpy as np

from myGym.envs.env_object import EnvObject
import myGym.utils.colors as cs


def _filter_out_none(iterable):
    return [e for e in iterable if e is not None]


def _unpack_1_or_2_element_tuple(t):
    return t if len(t) == 2 else (t[0], None)


def _remove_first_word(s):
    return s[s.find(" ") + 1:]


def _remove_last_word(s):
    return s[:s.rfind(" ")]


class TaskType(Enum):
    REACH = auto(),
    PUSH = auto(),
    PNP = auto(),
    PNPROT = auto(),
    PNPSWIPE = auto(),
    PNPBGRIP = auto(),
    THROW = auto(),
    POKE = auto(),
    PRESS = auto(),
    TURN = auto(),
    SWITCH = auto(),
    FMRT = auto(),
    FMOT = auto(),
    COMPOSITIONAL = auto(),
    DICE_THROW = auto(),

    def to_string(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_string(task: str):
        for entry in TaskType:
            if entry.name.lower() == task:
                return entry
        msg = f"Unknown task type: {task}"
        raise Exception(msg)

    @staticmethod
    def get_pattern_reach_task_types() -> List:
        return [TaskType.REACH]

    @staticmethod
    def get_pattern_press_task_types() -> List:
        return [TaskType.PRESS, TaskType.TURN, TaskType.SWITCH]

    @staticmethod
    def get_pattern_push_task_types() -> List:
        return [TaskType.PUSH, TaskType.PNP, TaskType.PNPROT, TaskType.PNPSWIPE, TaskType.PNPBGRIP, TaskType.THROW, TaskType.POKE, TaskType.FMRT, TaskType.FMOT]


class VirtualObject:
    def __init__(self, obj: EnvObject):
        self.obj: EnvObject = obj
        name = self.obj.get_name()
        self.name = name if "_" not in name else name.split("_", 1)[0]
        self.properties = " ".join(_filter_out_none([cs.rgba_to_name(obj.get_color_rgba())]))

    def __deepcopy__(self, memo={}):
        cp = VirtualObject(self.obj)
        cp.__dict__.update(self.__dict__)
        return cp

    def get_env_object(self) -> EnvObject:
        return self.obj

    def get_name(self) -> str:
        return "the " + self.name

    def get_properties(self) -> str:
        return "the " + self.properties

    def get_name_with_properties(self) -> str:
        return "the " + self.properties + " " + self.name

    @staticmethod
    def extract_object_from_name_with_properties(desc: str, objects: List):
        color_name = _remove_last_word(_remove_first_word(desc))
        object_matches = [o for o in objects if o.properties == color_name]
        if len(object_matches) != 1:
            msg = f"Cannot uniquely determine object, there are {len(object_matches)} objects with description \"{desc}\""
            raise Exception(msg)
        return object_matches[0]

    def get_name_as_unknown_object_with_properties(self) -> str:
        return "the " + self.properties + " object"

    @staticmethod
    def extract_objects_from_unknown_object_with_properties(desc: str, objects: List):
        color_name = _remove_last_word(_remove_first_word(desc))
        object_matches = [o for o in objects if o.properties == color_name]
        if len(object_matches) == 0:
            print(" ".join([o.properties for o in objects]))
            msg = f"There are no objects with description \"{desc}\""
            raise Exception(msg)
        return object_matches

    def get_position(self) -> np.array:
        # TODO: Replace with a virtual position that can be dynamically changed
        return np.array(self.obj.get_position())


class VirtualEnv:
    """
    Internal class for WIP multistep task generation (to virtually simulate object movement).
    """
    def __init__(self, env):
        self.env = env
        self.task_type: TaskType = TaskType.from_string(env.task_type)
        self.objects: List[VirtualObject] = []
        self.real_object_indices: List[int] = []
        self.dummy_object_indices: List[int] = []
        self.set_objects(task_objects=env.task_objects)

    def set_objects(self, task_objects=None, init_goal_objects=None, all_objects=None):
        if bool(task_objects) + bool(init_goal_objects) + bool(all_objects) > 1:
            raise Exception("The only one argument must be passed")

        if task_objects:
            self.objects: List[VirtualObject] = [VirtualObject(o) if isinstance(o, EnvObject) else None for o in self.env.get_task_objects(with_none=True)]
            self.real_object_indices = list(
                range(1, len(self.objects), 2) if self.get_task_type() in TaskType.get_pattern_reach_task_types()
                else (
                    range(0, len(self.objects), 2) if self.get_task_type() in TaskType.get_pattern_push_task_types()
                    else []
                )
            )
        elif init_goal_objects:
            init, goal = init_goal_objects
            self.objects = list(map(VirtualObject, init + goal))
            if TaskType.from_string(self.env.task_type) in TaskType.get_pattern_push_task_types():
                if len(init) == 0 or len(goal) == 0:
                    raise Exception("Not enough real or dummy objects (every group must have at least 1 object)!")
                self.real_object_indices = list(range(len(init)))
                self.dummy_object_indices = list(range(len(init), len(init) + len(goal)))
            else:
                if len(goal) == 0:
                    raise Exception("Not enough real objects!")
                self.real_object_indices = list(range(len(goal)))
        elif all_objects:
            self.objects = list(map(VirtualObject, all_objects))

    def __copy__(self):
        cp = VirtualEnv(self.env)
        cp.__dict__.update(self.__dict__)
        cp.objects = copy.deepcopy(self.objects)
        return cp

    def get_env(self):
        return self.env

    def get_task_type(self) -> TaskType:
        return self.task_type

    def _get_objects(self, indices) -> List[VirtualObject]:
        return [self.objects[i] for i in indices]

    def get_real_objects(self) -> List[VirtualObject]:
        return self._get_objects(self.real_object_indices)

    def get_dummy_objects(self) -> List[VirtualObject]:
        return self._get_objects(self.dummy_object_indices)

    def get_all_objects(self, excluding=None) -> List[VirtualObject]:
        return self.objects if not excluding else [o for o in self.objects if o not in excluding]

    def _get_all_objects_in_relation(self, obj: VirtualObject, relation: str) -> List[VirtualObject]:
        # TODO: Check whether the angle between objects isn't too large
        # TODO: Add the relations above/below
        objects = []
        p1 = obj.get_position()

        for o in self.get_all_objects(excluding=[obj]):
            if o is not obj:
                p2 = o.get_position()
                if relation == "left" and p1[0] > p2[0] or relation == "right" and p1[0] < p2[0]:
                    objects.append(o)

        return objects

    def get_all_objects_left_of(self, obj: VirtualObject) -> List[VirtualObject]:
        return self._get_all_objects_in_relation(obj, "left")

    def get_all_objects_right_of(self, obj: VirtualObject) -> List[VirtualObject]:
        return self._get_all_objects_in_relation(obj, "right")

    def get_subtask_objects(self) -> List[Tuple]:
        return [tuple(_filter_out_none([self.objects[i], self.objects[i + 1]]))
                for i in range(0, len(self.objects), 2)]

    def get_current_subtask_idx(self) -> int:
        return self.env.task.current_task


class NaturalLanguage:
    """
    Class for generating a natural language description and producing new natural language tasks based on the given environment.

    Parameters:
        :param env: (GymEnv) environment to generate description from
        :param seed: (int) seed for NumPy's random generator
    """
    def __init__(self, env, seed=0):
        self.venv: VirtualEnv = VirtualEnv(env)
        self.current_subtask_description: str or None = None
        self.rng = np.random.default_rng(seed)

    def get_venv(self) -> VirtualEnv:
        """
        Return internal virtual environment.

        Returns:
            :return venv: (VirtualEnv) virtual environment
        """
        return self.venv

    def set_current_subtask_description(self, desc: str):
        """
        Set a description for the internal environment.
        """
        self.current_subtask_description = desc

    def get_previously_generated_subtask_description(self) -> str:
        """
        Return a previously generated subtask description.

        Returns:
            :return desc: (str) description, which was previously generated by the corresponding method
        """
        return self.current_subtask_description

    @staticmethod
    def _form_subtask_description(venv: VirtualEnv, *objects_descriptions, task_type: TaskType = None) -> str:
        task_type = venv.get_task_type() if task_type is None else task_type
        d1, d2 = _unpack_1_or_2_element_tuple(objects_descriptions)

        # pattern reach
        if task_type is TaskType.REACH:
            tokens = ["reach", d1]

        # pattern press
        elif task_type is TaskType.PRESS:
            tokens = ["press", d1]
        elif task_type is TaskType.TURN:
            tokens = ["turn", d1]
        elif task_type is TaskType.SWITCH:
            tokens = ["switch", d1]

        # pattern push
        elif task_type is TaskType.PUSH:
            tokens = ["push", d1, "to the same position as", d2]
        elif task_type is TaskType.PNP:
            tokens = ["pick", d1, "and place it to the same position as", d2]
        elif task_type is TaskType.PNPROT:
            tokens = ["pick", d1, "and rotate it to the same position as", d2]
        elif task_type is TaskType.PNPSWIPE:
            tokens = ["swipe", d1, "along the line to the position of", d2]
        elif task_type is TaskType.PNPBGRIP:
            bgrip = " with mechanic gripper" if "bgrip" in venv.get_env().robot.get_name() else ""
            tokens = ["pick", d1, "and place it" + bgrip, d2]
        elif task_type is TaskType.THROW:
            tokens = ["throw", d1, "to the same position as", d2]
        elif task_type is TaskType.POKE:
            tokens = ["poke", d1, "to the same position as", d2]
        else:
            exc = f"Unknown task type {task_type}"
            raise Exception(exc)

        return " ".join(tokens)

    @staticmethod
    def _decompose_subtask_description(desc: str):
        # pattern reach
        if desc.startswith("reach"):
            return TaskType.REACH, _remove_first_word(desc)

        elif desc.startswith("push") or desc.startswith("throw") or desc.startswith("poke"):
            task_type = TaskType.PUSH if desc.startswith("push") else (TaskType.THROW if desc.startswith("throw") else TaskType.POKE)
            return task_type, _remove_first_word(desc).split(" to the same position as ")
        if desc.startswith("pick") and "rotate" not in desc:
            return TaskType.PNP, _remove_first_word(desc).split(" and place it to the same position as ")
        elif desc.startswith("pick") and "rotate" in desc:
            return TaskType.PNPROT, _remove_first_word(desc).split(" and rotate it to the same position as ")
        elif desc.startswith("swipe"):
            return TaskType.PNPSWIPE, _remove_first_word(desc).split(" along the line to the position of ")
        else:
            msg = f"Cannot determine the task type: {desc}"
            raise Exception(msg)

    @staticmethod
    def _get_object_descriptions(venv: VirtualEnv, obj: VirtualObject):
        if venv.get_env().reach_gesture:
            return ["here", "there"]

        descs = [obj.get_name_with_properties()]
        for o in venv.get_all_objects_left_of(obj):
            descs.append(" ".join([obj.get_name_as_unknown_object_with_properties(), "right to", o.get_name_with_properties()]))
        for o in venv.get_all_objects_right_of(obj):
            descs.append(" ".join([obj.get_name_as_unknown_object_with_properties(), "left to", o.get_name_with_properties()]))
        return descs

    @staticmethod
    def _extract_object_from_object_description(venv: VirtualEnv, desc: str) -> VirtualObject:
        all_objects = venv.get_all_objects()

        if "left" in desc or "right" in desc:
            is_left = "left" in desc
            descs = desc.split(" left to " if is_left else " right to ")
            d1, d2 = descs[0], descs[1]

            objects_with_same_color = VirtualObject.extract_objects_from_unknown_object_with_properties(d1, all_objects)
            o2 = VirtualObject.extract_object_from_name_with_properties(d2, all_objects)

            if len(objects_with_same_color) == 1:
                return objects_with_same_color[0]
            else:
                objects_in_relation = venv.get_all_objects_left_of(o2) if is_left else venv.get_all_objects_right_of(o2)
                object_matches = list(set(objects_with_same_color) & set(objects_in_relation))

                if len(object_matches) == 1:
                    return object_matches[0]
                else:
                    msg = f"Error, there are {len(object_matches)} objects with description \"{desc}\""
                    raise Exception(msg)
        elif "here" in desc or "there" in desc:
            objects = [vo.get_env_object() for vo in venv.get_all_objects()]
            return VirtualObject(venv.get_env().human.find_object_human_is_pointing_at(objects=objects))
        else:
            return VirtualObject.extract_object_from_name_with_properties(desc, all_objects)

    def generate_subtask_with_random_description(self) -> None:
        """
        Generate the description of the current subtask and save it to the internal variable.
        """
        task_type = self.venv.get_task_type()
        assert task_type in TaskType.get_pattern_push_task_types() or task_type == TaskType.REACH  # TODO: Implement the remaining tasks

        if task_type in TaskType.get_pattern_push_task_types():
            d1 = self.rng.choice(self._get_object_descriptions(self.venv, self.rng.choice(self.venv.get_real_objects())))
            d2 = self.rng.choice(self._get_object_descriptions(self.venv, self.rng.choice(self.venv.get_dummy_objects())))
            self.current_subtask_description = self._form_subtask_description(self.venv, d1, d2)
        else:
            o2 = self.rng.choice(self.venv.get_real_objects())
            env = self.venv.get_env()

            if env.reach_gesture:
                if env.training:
                    for _ in range(1):
                        env.human.point_finger_at(position=o2.get_env_object().get_position())
                        env.p.stepSimulation()
                    o2 = env.human.find_object_human_is_pointing_at(objects=self.venv.get_real_objects())
                else:
                    objects = [vo.get_env_object() for vo in self.venv.get_all_objects()]
                    o2 = VirtualObject(env.choose_goal_object_by_human_with_keys(objects=objects))

            d2 = self.rng.choice(self._get_object_descriptions(self.venv, o2))
            self.current_subtask_description = self._form_subtask_description(self.venv, d2)

    def extract_subtask_info_from_description(self, desc: str) -> Tuple[str, str, int, EnvObject, EnvObject]:
        """
        Given a list of environment objects (using one of the methods above), extract from a natural language
        description all the information needed for reproducing the task.

        Parameters:
            :param desc: (str) natural language description of the subtask
        Returns:
            :return task_type, reward, n_nets, init, goal: (tuple) task type, corresponding reward, number of neural networks,
            initial object, goal object
        """
        desc = re.sub(' +', ' ', desc.strip().lower())
        task_type, descs = self._decompose_subtask_description(desc)
        assert task_type in TaskType.get_pattern_push_task_types() or task_type == TaskType.REACH

        if task_type in TaskType.get_pattern_push_task_types():
            d1, d2 = descs[0], descs[1]
            init = self._extract_object_from_object_description(self.venv, d1)
            goal = self._extract_object_from_object_description(self.venv, d2)
        else:
            d2 = descs
            init = None
            goal = self._extract_object_from_object_description(self.venv, d2)

        if task_type is TaskType.REACH or task_type is TaskType.PUSH:
            reward, n_nets = "distance", 1
        else:
            reward, n_nets = task_type.to_string(), 3

        return task_type.to_string(), reward, n_nets, init.get_env_object() if init is not None else init, goal.get_env_object()

    def generate_random_description_for_current_subtask(self) -> None:
        """
        Generate a random description (with random object relations) for the current subtask and save it internally.
        """
        o1, o2 = _unpack_1_or_2_element_tuple(self.venv.get_subtask_objects()[self.venv.get_current_subtask_idx()])
        task_type = self.venv.get_task_type()
        d1 = self.rng.choice(self._get_object_descriptions(self.venv, o1))
        d2 = self.rng.choice(self._get_object_descriptions(self.venv, o2)) if o2 is not None else None
        ds = (d1, d2) if task_type in TaskType.get_pattern_push_task_types() else (d1,)
        self.set_current_subtask_description(self._form_subtask_description(self.venv, *ds, task_type=task_type))

    def generate_task_description(self) -> str:
        """
        Deprecated function. Have been used for producing a description of a multistep task. But the function could be
        updated to satisfy the new interface of the NL module.
        """
        subtasks = []
        task_type = self.venv.get_task_type()

        for objects in self.venv.get_subtask_objects():
            o1, o2 = objects if len(objects) == 2 else (objects[0], None)

            if task_type in TaskType.get_pattern_reach_task_types():
                subtasks.append(NaturalLanguage._form_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name_with_properties()))
            elif task_type in TaskType.get_pattern_press_task_types():
                subtasks.append(NaturalLanguage._form_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name()))
            else:
                subtasks.append(NaturalLanguage._form_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name_with_properties(), "to " + o2.get_name_with_properties()))

        return ", ".join(subtasks)

    def _generate_new_subtasks_from_1_env(self, task_desc: str, venv: VirtualEnv) -> List[Tuple[str, VirtualEnv]]:
        """
        Deprecated function. Have been used for generating new subtasks based on the current environment state. The function
        can be updated and used in the future.
        """
        tuples = []
        main_clauses, main_clauses_with_to, place_preposition_clauses = NaturalLanguage._get_movable_object_clauses(venv)

        if task_desc is not "":
            task_desc += ", "

        # pattern reach
        for c in itertools.chain(main_clauses, place_preposition_clauses):
            tuples.append((task_desc + NaturalLanguage._form_subtask_description(venv, TaskType.REACH, c), venv))

        # pattern push
        task_types = [TaskType.PUSH, TaskType.PNP, TaskType.POKE, TaskType.THROW]
        for c2 in itertools.chain(main_clauses_with_to, place_preposition_clauses):
            for tt, c1 in zip(task_types, self.rng.choice(main_clauses, len(task_types), replace=True)):
                tuples.append((task_desc + NaturalLanguage._form_subtask_description(venv, tt, c1, c2), venv))

        return tuples

    def _generate_new_subtasks(self, tuples: List[Tuple[str, VirtualEnv]]) -> List[Tuple[str, VirtualEnv]]:
        """
        Deprecated function. Have been used for generating new subtasks based on the current environment state.
        Specifically, this function takes tuples of environment-NL description and produces new tuples
        extended by a 1 new subtask. The function can be updated to satisfy new NL mode interface and can be used in the future.
        """
        return list(itertools.chain(*[self._generate_new_subtasks_from_1_env(*t) for t in tuples]))

    def generate_new_tasks(self, max_tasks=10, max_subtasks=3) -> List[str]:
        """
        Deprecated in a new NL mode interface, but it can be updated for reuse.
        Generate new natural language tasks from the given environment.

        Parameters:
            :param max_tasks: (int) Maximum number of tasks. If it is possible to generate more tasks, the function will randomly pick max_tasks tasks
            :param max_subtasks: (int) Maximum number of subtasks in one task
        Returns:
            :return new_tasks: (list) List of newly generated tasks
        """
        if self.venv.get_task_type() in TaskType.get_pattern_press_task_types():
            raise NotImplementedError()

        tuples = [("", self.venv)]
        for i in range(max_subtasks):
            tuples = self._generate_new_subtasks(tuples)
            if len(tuples) > max_tasks:
                tuples = self.rng.choice(tuples, max_tasks, replace=False)

        return [t[0] for t in tuples]
