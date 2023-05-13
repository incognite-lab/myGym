import copy
import itertools
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
        return [TaskType.PUSH, TaskType.PNP, TaskType.PNPROT, TaskType.PNPSWIPE, TaskType.PNPBGRIP, TaskType.THROW, TaskType.POKE]


class VirtualObject:
    def __init__(self, obj: EnvObject):
        self.obj: EnvObject = obj
        self.name = obj.get_name()
        self.properties = " ".join(_filter_out_none([cs.rgba_to_name(obj.get_color_rgba())]))

    def __deepcopy__(self, memo={}):
        cp = VirtualObject(self.obj)
        cp.__dict__.update(self.__dict__)
        return cp

    def get_env_object(self) -> EnvObject:
        return self.obj

    def get_name(self) -> str:
        return "the " + self.name

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
            msg = f"There are no objects with description \"{desc}\""
            raise Exception(msg)
        return object_matches

    def get_position(self) -> np.array:
        # TODO: Replace with a virtual position that can be dynamically changed
        return np.array(self.obj.get_position())


class VirtualEnv:
    def __init__(self, env):
        self.env = env
        self.task_type: TaskType = TaskType.from_string(env.task_type)
        assert self.task_type not in TaskType.get_pattern_press_task_types()  # not implemented yet
        self.objects: List[VirtualObject] = []
        self.real_object_indices: List[int] = []
        self.dummy_object_indices: List[int] = []
        self.set_objects(task_objects=env.task_objects)

    def set_objects(self, task_objects=None, init_goal_objects=None, all_objects=None):
        if bool(task_objects) + bool(init_goal_objects) + bool(all_objects) > 1:
            raise Exception("The only one argument must be passed")

        if task_objects:
            self.objects: List[VirtualObject] = [
                VirtualObject(o) if isinstance(o, EnvObject) else None for o in
                [self.env.task_objects["actual_state"], self.env.task_objects["goal_state"]] +
                (self.env.task_objects["distractor"] if "distractor" in self.env.task_objects else [])
            ]
            self.real_object_indices = list(
                range(1, len(self.objects), 2) if self.get_task_type() in TaskType.get_pattern_reach_task_types()
                else (
                    range(0, len(self.objects), 2) if self.get_task_type() in TaskType.get_pattern_push_task_types()
                    else []
                )
            )
        elif init_goal_objects:
            init, goal = init_goal_objects
            if len(init) == 0 or len(goal) == 0:
                raise Exception("Not enough real or dummy objects (every group must have at least 1 object)!")
            self.objects = list(map(VirtualObject, init + goal))
            self.real_object_indices = list(range(len(init)))
            self.dummy_object_indices = list(range(len(init), len(init) + len(goal)))
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
    """
    def __init__(self, env, seed=0):
        self.venv: VirtualEnv = VirtualEnv(env)
        self.current_subtask_description: str or None = None
        self.rng = np.random.default_rng(seed)

    def get_venv(self) -> VirtualEnv:
        return self.venv

    def get_current_subtask_description(self) -> str:
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
            tokens = ["push", d1, d2]
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
            tokens = ["throw", d1, d2]
        elif task_type is TaskType.POKE:
            tokens = ["poke", d1, d2]
        else:
            exc = f"Unknown task type {task_type}"
            raise Exception(exc)

        return " ".join(tokens)

    @staticmethod
    def _decompose_subtask_description(desc: str):
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
        else:
            return VirtualObject.extract_object_from_name_with_properties(desc, all_objects)

    def generate_random_subtask_with_random_description(self):
        task_type = self.venv.get_task_type()
        assert task_type in [TaskType.PNP, TaskType.PNPROT, TaskType.PNPSWIPE]  # TODO: Implement the remaining tasks
        init = self.rng.choice(self.venv.get_real_objects())
        goal = self.rng.choice(self.venv.get_dummy_objects())
        d1 = self.rng.choice(self._get_object_descriptions(self.venv, init))
        d2 = self.rng.choice(self._get_object_descriptions(self.venv, goal))
        self.current_subtask_description = self._form_subtask_description(self.venv, d1, d2)

    def extract_subtask_info_from_description(self, desc: str):
        task_type, descs = self._decompose_subtask_description(desc)
        d1, d2 = descs[0], descs[1]
        init = self._extract_object_from_object_description(self.venv, d1)
        goal = self._extract_object_from_object_description(self.venv, d2)
        return task_type.to_string(), 3, init.get_env_object(), goal.get_env_object()

    @staticmethod
    def _get_movable_object_clauses(venv: VirtualEnv) -> Tuple[List[str], List[str], List[str]]:
        objects = venv.get_real_objects()
        main_clauses = [o.get_name_with_properties() for o in objects]
        place_preposition_clauses = []

        for obj in objects:
            for preposition in ["left to", "right to", "close to", "above"]:
                place_preposition_clauses.append(preposition + " " + obj.get_name_with_properties())

        for i in range(len(objects) - 1):
            for j in range(i + 1, len(objects)):
                place_preposition_clauses.append("between " + objects[i].get_name_with_properties() + " and " + objects[j].get_name_with_properties())

        return main_clauses, ["to " + c for c in main_clauses], place_preposition_clauses

    def generate_current_subtask_description(self):
        o1, o2 = _unpack_1_or_2_element_tuple(self.venv.get_subtask_objects()[self.venv.get_current_subtask_idx()])
        task_type = self.venv.get_task_type()
        c1 = self.rng.choice(self._get_object_descriptions(self.venv, o1))
        c2 = self.rng.choice(self._get_object_descriptions(self.venv, o2, as_goal=True)) if o2 is not None else None

        if task_type in TaskType.get_pattern_reach_task_types():
            return self._form_subtask_description(self.venv, task_type, c1)
        elif task_type in TaskType.get_pattern_push_task_types():
            return self._form_subtask_description(self.venv, task_type, c1, c2)
        else:
            return self._form_subtask_description(self.venv, task_type, c1)

    def generate_task_description(self) -> str:
        """
        Generate a natural language description for the environment task.
        Warning: in multistep tasks must be called during the 1-st subtask
        (due to the assumption about object's order in GymEnv.task_objects), otherwise the behaviour is undefined.

        Returns:
            :return description: (string) Natural language description
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
        return list(itertools.chain(*[self._generate_new_subtasks_from_1_env(*t) for t in tuples]))

    def generate_new_tasks(self, max_tasks=10, max_subtasks=3) -> List[str]:
        """
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
