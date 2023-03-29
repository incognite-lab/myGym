import copy
import itertools
from enum import Enum, auto
from typing import Tuple, List

import numpy as np

from myGym.envs.gym_env import GymEnv
from myGym.envs.env_object import EnvObject
import myGym.utils.colors as cs


def _filter_out_none(iterable):
    return [e for e in iterable if e is not None]


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

    @classmethod
    def from_string(cls, task: str):
        for entry in cls:
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
        self.properties = " ".join(_filter_out_none([self._infer_size_property(), cs.rgba_to_name(obj.get_color_rgba())]))

    def __deepcopy__(self, memo={}):
        cp = VirtualObject(self.obj)
        cp.__dict__.update(self.__dict__)
        return cp

    def _infer_size_property(self):
        v = np.prod(self.obj.get_cuboid_dimensions())

        if v < 0.0001:
            return "tiny"
        elif v < 0.0003:
            return "small"
        elif v < 0.0015:
            return None
        else:
            return "big"

    def get_name(self) -> str:
        return "the " + self.name

    def get_name_with_properties(self) -> str:
        return "the " + self.properties + " " + self.name


class VirtualEnv:
    def __init__(self, env: GymEnv):
        self.env: GymEnv = env
        self.task_type: TaskType = TaskType.from_string(env.task_type)
        self.objects: List[VirtualObject] = [
            VirtualObject(o) if isinstance(o, EnvObject) else None for o in
            [env.task_objects["actual_state"], env.task_objects["goal_state"]] +
            (env.task_objects["distractor"] if "distractor" in env.task_objects else [])
        ]
        self.movable_object_indices = list(
            range(1, len(self.objects), 2) if self.get_task_type() in TaskType.get_pattern_reach_task_types()
            else (
                range(0, len(self.objects), 2) if self.get_task_type() in TaskType.get_pattern_push_task_types()
                else []
            )
        )

    def __copy__(self):
        cp = VirtualEnv(self.env)
        cp.__dict__.update(self.__dict__)
        cp.objects = copy.deepcopy(self.objects)
        return cp

    def get_real_env(self) -> GymEnv:
        return self.env

    def get_task_type(self) -> TaskType:
        return self.task_type

    def get_subtask_objects(self) -> List[Tuple]:
        return [tuple(_filter_out_none([self.objects[i], self.objects[i + 1]]))
                for i in range(0, len(self.objects), 2)]

    def _get_objects(self, indices) -> List[VirtualObject]:
        return [self.objects[i] for i in indices]

    def get_movable_objects(self) -> List[VirtualObject]:
        return self._get_objects(self.movable_object_indices)


class NaturalLanguage:
    """
    Class for generating a natural language description and producing new natural language tasks based on the given environment.
    """
    def __init__(self, env: GymEnv, seed=0):
        self.venv: VirtualEnv = VirtualEnv(env)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _generate_subtask_description(venv: VirtualEnv, task: TaskType, *place_clauses) -> str:
        c1, c2 = place_clauses if len(place_clauses) == 2 else (place_clauses[0], None)

        # pattern reach
        if task is TaskType.REACH:
            tokens = ["reach", c1]

        # pattern press
        elif task is TaskType.PRESS:
            tokens = ["press", c1]
        elif task is TaskType.TURN:
            tokens = ["turn", c1]
        elif task is TaskType.SWITCH:
            tokens = ["switch", c1]

        # pattern push
        elif task is TaskType.PUSH:
            tokens = ["push", c1, c2]
        elif task is TaskType.PNP:
            tokens = ["pick", c1, "and place it", c2]
        elif task is TaskType.PNPROT:
            tokens = ["pick", c1, ", place it", c2, "and rotate it"]
        elif task is TaskType.PNPSWIPE:
            tokens = ["pick", c1, "and swiping place it", c2]
        elif task is TaskType.PNPBGRIP:
            bgrip = " with mechanic gripper" if "bgrip" in venv.get_real_env().robot.get_name() else ""
            tokens = ["pick", c1, "and place it" + bgrip, c2]
        elif task is TaskType.THROW:
            tokens = ["throw", c1, c2]
        elif task is TaskType.POKE:
            tokens = ["poke", c1, c2]
        else:
            exc = f"Unknown task type {task}"
            raise Exception(exc)

        return " ".join(tokens)

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
                subtasks.append(NaturalLanguage._generate_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name_with_properties()))
            elif task_type in TaskType.get_pattern_press_task_types():
                subtasks.append(NaturalLanguage._generate_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name()))
            else:
                subtasks.append(NaturalLanguage._generate_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name_with_properties(), "to " + o2.get_name_with_properties()))

        return ", ".join(subtasks)

    @staticmethod
    def _get_movable_object_clauses(venv: VirtualEnv) -> Tuple[List[str], List[str], List[str]]:
        objects = venv.get_movable_objects()
        main_clauses = [o.get_name_with_properties() for o in objects]
        place_preposition_clauses = []

        for obj in objects:
            for preposition in ["left to", "right to", "close to", "above"]:
                place_preposition_clauses.append(preposition + " " + obj.get_name_with_properties())

        for i in range(len(objects) - 1):
            for j in range(i + 1, len(objects)):
                place_preposition_clauses.append("between " + objects[i].get_name_with_properties() + " and " + objects[j].get_name_with_properties())

        return main_clauses, ["to " + c for c in main_clauses], place_preposition_clauses

    def _generate_new_subtasks_from_1_env(self, task_desc: str, venv: VirtualEnv) -> List[Tuple[str, VirtualEnv]]:
        tuples = []
        main_clauses, main_clauses_with_to, place_preposition_clauses = NaturalLanguage._get_movable_object_clauses(venv)

        if task_desc is not "":
            task_desc += ", "

        # pattern reach
        for c in itertools.chain(main_clauses, place_preposition_clauses):
            tuples.append((task_desc + NaturalLanguage._generate_subtask_description(venv, TaskType.REACH, c), venv))

        # pattern push
        task_types = [TaskType.PUSH, TaskType.PNP, TaskType.POKE, TaskType.THROW]
        for c2 in itertools.chain(main_clauses_with_to, place_preposition_clauses):
            for tt, c1 in zip(task_types, self.rng.choice(main_clauses, len(task_types), replace=True)):
                tuples.append((task_desc + NaturalLanguage._generate_subtask_description(venv, tt, c1, c2), venv))

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
