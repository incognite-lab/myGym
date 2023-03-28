from enum import Enum, auto
from typing import Tuple, List

import numpy as np

from myGym.envs.gym_env import GymEnv
from myGym.envs.env_object import EnvObject
import myGym.utils.colors as cs


def _filter_out_none(iterable):
    return [e for e in iterable if e is not None]


def _concatenate_clauses(clauses, with_and=False):
    n = len(clauses)
    if n == 1:
        return clauses[0]
    elif n == 2:
        return " and ".join(clauses) if with_and else ", ".join(clauses)
    elif n > 2:
        return _concatenate_clauses([", ".join(clauses[:-1]), clauses[-1]], with_and)
    else:
        exc = "No clauses to concatenate"
        raise Exception(exc)


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


class VirtualObject:
    def __init__(self, obj: EnvObject):
        self.obj: EnvObject = obj
        self.name = obj.get_name()
        self.properties = " ".join(_filter_out_none([self._infer_size_property(), cs.rgba_to_name(obj.get_color_rgba())]))

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

    def get_name(self):
        return self.name

    def get_properties(self):
        return self.properties

    def get_name_with_properties(self):
        return self.properties + " " + self.name


class VirtualEnv:
    def __init__(self, env: GymEnv):
        self.env: GymEnv = env
        self.task_type: TaskType = TaskType.from_string(env.task_type)
        all_task_objects = [
            VirtualObject(o) if isinstance(o, EnvObject) else None for o in
            [env.task_objects["actual_state"], env.task_objects["goal_state"]] +
            (env.task_objects["distractor"] if "distractor" in env.task_objects else [])
        ]
        self.subtask_objects: List[Tuple] = [
            tuple(_filter_out_none([all_task_objects[i], all_task_objects[i + 1]]))
            for i in range(0, len(all_task_objects), 2)
        ]

    def get_real_env(self) -> GymEnv:
        return self.env

    def get_task_type(self) -> TaskType:
        return self.task_type

    def get_subtask_objects(self) -> List[Tuple]:
        return self.subtask_objects


class NaturalLanguage:
    """
    Class for generating a natural language description for a given environment task
    and producing new natural language tasks based on a given environment.
    """
    def __init__(self, env: GymEnv):
        self.venv: VirtualEnv = VirtualEnv(env)

    def _get_subtask_description(self, task: TaskType, *place_clauses):
        c1, c2 = place_clauses if len(place_clauses) == 2 else (place_clauses[0], None)

        # pattern 1
        if task == TaskType.REACH:
            tokens = ["reach", c1]

        # pattern 2
        elif task == TaskType.PRESS:
            tokens = ["press", c1]
        elif task == TaskType.TURN:
            tokens = ["turn", c1]
        elif task == TaskType.SWITCH:
            tokens = ["switch", c1]

        # pattern 3
        elif task == TaskType.PUSH:
            tokens = ["push", c1, c2]
        elif task == TaskType.PNP:
            tokens = ["pick", c1, "and place it", c2]
        elif task == TaskType.PNPROT:
            tokens = ["pick", c1, ", place it", c2, "and rotate it"]
        elif task == TaskType.PNPSWIPE:
            tokens = ["pick", c1, "and swiping place it", c2]
        elif task == TaskType.PNPBGRIP:
            bgrip = " with mechanic gripper" if "bgrip" in self.venv.get_real_env().robot.get_name() else ""
            return ["pick", c1, "and place it" + bgrip, c2]
        elif task == TaskType.THROW:
            tokens = ["throw", c1, c2]
        elif task == TaskType.POKE:
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
        subtask_descriptions = []
        task_type = self.venv.get_task_type()

        for objects in self.venv.get_subtask_objects():
            o1, o2 = objects if len(objects) == 2 else (objects[0], None)

            if task_type is TaskType.REACH:
                subtask_descriptions.append(self._get_subtask_description(task_type, "the " + o1.get_name_with_properties()))
            elif task_type in [TaskType.PRESS, TaskType.TURN, TaskType.SWITCH]:
                subtask_descriptions.append(self._get_subtask_description(task_type, "the " + o1.get_name()))
            else:
                subtask_descriptions.append(self._get_subtask_description(
                    task_type, "the " + o1.get_name_with_properties(), "to the " + o2.get_name_with_properties()
                ))

        return _concatenate_clauses(subtask_descriptions)

    def generate_new_tasks(self) -> List[str]:
        raise NotImplementedError()
