import re
from typing import List

import numpy as np

from myGym.envs.gym_env import GymEnv
from myGym.envs.env_object import EnvObject
import myGym.utils.colors as cs


def _concatenate_clauses(clauses, with_and=False):
    n = len(clauses)
    if n == 1:
        return clauses[0]
    elif n == 2:
        return ' and '.join(clauses) if with_and else ', '.join(clauses)
    elif n > 2:
        return _concatenate_clauses([', '.join(clauses[:-1]), clauses[-1]], with_and)
    else:
        exc = 'No clauses to concatenate'
        raise Exception(exc)


def _remove_extra_spaces(iterable):
    return list(map(lambda s: re.sub(' +', ' ', s).strip(), iterable))


def _get_tuple(lst, i):
    return lst[i * 2], lst[i * 2 + 1]


class Language:
    """
    Class for generating a natural language description for a given environment task
    and producing new natural language tasks based on a given environment.
    """
    def __init__(self, env: GymEnv):
        self.env: GymEnv = env
        self.task: str = self.env.task_type
        self.all_objects = [self.env.task_objects['actual_state'], self.env.task_objects['goal_state']] + \
                           (self.env.task_objects['distractor'] if 'distractor' in self.env.task_objects else [])

    @staticmethod
    def _volume_to_str(v):
        if v > 0.0015:
            return 'big'
        elif v > 0.0003:
            return ''
        elif v > 0.0001:
            return 'small'
        else:
            return 'tiny'

    def _to_clause(self, name_tuple, property_tuple):
        n1, n2 = name_tuple
        pn1, pn2 = property_tuple[0] + ' ' + n1, property_tuple[1] + ' ' + n2

        if self.task == 'reach':
            tokens = ['reach the', pn2]
        elif self.task == 'push':
            tokens = ['push the', pn1, 'to the', n2]
        elif self.task == 'pnp':
            tokens = ['pick the', pn1, 'and place it to the', n2]
        elif self.task == 'pnprot':
            tokens = ['pick the', pn1 + ',', 'place it to the', n2, 'and rotate it']
        elif self.task == 'pnpswipe':
            tokens = ['pick the', pn1, 'and swiping place it to the', n2]
        elif self.task == 'pnpbgrip':
            bgrip = ' with mechanic gripper ' if 'bgrip' in self.env.robot.get_name() else ' '
            return ['pick the', pn1, 'and place it' + bgrip + 'to the', n2]
        elif self.task == 'press':
            tokens = ['press the', pn2]
        elif self.task == 'poke':
            tokens = ['poke the', pn1, 'to the', n2]
        elif self.task == 'switch':
            tokens = ['switch the', pn2]
        elif self.task == 'throw':
            tokens = ['throw the', pn1, 'to the', n2]
        elif self.task == 'turn':
            tokens = ['turn the', pn2]
        else:
            exc = f'Unknown task type {self.task}'
            raise Exception(exc)

        return ' '.join(tokens)

    def generate_task_description(self) -> str:
        """
        Generate a natural language description for the environment task.
        Warning: in multistep tasks must be called during the 1-st subtask
        (due to the assumption about object's order in GymEnv.task_objects), otherwise the behaviour is undefined.

        Returns:
            :return description: (string) Natural language description
        """
        names = list(map(lambda o: o.get_name() if isinstance(o, EnvObject) else '', self.all_objects))
        colors = list(map(lambda o: cs.rgba_to_name(o.get_color_rgba()) if isinstance(o, EnvObject) else '', self.all_objects))
        sizes = list(map(lambda o: Language._volume_to_str(np.prod(o.get_cuboid_dimensions())) if isinstance(o, EnvObject) else '', self.all_objects))
        properties = _remove_extra_spaces(map(' '.join, zip(sizes, colors)))
        clauses = [self._to_clause(_get_tuple(names, i), _get_tuple(properties, i)) for i in range(len(self.all_objects) // 2)]
        return _concatenate_clauses(clauses)

    def generate_new_tasks(self) -> List[str]:
        raise NotImplementedError()
