"""
Module for generating a natural language description for a given environment task.
"""
import re
from typing import List

import numpy as np

from myGym.envs.env_object import EnvObject
import myGym.utils.colors as cs
from myGym.envs.gym_env import GymEnv


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


def _to_clause(env, names, properties):
    task = env.task_type
    n1, n2 = names
    pn1, pn2 = properties[0] + ' ' + n1, properties[1] + ' ' + n2

    if task == 'reach':
        tokens = ['reach the', pn2]
    elif task == 'push':
        tokens = ['push the', pn1, 'to the', n2]
    elif task == 'pnp':
        tokens = ['pick the', pn1, 'and place it to the', n2]
    elif task == 'pnprot':
        tokens = ['pick the', pn1 + ',', 'place it to the', n2, 'and rotate it']
    elif task == 'pnpswipe':
        tokens = ['pick the', pn1, 'and swiping place it to the', n2]
    elif task == 'pnpbgrip':
        bgrip = ' with mechanic gripper ' if 'bgrip' in env.robot.get_name() else ' '
        return ['pick the', pn1, 'and place it' + bgrip + 'to the', n2]
    elif task == 'press':
        tokens = ['press the', pn2]
    elif task == 'poke':
        tokens = ['poke the', pn1, 'to the', n2]
    elif task == 'switch':
        tokens = ['switch the', pn2]
    elif task == 'throw':
        tokens = ['throw the', pn1, 'to the', n2]
    elif task == 'turn':
        tokens = ['turn the', pn2]
    else:
        exc = f'Unknown task type {task}'
        raise Exception(exc)

    return ' '.join(tokens)


def _volume_to_str(v):
    if v > 0.0015:
        return 'big'
    elif v > 0.0003:
        return ''
    elif v > 0.0001:
        return 'small'
    else:
        return 'tiny'


def _remove_extra_spaces(iterable):
    return list(map(lambda s: re.sub(' +', ' ', s).strip(), iterable))


def _get_tuple(lst, i):
    return lst[i * 2], lst[i * 2 + 1]


def generate_task_description(env: GymEnv) -> str:
    """
    Generate a natural language description for a given environment task.
    Warning: in multistep tasks must be called during the 1-st subtask
    (due to the assumption about object's order in GymEnv.task_objects), otherwise the behaviour is undefined.

    Parameters:
        :param env: (GymEnv) GymEnv instance to generate description from
    Returns:
        :return description: (string) Natural language description
    """
    objects = [env.task_objects['actual_state'], env.task_objects['goal_state']] + (env.task_objects['distractor'] if 'distractor' in env.task_objects else [])
    names = list(map(lambda o: o.get_name() if isinstance(o, EnvObject) else '', objects))
    colors = list(map(lambda o: cs.rgba_to_name(o.get_color_rgba()) if isinstance(o, EnvObject) else '', objects))
    sizes = list(map(lambda o: _volume_to_str(np.prod(o.get_cuboid_dimensions())) if isinstance(o, EnvObject) else '', objects))
    properties = _remove_extra_spaces(map(' '.join, zip(sizes, colors)))
    clauses = [_to_clause(env, _get_tuple(names, i), _get_tuple(properties, i)) for i in range(len(objects) // 2)]
    return _concatenate_clauses(clauses)


def generate_new_tasks(env: GymEnv) -> List[str]:
    raise NotImplementedError()
