"""
Module for generating a natural language description for a given environment task.
"""
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


def _extract_object_colors(env):
    objects = [env.task_objects['actual_state'], env.task_objects['goal_state']] + \
              (env.task_objects['distractor'] if 'distractor' in env.task_objects else [])

    # exclude objects other than EnvObject (e.g. Robot)
    colors = [cs.rgba_to_name(o.get_color_rgba()) if isinstance(o, EnvObject) else None for o in objects]

    # separate into init-goal tuples
    color_tuples = []
    for i, c in enumerate(colors):
        if i % 2 == 0:
            color_tuples.append((c, None))
        else:
            color_tuples[i // 2] = (color_tuples[i // 2][0], c)

    return color_tuples


def _to_clause(task, objects, properties):
    o1, o2 = objects
    p1, p2 = properties

    if task == 'reach':
        tokens = ['reach the', p2, o2]
    elif task == 'push':
        tokens = ['push the', p1, o1, 'to the', o2]
    elif task == 'pnp':
        tokens = ['pick the', p1, o1, 'and place it to the', o2]
    elif task == 'pnprot':
        tokens = ['pick the', p1, o1 + ',', 'place it to the', o2, 'and rotate it']
    elif task == 'pnpswipe':
        tokens = ['pick the', p1, o1, 'and swiping place it to the', o2]
    elif task == 'pnpbgrip':
        raise NotImplementedError()
    elif task == 'press':
        tokens = ['press the', p2, o2]
    elif task == 'poke':
        tokens = ['poke the', p1, o1, 'to the', o2]
    elif task == 'switch':
        tokens = ['switch the', p2, o2]
    elif task == 'throw':
        tokens = ['throw the', p1, o1, 'to the', o2]
    elif task == 'turn':
        tokens = ['turn the', p2, o2]
    else:
        exc = f'Unknown task type {task}'
        raise Exception(exc)

    return ' '.join(tokens)


def generate_description(env) -> str:
    """
    Generate a natural language description for a given environment task.
    Warning: in multistep tasks must be called during the 1-st subtask
    (due to the assumption about object's order in GymEnv.task_objects), otherwise the behaviour is undefined.

    Parameters:
        :param env: (GymEnv) GymEnv instance to generate description from
    Returns:
        :return description: (string) Natural language description
    """
    objects = [(d['init']['obj_name'], d['goal']['obj_name']) for d in env.task_objects_dict]
    colors = _extract_object_colors(env)
    clauses = [_to_clause(env.task_type, objects[i], colors[i]) for i in range(len(objects))]
    return _concatenate_clauses(clauses)
