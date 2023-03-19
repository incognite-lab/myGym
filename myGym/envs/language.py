from myGym.envs.env_object import EnvObject


def concatenate_clauses(clauses, with_and=False):
    """
    Concatenate clauses using the comma and 'and'

    Parameters:
        :param clauses: (list) List of strings to concatenate
        :param with_and: (bool) Whether to use 'and' for concatenation of the last two clauses
    Returns:
        :return clause: (string) Concatenated clause
    """
    n = len(clauses)
    if n == 1:
        return clauses[0]
    elif n == 2:
        return ' and '.join(clauses) if with_and else ', '.join(clauses)
    elif n > 2:
        return concatenate_clauses([', '.join(clauses[:-1]), clauses[-1]], with_and)
    else:
        exc = 'No clauses to concatenate'
        raise Exception(exc)


class Language:
    """
    The class for generating a language description for a task
    """
    @staticmethod
    def _extract_object_colors(env):
        objects = [env.task_objects['actual_state'], env.task_objects['goal_state']] + \
                  (env.task_objects['distractor'] if 'distractor' in env.task_objects else [])

        # exclude objects other than EnvObject (e.g. Robot)
        colors = [env.colors.rgba_to_name(o.get_color_rgba()) if isinstance(o, EnvObject) else None for o in objects]

        # separate into init-goal tuples
        color_tuples = []
        for i, c in enumerate(colors):
            if i % 2 == 0:
                color_tuples.append((c, None))
            else:
                color_tuples[i // 2] = (color_tuples[i // 2][0], c)

        return color_tuples

    @staticmethod
    def generate_description(env) -> str:
        """
        Generate description of the environment task in the natural language

        Parameters:
            :param env: (GymEnv) GymEnv instance to generate description from
        Returns:
            :return description: (string) Description in the natural language
        """
        task = env.task_type
        colors = Language._extract_object_colors(env)

        def to_clause(objects, colors):
            o1, o2 = objects
            c1, c2 = colors

            if task == 'reach':
                tokens = ['reach the', c2, o2]
            elif task == 'push':
                tokens = ['push the', c1, o1, 'to the', o2]
            elif task == 'pnp':
                tokens = ['pick the', c1, o1, 'and place it to the', o2]
            elif task == 'pnprot':
                tokens = ['pick the', c1, o1 + ',', 'place it to the', o2, 'and rotate it']
            elif task == 'pnpswipe':
                tokens = ['pick the', c1, o1, 'and swiping place it to the', o2]
            elif task == 'press':
                tokens = ['press the', c2, o2]
            elif task == 'poke':
                tokens = ['poke the', c1, o1, 'to the', o2]
            elif task == 'switch':
                tokens = ['switch the', c2, o2]
            elif task == 'throw':
                tokens = ['throw the', c1, o1, 'to the', o2]
            elif task == 'turn':
                tokens = ['turn the', c2, o2]
            else:
                exc = f'Unknown task type {task}'
                raise Exception(exc)

            return ' '.join(tokens)

        init_goal = [(d['init']['obj_name'], d['goal']['obj_name']) for d in env.task_objects_dict]
        return concatenate_clauses([to_clause(init_goal[i], colors[i]) for i in range(len(init_goal))])
