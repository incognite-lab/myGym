def concatenate_clauses(clauses, with_and=False):
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
    def __init__(self):
        pass

    def generate_description(self, env) -> str:
        task = env.task_type

        def to_clause(d):
            if task == 'reach':
                return ' '.join([task, d[1]])
            elif task == 'push':
                return ' '.join([task, d[0], 'to the', d[1]])
            elif task == 'pnp':
                return ' '.join(['pick', d[0], 'and place it to the', d[1]])
            elif task == 'pnprot':
                return ' '.join(['pick', d[0] + ',', 'place it to the', d[1], 'and rotate it'])
            elif task == 'pnpswipe':
                raise NotImplementedError()
            elif task == 'press':
                return ' '.join([task, 'the', d[1]])
            elif task == 'poke':
                raise NotImplementedError()
            elif task == 'switch':
                return ' '.join([task, 'the', d[1]])
            elif task == 'throw':
                raise NotImplementedError()
            elif task == 'turn':
                return ' '.join([task, 'the', d[1]])
            else:
                exc = f'Unknown task type {task}'
                raise Exception(exc)

        init_goal = [(d['init']['obj_name'], d['goal']['obj_name']) for d in env.task_objects_dict]
        return concatenate_clauses(list(map(to_clause, init_goal)))
