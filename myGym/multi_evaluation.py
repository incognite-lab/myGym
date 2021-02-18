import os
import re
import threading
import subprocess
from sklearn.model_selection import ParameterGrid
import json

parameters = {
    "robot_action": ["joints", "step", "absolute"],
    "task_type": ["reach", "push", "pnp"],
    "max_episode_steps": [256, 512, 1024]
}
parameter_grid = ParameterGrid(parameters)
configfile = 'configs/train.json'
evaluation_results_paths = [None] * len(parameter_grid)
last_eval_results = {}


def train(params, i):
    if params["task_type"] != "reach":
        params["task_objects"] = "cube_holes target"
    print((" ".join(f"--{key} {value}" for key, value in params.items())).split())
    command = 'python train.py --config {configfile} '.format(configfile=configfile) + " ".join(f"--{key} {value}" for key, value in params.items())
    output = subprocess.check_output(command.split())
    evaluation_results_paths[i] = output.splitlines()[-1].decode('UTF-8') + "/evaluation_results.json"
    # os.system('python train.py --config {configfile} '.format(configfile=configfile)
    #           + " ".join(f"--{key} {value}" for key, value in params.items()))


if __name__ == '__main__':
    threads = []
    for i, params in enumerate(parameter_grid):
        train(params.copy(), i)
        with open(evaluation_results_paths[i]) as f:
            data = json.load(f)
        last_eval_results[str(list(params.values()))] = list(data.values())[-1]
        # thread = threading.Thread(target=train, args=(params, i))
        # thread.start()
        # threads.append(thread)

    # for thread in threads:
    #     thread.join()

    print(last_eval_results)
    with open("trained_models/multi_evaluation_results.json", 'w') as f:
        json.dump(last_eval_results, f, indent=4)
