import os
import re
import threading
import subprocess
from sklearn.model_selection import ParameterGrid

parameters = {
    "robot_action": ["joints", "step", "absolute"],
    "task_type": ["reach", "push", "pnp"],
    "max_episode_steps": [256, 512, 1024]
}
parameter_grid = ParameterGrid(parameters)
configfile = 'configs/train.json'
evaluation_results = [None] * len(parameter_grid)


def train(params, i):
    if params["task_type"] != "reach":
        params["task_objects"] = "cube_holes target"
    print((" ".join(f"--{key} {value}" for key, value in params.items())).split())
    command = 'python train.py --config {configfile} '.format(configfile=configfile) + " ".join(f"--{key} {value}" for key, value in params.items())
    #command = re.sub(r'(?<=,) ', '', command)
    output = subprocess.check_output(command.split()) # space which is not preceded by comma
    evaluation_results[i] = output.splitlines()[-1].decode('UTF-8') + "/evaluation_results.json"
    # os.system('python train.py --config {configfile} '.format(configfile=configfile)
    #           + " ".join(f"--{key} {value}" for key, value in params.items()))


if __name__ == '__main__':
    threads = []
    for i, params in enumerate(parameter_grid):
        train(params, i)
        # thread = threading.Thread(target=train, args=(params, i))
        # thread.start()
        # threads.append(thread)

    # for thread in threads:
    #     thread.join()

    print(evaluation_results)
