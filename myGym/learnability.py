import os
import re
import threading
import subprocess
from sklearn.model_selection import ParameterGrid
import json
import time

parameters = {
    "robot_action": ["step", "joints","joints_gripper"],
    "task_type": ["reach", "push","pnp"],
    "max_episode_steps": [1024, 512, 256]
}
parameter_grid = ParameterGrid(parameters)
configfile = 'configs/multi.json'
evaluation_results_paths = [None] * len(parameter_grid)
threaded = True
last_eval_results = {}


def train(params, i):
    if "task_type" in params and params["task_type"] != "reach":
        params["task_objects"] = "cube_holes target"
        params["reward"] = "complex_distance"
    print((" ".join(f"--{key} {value}" for key, value in params.items())).split())
    command = 'python train.py --config {configfile} '.format(configfile=configfile) + " ".join(f"--{key} {value}" for key, value in params.items())
    output = subprocess.check_output(command.split())
    evaluation_results_paths[i] = output.splitlines()[-1].decode('UTF-8') + "/evaluation_results.json"
    with open(evaluation_results_paths[i]) as f:
        data = json.load(f)
        last_eval_results[str(list(params.values()))] = list(data.values())[-1]
    print(last_eval_results)
    with open("trained_models/multi_evaluation_results.json", 'w') as f:
        json.dump(last_eval_results, f, indent=4)
    
    # os.system('python train.py --config {configfile} '.format(configfile=configfile)
    #           + " ".join(f"--{key} {value}" for key, value in params.items()))


if __name__ == '__main__':
    threads = []
    starttime=time.time()
    for i, params in enumerate(parameter_grid):
        if threaded:
            thread = threading.Thread(target=train, args=(params, i))
            thread.start()
            threads.append(thread)
        else:
            train(params.copy(), i)
    
    if threaded:
        for thread in threads:
            thread.join()
    endtime=time.time()
    print (endtime-starttime)
