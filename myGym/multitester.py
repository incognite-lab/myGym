import os
import re
import threading
import subprocess
from sklearn.model_selection import ParameterGrid
import json
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", type=str, default="./configs/pnp_2n_mag.json", help="config file for evaluation")
parser.add_argument("-b", "--robot",  default=["kuka_gripper","panda1"], nargs='*', help="what robots to test")
parser.add_argument("-ba", "--robot_action",  default=["absolute","step"], nargs='*', help="what actions to test")
parser.add_argument("-ar", "--action_repeat", default=[1,5,10], nargs='*', help="simuilation steps without env action")
parser.add_argument("-mv", "--max_velocity", default=[1,10,20], nargs='*', help="arm speed")
parser.add_argument("-mf", "--max_force", default=[1,50,100,300], nargs='*', help="arm force")
parser.add_argument("-frame", "--framework", default=["tensorflow"], nargs='*', help="what algos to test")
#parser.add_argument("-algo", "--algorithms", default=["ppo", "ppo2", "sac", "acktr", "ddpg", "td3", "trpo", "a2c", "myalgo","multi"], nargs='*', help="what algos to test")
parser.add_argument("-algo", "--algorithms", default=["ppo2","acktr"], nargs='*', help="what algos to test")
parser.add_argument("-thread", "--threaded", type=bool, default="True", help="run in threads")
parser.add_argument("-out", "--output", type=str, default="./trained_models/testermag.json", help="output file")

args = parser.parse_args()

parameters = {
    "robot": args.robot,
    "robot_action": args.robot_action,
    "algo": args.algorithms,
    "max_velocity": args.max_velocity
}
parameter_grid = ParameterGrid(parameters)
configfile = args.config
evaluation_results_paths = [None] * len(parameter_grid)
threaded = args.threaded
last_eval_results = {}


def train(params, i):
    print((" ".join(f"--{key} {value}" for key, value in params.items())).split())
    command = 'python train.py --config {configfile} '.format(configfile=configfile) + " ".join(f"--{key} {value}" for key, value in params.items())
    output = subprocess.check_output(command.split())
    evaluation_results_paths[i] = output.splitlines()[-1].decode('UTF-8') + "/evaluation_results.json"
    with open(evaluation_results_paths[i]) as f:
        data = json.load(f)
        last_eval_results[str(list(params.values()))] = list(data.values())[-1]
    print(last_eval_results)
    with open(args.output, 'w') as f:
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
