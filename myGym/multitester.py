import argparse
import json
import subprocess
import threading
import time

from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", type=str, default="./configs/debugdist.json", help="config file for evaluation")
parser.add_argument("-b", "--robot", default=["kuka"], nargs='*', help="what robots to test")
parser.add_argument("-ba", "--robot_action", default=["joints"], nargs='*', help="what actions to test")
parser.add_argument("-ar", "--action_repeat", default=[1], nargs='*', help="simuilation steps without env action")
parser.add_argument("-mv", "--max_velocity", default=[3], nargs='*', help="arm speed")
parser.add_argument("-mf", "--max_force", default=[100], nargs='*', help="arm force")
#task
parser.add_argument("-tt", "--task_type", default=["reach"], nargs='*',
                    help="Type of task to learn: reach, push, throw, pick_and_place")

parser.add_argument("-w", "--train_framework", default=["tensorflow"], nargs='*', help="what algos to test")
parser.add_argument("-a", "--algo", default=["acktr"], nargs='*', help="what algos to test")
parser.add_argument("-l", "--logdir", type=str, default="./trained_models/reach", help="where to save the results")
parser.add_argument("-thread", "--threaded", type=bool, default="True", help="run in threads")
parser.add_argument("-out", "--output", type=str, default="./trained_models/multitester.json", help="output file")

args = parser.parse_args()

parameters = {}
# find args with multiple values
for arg in vars(args):
    if type(getattr(args, arg)) == list:
        if len(getattr(args, arg)) > 1:
            #add arg as key to parameters dict and gettatr(args, arg) as value
            parameters[arg] = getattr(args, arg)


parameter_grid = ParameterGrid(parameters)
configfile = args.config
logdirfile = args.logdir
evaluation_results_paths = [None] * len(parameter_grid)
threaded = args.threaded
last_eval_results = {}


def train(params):
    print((" ".join(f"--{key} {value}" for key, value in params.items())).split())
    command = 'python train.py --config {configfile} --logdir {logdirfile} '.format(configfile=configfile,
                                                                                    logdirfile=logdirfile) + " ".join(
        f"--{key} {value}" for key, value in params.items())
    subprocess.check_output(command.split())

    with open(args.output, 'w') as f:
        json.dump(last_eval_results, f, indent=4)


if __name__ == '__main__':
    threads = []
    starttime = time.time()
    for i, params in enumerate(parameter_grid):
        if threaded:
            thread = threading.Thread(target=train, args=params)
            thread.start()
            threads.append(thread)
        else:
            train(params.copy())

    if threaded:
        for thread in threads:
            thread.join()
    endtime = time.time()
    print(endtime - starttime)
