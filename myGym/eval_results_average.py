import functools
import json
import operator
import os
import re
import shutil
from typing import List, Any

import numpy as np
import pandas as pd


def atoi(text: str) -> Any:
    """Convert text to integer if it's a digit, otherwise return the text."""
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[Any]:
    """Split text into natural sort keys."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def functools_reduce_iconcat(data: List[List[float]]) -> List[float]:
    """Flatten a nested list using functools.reduce and operator.iconcat."""
    return functools.reduce(operator.iconcat, data, [])


def average_results(logdir: str) -> None:
    """
    Compute and save the average results from evaluation logs.

    Args:
        logdir (str): The directory containing evaluation result subfolders.
    """
    root, dirs, _ = next(os.walk(str(logdir)))
    dirs.sort(key=natural_keys)
    res_file = "evaluation_results.json"

    episodes, n_eval_episodes, success_episodes = [], [], []
    success, mean_steps, mean_reward = [], [], []
    std_reward, mean_distance_error = [], []
    mean_subgoals_finished, mean_subgoal_reward, mean_subgoals_steps = [], [], []

    for idx, file in enumerate(dirs):
        try:
            # Load evaluation results
            with open(os.path.join(logdir, file, "evaluation_results.json")) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            x = df.to_numpy()

            # Process multi-step reward data
            try:
                temp_rew = []
                for item in x[11]:
                    str_list = [float(i) for i in item.strip('[]').split()]
                    temp_rew.append([str_list])
                mean_subgoal_reward.append(temp_rew)
                x = np.delete(x, [11], axis=0)
                temp = []
                for item in x[11]:
                    str_list = [float(i) for i in item.strip('[]').split()]
                    temp.append([str_list])
                mean_subgoals_steps.append(temp)
                x = np.delete(x, [11], axis=0)
            except FileNotFoundError:
                print(f"No multi-step reward data in {file}")

            x = x.astype(float)

            episodes = x[0].astype(int)
            n_eval_episodes = x[1].astype(int)
            success_episodes.append(x[2])
            success.append(x[3])
            mean_distance_error.append(x[4])
            mean_steps.append(x[5])
            mean_reward.append(x[6])
            std_reward.append(x[7])
            num_tasks = x[8].astype(int)
            num_networks = x[9].astype(int)
            mean_subgoals_finished.append(x[10])

        except FileNotFoundError:
            print(f"Error reading data in file: {file}")
            shutil.rmtree(os.path.join(logdir, file))
            print(f"Removed {file}")

    mean_success_episodes = np.mean(success_episodes, 0)
    mean_success = np.mean(success, 0)
    mean_mean_distance_error = np.mean(mean_distance_error, 0)
    mean_mean_steps = np.mean(mean_steps, 0)
    mean_mean_reward = np.mean(mean_reward, 0)
    mean_std_reward = np.std(std_reward, 0)
    mean_mean_subgoals_finished = np.mean(mean_subgoals_finished, 0)
    mean_mean_subgoal_reward = np.mean(mean_subgoal_reward, 0)
    mean_mean_subgoals_steps = np.mean(mean_subgoals_steps, 0)

    results = {}
    for i, episode in enumerate(episodes):
        results[f"evaluation_after_{episode}_steps"] = {
            "episode": f"{episode}",
            "n_eval_episodes": f"{n_eval_episodes[i]}",
            "success_episodes_num": f"{mean_success_episodes[i]}",
            "success_rate": f"{mean_success[i]}",
            "mean_distance_error": f"{mean_mean_distance_error[i]:.2f}",
            "mean_steps_num": f"{mean_mean_steps[i]}",
            "mean_reward": f"{mean_mean_reward[i]:.2f}",
            "std_reward": f"{mean_std_reward[i]:.2f}",
            "number of tasks": f"{num_tasks[i]}",
            "number of networks": f"{num_networks[i]}",
            "mean subgoals finished": f"{mean_mean_subgoals_finished[i]}",
            "mean subgoal reward": f"{functools_reduce_iconcat(mean_mean_subgoal_reward[i])}",
            "mean subgoal steps": f"{functools_reduce_iconcat(mean_mean_subgoals_steps[i])}",
        }

    output_path = os.path.join(logdir, res_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    logdir = "/home/student/mygym/myGym/trained_models/AGM_table_tiago_tiago_dual_joints_gripper_ppo_"
    average_results(logdir)
