import argparse
import json
import os
import re
from math import ceil as ceiling
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import shutil

# Color mapping for algorithms
color_map: Dict[str, str] = {
    "ppo": "red",
    "ppo2": "green",
    "multippo": "aquamarine",
    "acktr": "yellow",
    "multiacktr": "magenta",
    "sac": "salmon",
    "ddpg": "blue",
    "a2c": "grey",
    "acer": "brown",
    "trpo": "gold",
    "multippo2": "limegreen"
}

# Action dictionary
dict_acts: Dict[str, str] = {
    "A": "approach", "G": "grasp", "M": "move", "D": "drop",
    "W": "withdraw", "R": "rotate", "T": "transform",
    "F": "find", "r": "reach", "L": "leave"
}

# Color mapping for actions
color_map_acts: Dict[str, str] = {
    "approach": "#339dff",  # muted blue
    "withdraw": "#faa22e",  # muted orange
    "grasp": "#10e600",  # muted green
    "drop": "#f1160e",  # muted red
    "move": "#a838ff",  # muted purple
    "rotate": "#745339",  # muted brown
    "transform": "#f787d3",  # muted pink
    "follow": "#99acb8",  # muted gray
    "reach": "#fff35c",  # muted yellow
    "find": "#339dff",  # muted blue
    "leave": "#faa22e"  # muted orange
}


def cut_before_last_slash(logdir: str) -> str:
    """Cut all strings prior to and including the last '/' in the given string."""
    parts = logdir.rsplit('/', 1)
    return parts[1] if len(parts) > 1 else logdir

def atoi(text: str) -> Any:
    """Convert text to integer if it's a digit, otherwise return the text."""
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[Any]:
    """Split text into natural sort keys."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def cfg_string2dict(cfg_raw: str) -> Dict[str, str]:
    """Convert a configuration string to a dictionary."""
    return {key: value for key, _, value, _ in re.findall(
        r"(\w+\s?)(=\s?)([^=]+)( (?=\w+\s?=)|$)", cfg_raw)}


def dict2cfg_string(dictionary: Dict[str, str], separator: str = "\n", assigner: str = "=") -> str:
    """Convert a dictionary into a configuration string."""
    return separator.join([f"{k}{assigner}{v}" for k, v in dictionary.items()])


def multi_dict_diff_by_line(dict_list: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """Compare multiple dictionaries line-by-line."""
    different_items, same_items = {}, {}
    for record in zip(*[d.items() for d in dict_list]):
        vals = [v for _, v in record]
        keys = [k for k, _ in record]
        if not all(k == keys[0] for k in keys):
            raise ValueError("Inconsistent keys in the configurations.")
        if all(v == vals[0] for v in vals):
            same_items[keys[0]] = vals[0]
        else:
            different_items[keys[0]] = vals
    return different_items, same_items


def multi_dict_diff_scary(dict_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[bool]]:
    """Compare a list of dictionaries and return differences and similarities."""
    all_vals = [[d[k] for d in dict_list] for k in dict_list[0].keys()]
    all_same = [all(v == line[0] for v in line) for line in all_vals]
    diff_dict = {record[0][0]: (record[0][1] if all_same[i] else [v for _, v in record])
                 for i, record in enumerate(zip(*[d.items() for d in dict_list]))}
    return diff_dict, all_same


def multi_dict_diff_by_key(dict_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compare dictionaries by keys and return differences and similarities."""
    diff_dict, all_same = multi_dict_diff_scary(dict_list)
    return {k: v for i, (k, v) in enumerate(diff_dict.items()) if not all_same[i]}, \
        {k: v for i, (k, v) in enumerate(diff_dict.items()) if all_same[i]}


def get_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", default='/home/student/mygym/myGym/weight_visualizer/AGMDW_stable', type=str)
    parser.add_argument("--robot", default=["kuka", "panda"], nargs='*', type=str)
    parser.add_argument("--algo", default=["multiacktr", "multippo2", "ppo2", "ppo", "acktr", "sac", "ddpg",
                                           "a2c", "acer", "trpo", "multippo"], nargs='*', type=str)
    return parser.parse_args()


def legend_without_duplicate_labels(ax: plt.Axes) -> None:
    """Create a legend without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize="20", loc="center right")


def ax_set(ax: plt.Axes, title: str, y_axis: str) -> None:
    """Set title and labels for the given axis."""
    ax.set_title(title, fontsize=23)
    ax.set_xlabel('Training steps', fontsize=18)
    ax.set_ylabel(y_axis, fontsize=18)
    legend_without_duplicate_labels(ax)


def ax_plot(ax: plt.Axes, steps: np.ndarray, data: np.ndarray, color: str, label: str) -> None:
    """Plot data on the given axis."""
    ax.plot(steps, data, color=color, linestyle='solid', linewidth=3, marker='o', markerfacecolor=color,
            markersize=4, label=label)


def ax_fill(ax: plt.Axes, steps: np.ndarray, meanvalue: np.ndarray, data: List[np.ndarray],
            index: List[int], color: str) -> None:
    """Fill the area between the lines on the given axis."""
    ax.fill_between(steps, meanvalue - np.std(np.take(data, index, 0), 0),
                    meanvalue + np.std(np.take(data, index, 0), 0), color=color, alpha=0.2)


def main() -> None:
    global color_map_acts
    args = get_arguments()
    root, dirs, _ = next(os.walk(str(args.pth)))
    dirs.sort(key=natural_keys)

    plt.rcParams.update({'font.size': 12})
    colors: List[str] = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black', 'grey', 'brown', 'gold',
                         'limegreen',
                         'silver', 'aquamarine', 'olive', 'hotpink', 'salmon']

    # Initialize data storage
    configs: List[Dict[str, Any]] = []
    success, mean_steps, mean_reward = [], [], []
    std_reward, mean_distance_error = [], []
    mean_subgoals_finished, mean_subgoals_steps = [], []

    ticks_num = 0
    min_steps = 100
    steps = []
    x = []

    # Process data from directories
    for idx, file in enumerate(dirs):
        try:
            # Load evaluation results
            with open(os.path.join(args.pth, file, "evaluation_results.json")) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            x = df.to_numpy()

            # Process multi-step reward data
            try:
                x = np.delete(x, [11], axis=0)
                temp = []
                for item in x[11]:
                    str_list = [float(i) for i in item.strip('[]').split()]
                    temp.append([str_list])
                mean_subgoals_steps.append(temp)
                x = np.delete(x, [11], axis=0)
            except FileNotFoundError:
                print("No multistep reward data")

            x = x.astype(float)

            # Load individual configs
            with open(os.path.join(args.pth, file, "train.json"), "r") as f:
                cfg_raw = yaml.full_load(f)
                task = cfg_raw['task_type']
                alg = cfg_raw['algo']
                robot = cfg_raw['robot']
                logdir = cfg_raw['logdir']

            # Get trained actions
            acts = []
            for l in task:
                if l in dict_acts.keys():
                    acts.append(dict_acts[l])

            success.append(x[3])
            mean_distance_error.append(x[4])
            mean_steps.append(x[5])
            mean_reward.append(x[6])
            std_reward.append(x[7])
            mean_subgoals_finished.append(x[10])
            ticks_num = 100 / int(x[9][0])

            configs.append(cfg_raw)
            min_steps = min(min_steps, len(x[0]))
            steps = x[0][:min_steps]
            print(f"{len(x[3])} datapoints in folder: {file}")
        except FileNotFoundError:
            print(f"0 datapoints in folder: {file}")
            shutil.rmtree(os.path.join(args.pth,file))
            print (f"Removed {file}")


    for i in range(len(mean_subgoals_steps)):
        mean_subgoals_steps[i] = [item for sublist in mean_subgoals_steps[i] for item in sublist]
    for i in range(len(mean_subgoals_steps)):
        for j in range(len(mean_subgoals_steps[i])):
            mean_sum = sum(mean_subgoals_steps[i][j])
            for k in range(len(mean_subgoals_steps[i][j])):
                mean_subgoals_steps[i][j][k] = (mean_subgoals_steps[i][j][k] / mean_sum) * 100

    for i in range(len(mean_subgoals_steps)):
        mean_subgoals_steps[i] = [[mean_subgoals_steps[i][k][j] for k in range(len(mean_subgoals_steps[i]))]
                                  for j in range(len(mean_subgoals_steps[i][0]))]

    # Get differences between configs
    diff, same = multi_dict_diff_by_key(configs)
    plot_num = len(set(diff['algo'])) if 'algo' in diff.keys() else 1
    if plot_num == 2:
        plot_num = 3
    ticks = list(range(0, 101, int(ticks_num)))
    width = steps[1] - steps[0]

    # Plotting
    fig1 = plt.figure(1, figsize=(35, 30))
    ax1, ax2, ax3, ax4, ax5 = [fig1.add_subplot(3, 2, i) for i in range(1, 6)]

    fig2 = plt.figure(2, figsize=(13 * ceiling(plot_num / 2), 10 * ceiling(plot_num / 2)))

    counter = 0
    for d in range(len(success)):
        success[d] = np.delete(success[d], np.s_[min_steps:])
        mean_reward[d] = np.delete(mean_reward[d], np.s_[min_steps:])
        std_reward[d] = np.delete(std_reward[d], np.s_[min_steps:])
        mean_steps[d] = np.delete(mean_steps[d], np.s_[min_steps:])
        mean_distance_error[d] = np.delete(mean_distance_error[d], np.s_[min_steps:])
        mean_subgoals_finished[d] = np.delete(mean_subgoals_finished[d], np.s_[min_steps:])
        for i, elem in enumerate(mean_subgoals_steps[d]):
            mean_subgoals_steps[d][i] = np.delete(elem, np.s_[min_steps:])
    if 'algo' in diff.keys() and len(args.algo) > 1:
        index = [[] for _ in range(len(args.algo))]
        for i, algo in enumerate(args.algo):
            for j, diffalgo in enumerate(diff['algo']):
                if algo == diffalgo:
                    index[i].append(j)
            if index[i]:
                meanvalue = np.mean(np.take(success, index[i], 0), 0)
                ax_plot(ax1, steps, meanvalue, color_map[algo], algo)
                ax_fill(ax1, steps, meanvalue, success, index[i], color_map[algo])

                meanvalue_rew = np.mean(np.take(mean_reward, index[i], 0), 0)
                ax_plot(ax2, steps, meanvalue_rew, color_map[algo], algo)
                ax_fill(ax2, steps, meanvalue_rew, mean_reward, index[i], color_map[algo])

                meanvalue_std = np.mean(np.take(std_reward, index[i], 0), 0)
                ax2.plot(steps, meanvalue_std, alpha=0.8, color=color_map[algo],
                         linestyle='dotted', linewidth=3, marker='o', markerfacecolor=color_map[algo], markersize=4,
                         label=f"{algo} std")
                ax_fill(ax2, steps, meanvalue_std, std_reward, index[i], color_map[algo])

                meanvalue_steps = np.mean(np.take(mean_steps, index[i], 0), 0)
                ax_plot(ax3, steps, meanvalue_steps, color_map[algo], algo)
                ax_fill(ax3, steps, meanvalue_steps, mean_steps, index[i], color_map[algo])

                meanvalue_error = np.mean(np.take(mean_distance_error, index[i], 0), 0)
                ax_plot(ax4, steps, meanvalue_error, color_map[algo], algo)
                ax_fill(ax4, steps, meanvalue_error, mean_distance_error, index[i], color_map[algo])

                meanvalue_subgoals = np.mean(np.take(mean_subgoals_finished, index[i], 0), 0)
                ax5.set_ylim([0, 100])
                ax_plot(ax5, steps, meanvalue_subgoals, color_map[algo], algo)
                ax_fill(ax5, steps, meanvalue_subgoals, mean_subgoals_finished, index[i], color_map[algo])
                ax5.set_yticks(ticks)

                ax = fig2.add_subplot(ceiling(plot_num / 2), ceiling(plot_num / 2), counter + 1)
                ax.set_ylim(0, 100)
                label_counter = 0
                bottom = [0] * len(steps)
                meanvalue_subgoals_steps = np.mean(np.take(mean_subgoals_steps, index[i], 0), 0)
                for l, _ in enumerate(mean_subgoals_steps[counter]):
                    if len(mean_subgoals_steps[counter][l]) != len(steps):
                        print(f"Data length mismatch: {len(mean_subgoals_steps[counter][l])} vs {len(steps)}")
                        continue
                    p = ax.bar(x=steps, height=meanvalue_subgoals_steps[l],
                               color=color_map_acts.get(acts[label_counter], "black"),
                               label=f"{acts[label_counter]}", bottom=bottom, width=-width, align='edge',
                               edgecolor='black')
                    bottom = [sum(x) for x in zip(bottom, meanvalue_subgoals_steps[l])]

                    ax.bar_label(
                        p,
                        labels=[f"{v:.1f}" for v in meanvalue_subgoals_steps[l]],
                        label_type='center',
                        color='black',
                        fontsize=13,
                        fmt='%g'
                    )
                    label_counter += 1
                counter += 1

                ax_set(ax, f'Mean subgoals steps over training steps, {algo}', 'Mean subgoals steps, %')
    elif 'robot' in diff.keys() and len(args.robot) > 1:
        index = [[] for _ in range(len(args.robot))]
        for i, robot in enumerate(args.robot):
            for j, diffrobot in enumerate(diff['robot']):
                if robot == diffrobot:
                    index[i].append(j)
            if index[i]:
                plt.plot(x[0], np.mean(np.take(success, index[i], 0), 0), color=colors[i], linestyle='solid',
                         linewidth=3, marker='o', markerfacecolor=colors[i], markersize=6)
                plt.fill_between(x[0],
                                 np.mean(np.take(success, index[i], 0), 0) - np.std(np.take(success, index[i], 0), 0),
                                 np.mean(np.take(success, index[i], 0), 0) + np.std(np.take(success, index[i], 0), 0),
                                 color=colors[i], alpha=0.2)
                plt.title("Robots")
    else:
        print("No data to compare")
        ax_plot(ax1, steps, [item for row in success for item in row], color_map[alg], alg)
        ax_plot(ax2, steps, [item for row in mean_reward for item in row], color_map[alg], alg)
        ax2.plot(steps, [item for row in std_reward for item in row], alpha=0.8, color=color_map[alg],
                 linestyle='dotted', linewidth=3, marker='o', markerfacecolor=color_map[alg], markersize=4,
                 label=f"{alg} std")
        ax_plot(ax3, steps, [item for row in mean_steps for item in row], color_map[alg], alg)
        ax_plot(ax4, steps, [item for row in mean_distance_error for item in row], color_map[alg], alg)
        ax5.set_ylim([0, 100])
        ax_plot(ax5, steps, [item for row in mean_subgoals_finished for item in row], color_map[alg], alg)
        ax5.set_yticks(ticks)

        ax = fig2.add_subplot(ceiling(plot_num / 2), ceiling(plot_num / 2), counter + 1)
        ax.set_ylim(0, 100)
        label_counter = 0
        bottom = [0] * len(steps)

        for l, _ in enumerate(mean_subgoals_steps[counter]):
            if len(mean_subgoals_steps[counter][l]) != len(steps):
                print(f"Data length mismatch: {len(mean_subgoals_steps[counter][l])} vs {len(steps)}")
                continue
            p = ax.bar(x=steps, height=mean_subgoals_steps[counter][l],
                       color=color_map_acts.get(acts[label_counter], "black"),
                       label=f"{acts[label_counter]}", bottom=bottom, width=-width, align='edge',
                       edgecolor='black')
            bottom = [sum(x) for x in zip(bottom, mean_subgoals_steps[counter][l])]

            ax.bar_label(
                p,
                labels=[f"{v:.1f}" for v in mean_subgoals_steps[counter][l]],
                label_type='center',
                color='black',
                fontsize=13,
                fmt='%g'
            )
            label_counter += 1
        counter += 1

        ax_set(ax, f'Mean subgoals steps over training steps, {alg}', 'Mean subgoals steps, %')

    # Set titles for axes
    ax_set(ax1, 'Success rate over training steps', 'Successful episodes {}(%)'.format(task))
    ax_set(ax2, 'Mean/std rewards over training steps', 'Mean/std rewards')
    ax_set(ax3, 'Mean steps over training steps', 'Mean steps')
    ax_set(ax4, 'Mean distance error over training steps', 'Mean distance error')
    ax_set(ax5, 'Mean subgoals finished over training steps', 'Mean subgoals finished')

    # Save figures
    logdir = cut_before_last_slash(logdir)
    fig1.savefig(f"./trained_models/{logdir}_train.png")
    fig2.savefig(f"./trained_models/{logdir}_goals.png")
    plt.show()


if __name__ == "__main__":
    main()
