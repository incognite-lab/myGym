import argparse
import colorsys
import json
import os
import re
from math import ceil as ceiling

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Color mapping for algorithms
color_map = {
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
dict_acts = {"A": "approach", "W": "withdraw", "G": "grasp", "D": "drop", "M": "move", "R": "rotate",
             "T": "transform", "F": "follow"}

# Color mapping for actions
color_map_acts = {
    "approach": ["#b9e4ff"],  # muted blue
    "withdraw": ["#ffdab2"],  # muted orange
    "grasp": ["#d2ffcb"],  # muted green
    "drop": ["#ffcdcb"],  # muted red
    "move": ["#f0d8f8"],  # muted purple
    "rotate": ["#fcfdb3"],  # muted yellow
    "transform": ["#ffb2f2"],  # muted pink
    "follow": ["#e3e3e3"]  # muted gray
}


def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return r / 255.0, g / 255.0, b / 255.0


def rgb_to_hex(r, g, b):
    """Convert an RGB tuple to a hex color string."""
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def brighten_color(hex_color, brightness_increase=0.0, saturation_increase=0.15):
    """Brighten and increase the intensity of a color."""
    r, g, b = hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Increase brightness (Value) and saturation
    v = min(v + brightness_increase, 1.0)  # Ensure value is not more than 1
    s = min(s + saturation_increase, 1.0)  # Ensure saturation is not more than 1

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return rgb_to_hex(r, g, b)


def atoi(text):
    """Convert text to integer if it's a digit, otherwise return the text."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Split text into natural sort keys."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def cfg_string2dict(cfg_raw):
    """Convert a configuration string to a dictionary."""
    return {key: value for key, _, value, _ in re.findall(
        r"(\w+\s?)(=\s?)([^=]+)( (?=\w+\s?=)|$)", cfg_raw)}


def dict2cfg_string(dictionary, separator="\n", assigner="="):
    """Convert a dictionary into a configuration string."""
    return separator.join([f"{k}{assigner}{v}" for k, v in dictionary.items()])


def multi_dict_diff_by_line(dict_list):
    """Compare multiple dictionaries line-by-line."""
    different_items, same_items = {}, {}
    for record in zip(*[d.items() for d in dict_list]):
        vals = [v for k, v in record]
        keys = [k for k, v in record]
        if not all(k == keys[0] for k in keys):
            raise Exception("Oops, error! Not all keys were the same. Lines in the configs must be mixed up.")
        if all(v == vals[0] for v in vals):
            same_items[keys[0]] = vals[0]
        else:
            different_items[keys[0]] = vals
    return different_items, same_items


def multi_dict_diff_scary(dict_list):
    """Compare a list of dictionaries and return differences and similarities."""
    all_vals = [[d[k] for d in dict_list] for k in dict_list[0].keys()]
    all_same = [all(v == line[0] for v in line) for line in all_vals]
    diff_dict = {record[0][0]: (record[0][1] if all_same[i] else [v for k, v in record])
                 for i, record in enumerate(zip(*[d.items() for d in dict_list]))}
    return diff_dict, all_same


def multi_dict_diff_by_key(dict_list):
    """Compare dictionaries by keys and return differences and similarities."""
    diff_dict, all_same = multi_dict_diff_scary(dict_list)
    return {k: v for i, (k, v) in enumerate(diff_dict.items()) if not all_same[i]}, \
        {k: v for i, (k, v) in enumerate(diff_dict.items()) if all_same[i]}


def get_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    # '/home/student/mygym/myGym/weight_vis2/SB3_100mil/AGMDW_SB3'
    parser.add_argument("--pth", default='/home/student/mygym/myGym/weight_visualizer/AGMDW_stable')
    parser.add_argument("--robot", default=["kuka", "panda"], nargs='*')
    parser.add_argument("--algo", default=["multiacktr", "multippo2", "ppo2", "ppo", "acktr", "sac", "ddpg",
                                           "a2c", "acer", "trpo", "multippo"], nargs='*')
    return parser.parse_args()


def legend_without_duplicate_labels(ax):
    """Create a legend without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize="20", loc="center right")


def ax_set(ax, title, y_axis):
    """Set title and labels for the given axis."""
    ax.set_title(title, fontsize=23)
    ax.set_xlabel('Training steps', fontsize=18)
    ax.set_ylabel(y_axis, fontsize=18)
    legend_without_duplicate_labels(ax)


def ax_plot(ax, steps, data, color, label):
    """Plot data on the given axis."""
    ax.plot(steps, data, color=color, linestyle='solid', linewidth=3, marker='o', markerfacecolor=color,
            markersize=4, label=label)


def ax_fill(ax, steps, meanvalue, data, index, color):
    """Plot data on the given axis."""
    ax.fill_between(steps, meanvalue - np.std(np.take(data, index, 0), 0),
                    meanvalue + np.std(np.take(data, index, 0), 0), color=color, alpha=0.2)


def main():
    global color_map_acts
    args = get_arguments()
    root, dirs, files = next(os.walk(str(args.pth)))
    dirs.sort(key=natural_keys)
    plt.rcParams.update({'font.size': 12})
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black', 'grey', 'brown', 'gold', 'limegreen',
              'silver', 'aquamarine', 'olive', 'hotpink', 'salmon']
    # Initialize data storage
    configs = []
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

            # Get trained actions
            acts = []
            for key, val in dict_acts.items():
                if key in task:
                    acts.append(val)

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
    for i in range(len(steps)):
        for act in color_map_acts.keys():
            color_map_acts[act].append(brighten_color(color_map_acts[act][-1]))
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
                label_counter = 1
                bottom = [0] * len(steps)
                meanvalue_subgoals_steps = np.mean(np.take(mean_subgoals_steps, index[i], 0), 0)
                for l, _ in enumerate(mean_subgoals_steps[counter]):
                    if len(mean_subgoals_steps[counter][l]) != len(steps):
                        print(f"Data length mismatch: {len(mean_subgoals_steps[counter][l])} vs {len(steps)}")
                        continue
                    p = ax.bar(x=steps, height=meanvalue_subgoals_steps[l],
                               color=color_map_acts.get(acts[label_counter - 1], "black"),
                               label=f"{acts[label_counter - 1]}", bottom=bottom, width=-width, align='edge',
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
        label_counter = 1
        bottom = [0] * len(steps)

        for l, _ in enumerate(mean_subgoals_steps[counter]):
            if len(mean_subgoals_steps[counter][l]) != len(steps):
                print(f"Data length mismatch: {len(mean_subgoals_steps[counter][l])} vs {len(steps)}")
                continue
            p = ax.bar(x=steps, height=mean_subgoals_steps[counter][l],
                       color=color_map_acts.get(acts[label_counter - 1], "black"),
                       label=f"{acts[label_counter - 1]}", bottom=bottom, width=-width, align='edge',
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
    fig1.savefig(f"{root}-averaged.png")
    fig2.savefig(f"{root}-subgoals.png")
    plt.show()


if __name__ == "__main__":
    main()
