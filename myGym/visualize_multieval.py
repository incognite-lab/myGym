import os
import re
import threading
import subprocess
from sklearn.model_selection import ParameterGrid
import json
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


algo_1_results_path = "trained_models/fake_multi_evaluation_results_ppo2.json"
algo_2_results_path = "trained_models/fake_multi_evaluation_results_sac.json"
fixed_parameter_value = 256
selected_metric = "success_rate"


def load_algo_results(results_path):
    with open(results_path) as f:
        results = json.load(f)
    list_results = [json.loads(key.replace('\'', '"'))
                    + [float(value[selected_metric])] for key, value
                    in results.items() if str(fixed_parameter_value) in key]
    list_results = list(map(lambda x: [v for v in x if v != fixed_parameter_value], list_results))
    return list_results


def plot_algo_results(ax, list_results, label, c):
    x_labels = [result[0] for result in list_results]
    x_set = sorted(set(x_labels), key=x_labels.index)
    x = [x_set.index(label) for label in x_labels]
    ax.set_xticks(range(3))
    ax.set_xticklabels(x_set)

    y_labels = [result[1] for result in list_results]
    y_set = sorted(set(y_labels), key=y_labels.index)
    y = [y_set.index(label) for label in y_labels]
    ax.set_yticks(range(3))
    ax.set_yticklabels(y_set)

    z = [result[2] for result in list_results]

    ax.plot(y, x, z, 'o', label=label, markersize=8, color=c)

    # x,y = np.meshgrid(x, y)
    # z = np.tile(z, (len(x), 1))
    #print(x.shape, y.shape, z.shape)
    ax.plot_trisurf(y, x, z, lw=2, edgecolor=c, color="grey",
                alpha=0, )
    ax.legend()
    print("Mean succes rate for {}: {}".format(label, np.mean(z)))


if __name__ == '__main__':
    algo_1_list_results = load_algo_results(algo_1_results_path)
    algo_2_list_results = load_algo_results(algo_2_results_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylabel("Robot action", labelpad=15)
    ax.set_xlabel("Task type", labelpad=15)
    ax.set_zlabel("Success rate", labelpad=15)

    plot_algo_results(ax, algo_1_list_results, "ppo2", "C0")
    plot_algo_results(ax, algo_2_list_results, "sac", "C1")
    plt.savefig("multieval_visualization.png")
    plt.show()
