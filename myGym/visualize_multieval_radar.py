import os
import re
import threading
import subprocess
from sklearn.model_selection import ParameterGrid
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import seaborn as sns
import pandas as pd
#import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pkg_resources
import numpy as np
from matplotlib.lines import Line2D
https://python-graph-gallery.com/radar-chart/
https://plotly.com/python/radar-chart/

currentdir = pkg_resources.resource_filename("myGym", "trained_models")
algo_1_results_path = os.path.join(currentdir, "fake_multi_evaluation_results_ppo2.json")
algo_2_results_path = os.path.join(currentdir, "fake_multi_evaluation_results_sac.json")
fixed_parameter_value = 256
selected_metric = "success_rate"


def load_algo_results(results_path):
    with open(results_path) as f:
        results = json.load(f)
    list_results = [json.loads(key.replace('\'', '"'))
                    + [float(value[selected_metric])] for key, value
                    in results.items() if str(fixed_parameter_value) in key]
    #list_results = list(map(lambda x: [v for v in x if v != fixed_parameter_value], list_results))
    return list_results


def plot_algo_results(ax, list_results, label, c, nr):
    w_labels = [result[0] for result in list_results]
    w_set = sorted(set(w_labels), key=w_labels.index)
    w = np.asarray([w_set.index(label) for label in w_labels])
    #ax.set_zticks(range(3))
    #ax.set_zticklabels(w_set)

    x_labels = [result[1] for result in list_results]
    x_set = sorted(set(x_labels), key=x_labels.index)
    x = np.asarray([x_set.index(label) for label in x_labels])
    ax.set_xticks(np.arange(0,4,1))
    ax.set_xticklabels(x_set)

    y_labels = [result[2] for result in list_results]
    y_set = sorted(set(y_labels), key=y_labels.index)
    y = np.asarray([y_set.index(label) for label in y_labels])
    ax.set_yticks(np.arange(0,4,1))
    ax.set_yticklabels(y_set)

    z = np.asarray([result[3] for result in list_results])

    # setup a Normalization instance
    norm = colors.Normalize(0,np.asarray(w_labels).max())
    # define the colormap
    cmap = plt.get_cmap(c)
    # Use the norm and cmap to define the edge colours
    edgecols = cmap(norm(np.asarray(w_labels)))

    verts = np.asarray([[[0,0],[zz/100,0],[zz/100,zz/100],[0,zz/100],[0,0]] for zz in z])

    for i in range(len(z)):
        ax.scatter(x[i]+0.5*nr, y[i], alpha=1, s=100*500, c='None', lw=2, edgecolors=edgecols[i], marker=[[0,0],[1,0],[1,1],[0,1],[0,0]])
        ax.scatter(x[i]+0.5*nr, y[i], alpha=1, s=z[i]*500, c=edgecols[i], lw=2, edgecolors=edgecols[i], marker=verts[i])

    legend_elements = [Line2D([0], [0], linewidth=0, marker='s', color='r', label='PPO2', markersize=10),
                       Line2D([0], [0], linewidth=0, marker='s', color='b', label='SAC', markersize=10),
                       Line2D([0], [0], linewidth=0, marker='s', color='k', markerfacecolor='k', label='success high', markersize=4),
                       Line2D([0], [0], linewidth=0, marker='s', color='k', markerfacecolor='k', label='success higher', markersize=6),
                       Line2D([0], [0], linewidth=0, marker='s', color='k', markerfacecolor='k', label='success highest', markersize=8),
                       Line2D([0], [0], linewidth=0, marker='s', color='k', markerfacecolor='None', label='success 100%', markersize=10)]
                       #Line2D([0], [0], linewidth=0, marker='s', color=edgecols[len(z)//2], markerfacecolor='None', label='num steps '+str(w_set[1]), markersize=10),
                       #Line2D([0], [0], linewidth=0, marker='s', color=edgecols[-1], markerfacecolor='None', label='num steps '+str(w_set[2]), markersize=10)]
    ax.legend(handles=legend_elements)
    

    #plt.show()

    print("Mean succes rate for {}: {}".format(label, np.mean(z)))


if __name__ == '__main__':
    algo_1_list_results = load_algo_results(algo_1_results_path)
    algo_2_list_results = load_algo_results(algo_2_results_path)

    fig = plt.figure()
    ax = fig.add_subplot()    
    ax.set_xlabel("Robot control complexity", labelpad=5)
    ax.set_ylabel("Task complexity", labelpad=5)
    #ax.set_zlabel("Num steps", labelpad=15)

    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    # Hide the top and right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    plot_algo_results(ax, algo_1_list_results, "ppo2", "Reds", 0)
    plot_algo_results(ax, algo_2_list_results, "sac", "Blues", 1)
    plt.grid(axis='both', which='both')
    plt.title('Success evaluation - max episode steps {}'.format(fixed_parameter_value))
    #plt.savefig("multieval_visualization.png")

    plt.show()
