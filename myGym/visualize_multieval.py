import os
import re
import threading
import subprocess
from sklearn.model_selection import ParameterGrid
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pkg_resources
import numpy as np
from matplotlib.lines import Line2D
import operator

currentdir = pkg_resources.resource_filename("myGym", "trained_models")
algo_1_results_path = os.path.join(currentdir, "multi_evaluation_results_ppo2.json")
algo_2_results_path = os.path.join(currentdir, "multi_evaluation_results_ppo.json")
algo_3_results_path = os.path.join(currentdir, "multi_evaluation_results_sac.json")
algo_4_results_path = os.path.join(currentdir, "multi_evaluation_results_td3.json")
algo_5_results_path = os.path.join(currentdir, "multi_evaluation_results_trpo.json")
algo_6_results_path = os.path.join(currentdir, "multi_evaluation_results_ddpg.json")
algo_7_results_path = os.path.join(currentdir, "multi_evaluation_results_acktr.json")
algo_8_results_path = os.path.join(currentdir, "multi_evaluation_results_a2c.json")
selected_metric = "success_rate"


def load_algo_results(results_path):
    with open(results_path) as f:
        results = json.load(f)
    list_results = [json.loads(key.replace('\'', '"'))
                    + [float(value[selected_metric])] for key, value
                    in results.items()]# if str(fixed_parameter_value) in key]
    #list_results = list(map(lambda x: [v for v in x if v != fixed_parameter_value], list_results))
    df = pd.DataFrame(list_results)
    order2=['reach', 'push', 'pnp']
    order1=['step', 'joints', 'joints_gripper']
    order3=[1024, 512, 256]
    df[1] = pd.Categorical(df[1], order1)
    df[2] = pd.Categorical(df[2], order2)
    df[0] = pd.Categorical(df[0], order3)
    df = df.sort_values([2,1,0])
    list_results = df.values.tolist()
    return list_results

def plot_algo_results(ax, list_results, label, c):
    w_labels = [result[0] for result in list_results]
    w_set = sorted(set(w_labels), key=w_labels.index)
    w = np.asarray([w_set.index(label) for label in w_labels])

    x_labels = [result[1] for result in list_results]
    x_labels = [x.replace('joints_gripper', 'gripper') for x in x_labels]
    x_set = sorted(set(x_labels), key=x_labels.index)
    x = np.asarray([x_set.index(label) for label in x_labels])

    y_labels = [result[2] for result in list_results]
    y_set = sorted(set(y_labels), key=y_labels.index)
    y = np.asarray([y_set.index(label) for label in y_labels])

    z = [result[3] for result in list_results]
    z += z[:1]

    # Make data: I have 3 groups and 7 subgroups
    group_names = y_set
    n_groups = len(group_names)
    group_size = [100/n_groups]*n_groups
    
    subgroup_names = x_set*n_groups
    n_subgroups = len(subgroup_names)
    subgroup_size = [100/n_subgroups]*n_subgroups
    
    subsubgroup_names = w_set*n_subgroups
    n_subsubgroups = len(subsubgroup_names)
    subsubgroup_size = [100/n_subsubgroups]*n_subsubgroups
    
    # Create colors
    r, g, b=[plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]
    
    # First Ring (outside)
    ax.set(aspect="equal")
    mypie, txt = ax.pie(group_size, radius=1.6, labels=group_names, startangle=0, pctdistance=-5.125, colors=[r(0.4), r(0.6), r(0.8)], rotatelabels=True)
    plt.setp(mypie, width=0.12, edgecolor='white')

    # Second Ring (Inside)
    mypie2, txt2 = ax.pie(subgroup_size, radius=1.6-0.12, labels=subgroup_names, startangle=0, colors=[b(0.4), b(0.6), b(0.8)]*n_groups, rotatelabels=True)
    plt.setp(mypie2, width=0.12, edgecolor='white')

    # Third Ring    
    mypie3, txt3 = ax.pie(subsubgroup_size, radius=1.6-0.24, labels=subsubgroup_names, startangle=0, colors=[g(0.4), g(0.6), g(0.8)]*n_subgroups, rotatelabels=True)
    plt.setp(mypie3, width=0.12, edgecolor='white')
    plt.margins(0,0)

    for t in range(len(txt)):
        txt[t]._x = np.cos((0.5+t)*group_size[0]/100*np.pi*2)*(1.49)
        txt[t]._y = np.sin((0.5+t)*group_size[0]/100*np.pi*2)*(1.49)
        txt[t]._rotation -= 90*np.sign(txt[t]._x)
    for t in range(len(txt2)):
        txt2[t]._x = np.cos((0.5+t)*subgroup_size[0]/100*np.pi*2)*(1.35)
        txt2[t]._y = np.sin((0.5+t)*subgroup_size[0]/100*np.pi*2)*(1.35)
        txt2[t]._rotation -= 90*np.sign(txt2[t]._x)
    for t in range(len(txt3)):
        txt3[t]._x = np.cos((0.5+t)*subsubgroup_size[0]/100*np.pi*2)*(1.25)
        txt3[t]._y = np.sin((0.5+t)*subsubgroup_size[0]/100*np.pi*2)*(1.25)
        txt3[t]._rotation -= 90*np.sign(txt3[t]._x)

    #plt.show()

    categories=w_labels
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi + 0.5 / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    radar = fig.add_subplot(111, polar=True)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='None')
    
    # Draw ylabels
    yticks = np.arange(0,101,10)
    yticks_labels = [str(tick)+' %' for tick in yticks]
    radar.set_rlabel_position(90)
    plt.yticks(yticks, labels=yticks_labels, color="black", size=8)
    plt.ylim(0,110)
    
    # Plot data
    radar.plot(angles, z, c, linewidth=1, linestyle='solid')
    radar.fill(angles, z, c, alpha=0.3)

    #plt.show()
    learnability = np.mean(z)
    print("Learnability for {}: {}".format(label, learnability))
    return learnability

if __name__ == '__main__':
    algo_1_list_results = load_algo_results(algo_1_results_path)
    algo_2_list_results = load_algo_results(algo_2_results_path)
    algo_3_list_results = load_algo_results(algo_3_results_path)
    algo_4_list_results = load_algo_results(algo_4_results_path)
    algo_5_list_results = load_algo_results(algo_5_results_path)
    algo_6_list_results = load_algo_results(algo_6_results_path)
    algo_7_list_results = load_algo_results(algo_7_results_path)
    algo_8_list_results = load_algo_results(algo_8_results_path)

    fig = plt.figure()
    ax = fig.subplots()    

    learnability_ppo2 = plot_algo_results(ax, algo_1_list_results, "ppo2", "r")
    learnability_ppo = plot_algo_results(ax, algo_2_list_results, "ppo", "g")
    learnability_sac = plot_algo_results(ax, algo_3_list_results, "sac", "b")
    #learnability_td3 = plot_algo_results(ax, algo_4_list_results, "td3", "c")
    learnability_trpo = plot_algo_results(ax, algo_5_list_results, "trpo", "m")
    #learnability_ddpg = plot_algo_results(ax, algo_6_list_results, "ddpg", "y")
    learnability_acktr = plot_algo_results(ax, algo_7_list_results, "acktr", "c")
    #learnability_a2c = plot_algo_results(ax, algo_8_list_results, "a2c", "r")

    txt = plt.title('Learnability', fontsize=18)
    txt._x=1.07
    txt._y=1.08
    legend_elements = [Line2D([0], [1], linewidth=2, color='r', label='PPO2, {:.2f} %'.format(learnability_ppo2)),
                       Line2D([0], [1], linewidth=2, color='m', label='TRPO, {:.2f} %'.format(learnability_trpo)),
                       Line2D([0], [1], linewidth=2, color='c', label='ACKTR, {:.2f} %'.format(learnability_acktr)),
                       Line2D([0], [1], linewidth=2, color='b', label='SAC, {:.2f} %'.format(learnability_sac)),
                       Line2D([0], [1], linewidth=2, color='g', label='PPO, {:.2f} %'.format(learnability_ppo))]
                       #Line2D([0], [1], linewidth=2, color='c', label='TD3, {:.2f} %'.format(learnability_td3)),
                       #Line2D([0], [1], linewidth=2, color='y', label='DDPG, {:.2f} %'.format(learnability_ddpg)),
                       #Line2D([0], [1], linewidth=2, color='r', label='A2C, {:.2f} %'.format(learnability_a2c))]
    ax.legend(handles=legend_elements, loc=(0.97, 0.85), prop={'size': 16})
    plt.savefig(os.path.join(currentdir, "multieval_visualization.png"))

    plt.show()
