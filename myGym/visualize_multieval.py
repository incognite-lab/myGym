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


results_path = "trained_models/fake_multi_evaluation_results.json"
fixed_parameter_value = 256
selected_metric = "success_rate"

if __name__ == '__main__':
    with open(results_path) as f:
        results = json.load(f)
    #print(results)
    #print(json.loads(list(results.keys())[0].replace('\'', '"')))
    list_results = [json.loads(key.replace('\'', '"'))
                    + [float(value[selected_metric])] for key, value
                    in results.items() if str(fixed_parameter_value) in key]
    list_results = list(map(lambda x: [v for v in x if v != fixed_parameter_value], list_results))
    print(list_results)

    #df = pd.read_csv('2016.csv')
    #sns.set(style = "darkgrid")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    x_labels = [result[0] for result in list_results]
    x_set  = list(set(x_labels))
    x = [x_set.index(label) for label in x_labels]
    ax.set_xticks(range(3))
    ax.set_xticklabels(x_set)
    y_labels = [result[1] for result in list_results]
    y_set  = list(set(y_labels))
    y = [y_set.index(label) for label in y_labels]
    ax.set_yticks(range(3))
    ax.set_yticklabels(y_set)
    z = [result[2] for result in list_results]
    #
    # ax.set_xlabel("Happiness")
    # ax.set_ylabel("Economy")
    # ax.set_zlabel("Health")
    ax.scatter(x, y, z)
    plt.savefig("multieval_visualization.png")
    plt.show()
