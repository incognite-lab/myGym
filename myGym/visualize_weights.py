import argparse
import json
import os
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

'''
    - Choose algorithm for which you want to visualize the weights
    Directory name should contain the algorithm name and the task name
    Example: ./weight_visualizer/AGMDW_stable/AGMDW_table_tiago_tiago_dual_joints_gripper_multiacktr
    In this case, the algorithm is 'multiacktr' and the task is 'AGMDW' - 'A'pproach, 'G'rasp, 'M'ove, 'D'rop, 'W'ithdraw
    - Choose the root directory path
    Example: ./weight_visualizer/AGMDW_stable
    Program combines all the weight layers and visualizes them in 2D and 3D using t-SNE and PCA
    - Change the list of weight_layers to visualize different layers accordingly to your needs
    - Change the perplexity value to get different results
'''

# Action dictionary
dict_acts = {
    "A": "approach", "G": "grasp", "M": "move", "D": "drop",
    "W": "withdraw", "R": "rotate", "T": "transform",
    "F": "find", "r": "reach", "L": "leave", "Fo": "follow"
}

# Color mapping for actions
color_map = {
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

weight_layers = [
    "model/pi_fc0/w:0",
    "model/vf_fc0/w:0",
    "model/pi_fc1/w:0",
    "model/vf_fc1/w:0",
    "model/vf/w:0",
    "model/pi/w:0",
    "model/q/w:0"
]


def cut_before_last_slash(logdir: str) -> str:
    """Cut all strings prior to and including the last '/' in the given string."""
    parts = logdir.rsplit('/', 1)
    return parts[1] if len(parts) > 1 else logdir


def find_indices(strings, target):
    indices = [index for index, string in enumerate(strings) if string == target]
    return indices


def find_matching_index(string_list, search_string):
    for index, string in enumerate(string_list):
        if string in search_string:
            return index
    return -1  # Return -1 if no match is found


def visualize_weights(root_dir, weight_layer, layer_targets, alg, perplexity):
    # Iterate through all subdirectories recursively
    weights = []
    algotargets = []
    acts = []
    nets = 0
    algos = ['acktr', 'ppo2', 'trpo', 'a2c', 'ppo', 'dqn', 'ddpg', 'sac']
    for dirpath, _, filenames in os.walk(root_dir):
        if alg in dirpath:
            for filename in filenames:
                if "train.json" in filename:
                    # Open and read train.json
                    with open(os.path.join(dirpath, filename)) as f:
                        train_data = json.load(f)
                        if 'task_type' in train_data:
                            task_name = train_data['task_type']
                    nets += 1
                if filename.endswith('.zip'):
                    zip_path = os.path.join(dirpath, filename)
                    # Create a temporary directory to extract the zip contents
                    weight_dir = os.path.join(root_dir, 'weights')
                    temp_dir = os.path.join(weight_dir, 'temp')
                    os.makedirs(temp_dir, exist_ok=True)

                    # Extract the zip file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    # Find the 'parameters' file and save it as a txt file
                    parameters_path = os.path.join(temp_dir, 'parameters')
                    if os.path.exists(parameters_path):
                        parameters = np.load(parameters_path, allow_pickle=True)
                        for key, value in parameters.items():
                            # Save the 'parameters' as a txt file
                            if weight_layer in key:
                                # add value to a weight list
                                weights.append(value.reshape(-1))
                                algoindex = find_matching_index(algos, dirpath)
                                if 'submodel' in dirpath or any(key in dirpath for key in color_map.keys()):
                                    acts.append(os.path.split(dirpath)[-1])
                                if 'multi' in dirpath:
                                    algotargets.append(str('m' + algos[algoindex] + dirpath[-1]))
                                else:
                                    algotargets.append(str(algos[algoindex]))
                                    target_name = ''.join(e for e in dirpath if e.isalnum())
                                    target_name = target_name[17:22]
                    # Delete the temporary directory
                    shutil.rmtree(temp_dir)

    data = np.array(weights, ndmin=2)
    print(data.shape)
    print(len(algotargets))

    # Perform t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=5000, random_state=42)
    tsne_results = tsne.fit_transform(data)

    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=5000, random_state=42)
    tsne_results_3d = tsne.fit_transform(data)

    # Get the unique target labels
    unique_algotargets = np.unique(algotargets)
    unique_acts = np.unique(acts)

    unique_acts = dict.fromkeys(unique_acts)
    acts_temp = []

    for key, val in dict_acts.items():
        if key in task_name:
            acts_temp.append(val)

    if len(unique_acts) == len(acts_temp):
        i = 0
        for key in unique_acts:
            unique_acts[key] = acts_temp[i]
            i += 1
            acts = [unique_acts[key] if x == key else x for x in acts]

    unique_acts = np.unique(acts)

    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(weights)

    pca = PCA(n_components=3, random_state=42)
    pca_results_3d = pca.fit_transform(weights)

    title = "Task: {}, Alg: {}, Nets: {}, Samples: {}, Dimensions: {}, Perplexity: {}".format(task_name, alg, nets,
                                                                                              data.shape[0],
                                                                                              data.shape[1], perplexity)

    if len(unique_acts) > 0:
        layer_targets[weight_layer] = (acts, tsne_results, tsne_results_3d,
                                       unique_acts, pca_results, pca_results_3d)
    else:
        layer_targets[weight_layer] = (algotargets, tsne_results, tsne_results_3d,
                                       unique_algotargets, pca_results, pca_results_3d)

    return layer_targets, title


def combine_plots(layer_targets, title, root_dir):
    fig = plt.figure(figsize=(20, 20))

    # First subplot
    title1 = title.split(',', 1)
    title1 = title1[0] + " " + " TSNE 2D," + title1[1]
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title(title1)
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')

    # Second subplot
    title2 = title.split(',', 1)
    title2 = title2[0] + " " + " TSNE 3D," + title2[1]
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title(title2)
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.set_zlabel('Dimension 3')

    # Third subplot
    title3 = title.split(',', 1)
    title3 = title3[0] + " " + " PCA 2D," + title3[1]
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title(title3)
    ax3.set_xlabel('Dimension 1')
    ax3.set_ylabel('Dimension 2')

    # Forth subplot
    title4 = title.split(',', 1)
    title4 = title4[0] + " " + " PCA 3D," + title4[1]
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_title(title4)
    ax4.set_xlabel('Dimension 1')
    ax4.set_ylabel('Dimension 2')
    ax4.set_zlabel('Dimension 3')

    first_tsne_2d = True
    first_tsne_3d = True
    first_pca_2d = True
    first_pca_3d = True

    for layer, params in layer_targets.items():
        acts, tsne_results, tsne_results_3d, unique_acts, pca_results, pca_results_3d = params
        for act in unique_acts:
            indices = find_indices(acts, act)
            color = color_map.get(act, "black")
            s = [3 * n for n in range(len(tsne_results[indices, 0]))]
            # opacity = [0.2 + (0.8 / (len(tsne_results[indices, 0]) - 1)) * n for n in
            #            range(len(tsne_results[indices, 0]))]
            if first_tsne_2d:
                ax1.scatter(tsne_results[indices, 0], tsne_results[indices, 1], s=s,
                            label=str(act), color=color)
            else:
                ax1.scatter(tsne_results[indices, 0], tsne_results[indices, 1], s=s, color=color)
        first_tsne_2d = False

    for layer, params in layer_targets.items():
        acts, tsne_results, tsne_results_3d, unique_acts, pca_results, pca_results_3d = params
        for act in unique_acts:
            indices = find_indices(acts, act)
            color = color_map.get(act, "black")
            s = [3 * n for n in range(len(tsne_results_3d[indices, 0]))]
            if first_tsne_3d:
                ax2.scatter(tsne_results_3d[indices, 0], tsne_results_3d[indices, 1], tsne_results_3d[indices, 2], s=s,
                            label=str(act), color=color)
            else:
                ax2.scatter(tsne_results_3d[indices, 0], tsne_results_3d[indices, 1], tsne_results_3d[indices, 2], s=s,
                            color=color)
        first_tsne_3d = False

    for layer, params in layer_targets.items():
        acts, tsne_results, tsne_results_3d, unique_acts, pca_results, pca_results_3d = params
        for act in unique_acts:
            indices = find_indices(acts, act)
            color = color_map.get(act, "black")
            s = [3 * n for n in range(len(pca_results[indices, 0]))]
            if first_pca_2d:
                ax3.scatter(pca_results[indices, 0], pca_results[indices, 1], s=s,
                            label=str(act), color=color)
            else:
                ax3.scatter(pca_results[indices, 0], pca_results[indices, 1], s=s, color=color)
        first_pca_2d = False

    for layer, params in layer_targets.items():
        acts, tsne_results, tsne_results_3d, unique_acts, pca_results, pca_results_3d = params
        for act in unique_acts:
            indices = find_indices(acts, act)
            color = color_map.get(act, "black")
            s = [3 * n for n in range(len(pca_results_3d[indices, 0]))]
            if first_pca_3d:
                ax4.scatter(pca_results_3d[indices, 0], pca_results_3d[indices, 1], pca_results_3d[indices, 2], s=s,
                            label=str(act), color=color)
            else:
                ax4.scatter(pca_results_3d[indices, 0], pca_results_3d[indices, 1], pca_results_3d[indices, 2], s=s,
                            color=color)
        first_pca_3d = False

    ax1.legend(fontsize="20")
    ax2.legend(fontsize="20")
    ax3.legend(fontsize="20")
    ax4.legend(fontsize="20")

    # Uncomment this to change the view angle
    # ax1.view_init(90, 90)
    # ax2.view_init(-140, 60)
    # ax3.view_init(90, 90)
    # ax4.view_init(-140, 60)

    savedir = cut_before_last_slash(root_dir)

    path = './trained_models/' + savedir + '_weights.png'

    plt.savefig(path, bbox_inches=None)
    plt.show()


# Usage example:
# create main with argparser to select root directory path and weight layer
# python weight_visualizer.py --root_directory ./trained_models/swipe --weight_layer model/pi/w:0 --perplexity 30 --alg multiacktr


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Load weights for a specific layer in a root directory')

    # Define the command-line arguments
    parser.add_argument('--rootdir', type=str, default='./weight_visualizer/AGMDW_stable', help='Root directory path')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexity value for t-SNE')
    parser.add_argument('--alg', type=str, default='multiacktr', help='Algorithm name')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the visualize_weights function with the provided arguments
    layer_targets = {}
    for weight_layer in weight_layers:
        print("Visualizing weights for layer: {}".format(weight_layer))
        layer_targets, title = visualize_weights(args.rootdir, weight_layer, layer_targets, args.alg, args.perplexity)
    combine_plots(layer_targets, title, args.rootdir)
