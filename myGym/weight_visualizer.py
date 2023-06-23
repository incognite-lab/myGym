import os
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse


def find_indices(strings, target):
    indices = [index for index, string in enumerate(strings) if string == target]
    return indices

def find_matching_index(string_list, search_string):
    for index, string in enumerate(string_list):
        if string in search_string:
            return index
    return -1  # Return -1 if no match is found


def visualize_weights(root_dir, weight_layer):
    # Iterate through all subdirectories recursively
    index = 1
    weights = []
    targets = []
    algotargets = []
    nets=0
    algos = ['acktr', 'ppo2', 'trpo', 'a2c','ppo','dqn','ddpg','sac']
    for dirpath, _, filenames in os.walk(root_dir):
        
        for filename in filenames:
            if "train.json" in filename:
                nets+=1
            if filename.endswith('.zip'):
                zip_path = os.path.join(dirpath, filename)
                
                # Create a temporary directory to extract the zip contents
                weight_dir = os.path.join(root_dir,'weights')
                temp_dir = os.path.join(weight_dir,'temp')
                os.makedirs(temp_dir, exist_ok=True)
                
            
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    
                # Find the 'parameters' file and save it as a txt file
                parameters_path = os.path.join(temp_dir, 'parameters')
                parameters = np.load(parameters_path, allow_pickle=True)
                for key, value in parameters.items():
                    #print(f"Key: {key}")
                    #print(f"Value: {value}")
                    #print()
                    
                    # Save the 'parameters' as a txt file
                    if weight_layer in key:    
                        #parameters_txt_path = os.path.join(weight_dir, str(index) + '_' + dirpath[-1] + '_' + 'parameters.txt')
                        #with open(parameters_txt_path, 'w') as txt_file:
                        #    txt_file.write(str(value))
                    #print(value)    
                        #print(f"Extracted parameters from {zip_path} and saved as {parameters_txt_path}")
                        #print (value.shape)
                        index += 1  
                        # add value to weight list
                        weights.append(value.reshape(-1))
                        algoindex = find_matching_index(algos,dirpath)
                        if 'multi' in dirpath:
                            algotargets.append(str('m' + algos[algoindex] + dirpath[-1]))
                            targets.append(str('multi' + dirpath[-1]))
                        else:
                            algotargets.append(str(algos[algoindex]))
                            targets.append(str('single'))

                # Delete the temporary directory
                shutil.rmtree(temp_dir)

    data = np.array(weights)
    print (data.shape)
    print (len(targets))

    # Perform t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data)

    # Get the unique target labels
    unique_targets = np.unique(targets)
    unique_algotargets = np.unique(algotargets)
    # Plot t-SNE results
    
    #Remove all non-alphabet characters from weight_layer
    layername = ''.join(e for e in weight_layer if e.isalnum())
    taskname = ''.join(e for e in root_dir if e.isalnum())
    taskname = taskname[-5:]
    layername = layername[5:]
    title = "Task: {}, Layer: {},Nets:{}, Samples: {},Features:{}".format(taskname, layername,nets,data.shape[0],data.shape[1])
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(20, 9))

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1)
    
    for target in unique_targets:
        indices = find_indices(targets, target)
        #indices = np.where(targets == target)
        ax1.scatter(tsne_results[indices, 0], tsne_results[indices, 1],s=3, label=str(target))
    ax1.set_title(title)
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.legend()

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2)
    
    for target in unique_algotargets:
        indices = find_indices(algotargets, target)
        #indices = np.where(targets == target)
        ax2.scatter(tsne_results[indices, 0], tsne_results[indices, 1],s=8, label=str(target))
    ax2.set_title(title)
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.legend()
    
    plt.tight_layout() 
    #plt.show()
    plt.savefig(os.path.join(root_dir,taskname + layername + '.png'))

    # Perform PCA
    #pca = PCA(n_components=2, random_state=42)
    #pca_results = pca.fit_transform(weights)


    # Plot PCA results
    #plt.figure(figsize=(8, 6))
    #for target in unique_targets:
    #    indices = np.where(targets == target)
    #    plt.scatter(pca_results[indices, 0], pca_results[indices, 1], label=str(target))
    #plt.title('PCA Visualization')
    #plt.xlabel('Principal Component 1')
    #plt.ylabel('Principal Component 2')
    #plt.legend()
    #plt.show()          

# Usage example

#create main with argparser to select root directory path and weigth layer
#  python weight_visualizer.py --root_directory ./trained_models/swipe --weight_layer model/pi/w:0

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Load weights for a specific layer in a root directory')

    # Define the command-line arguments
    parser.add_argument('--rootdir', type=str, default = './trained_models/swipe', help='Root directory path')
    parser.add_argument('--weight_layer', type=str, default = 'model/pi/w:0', help='Weight layer name')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the load_weights function with the provided arguments
    weigth_layers = ["model/pi_fc0/w:0",
    "model/pi_fc0/b:0",
    "model/vf_fc0/w:0",
    "model/vf_fc0/b:0",
    "model/pi_fc1/w:0",
    "model/pi_fc1/b:0",
    "model/vf_fc1/w:0",
    "model/vf_fc1/b:0",
    "model/vf/w:0",
    "model/vf/b:0",
    "model/pi/w:0",
    "model/pi/b:0",
    "model/pi/logstd:0",
    "model/q/w:0",
    "model/q/b:0"]

    for weight_layer in weigth_layers:
        visualize_weights(args.rootdir,weight_layer)
