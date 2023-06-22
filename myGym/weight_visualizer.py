import os
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def extract_parameters_from_zip(root_dir):
    # Iterate through all subdirectories recursively
    index = 1
    weights = []
    targets = []
    for dirpath, _, filenames in os.walk(root_dir):
        # Check if the current directory contains string "multi"
        if 'multi' in dirpath:
            for filename in filenames:
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
                        if 'model/pi/w:0' in key:    
                            parameters_txt_path = os.path.join(weight_dir, str(index) + '_' + dirpath[-1] + '_' + 'parameters.txt')
                            with open(parameters_txt_path, 'w') as txt_file:
                                txt_file.write(str(value))
                        #print(value)    
                            print(f"Extracted parameters from {zip_path} and saved as {parameters_txt_path}")
                            print (value.shape)
                            index += 1  
                            # add value to weight list
                            weights.append(value.reshape(-1))
                            targets.append(int(dirpath[-1]))
                    # Delete the temporary directory
                    shutil.rmtree(temp_dir)

    data = np.array(weights)

    # Perform t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data)

    # Get the unique target labels
    unique_targets = np.unique(targets)

    # Plot t-SNE results
    plt.figure(figsize=(8, 6))
    for target in unique_targets:
        indices = np.where(targets == target)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=str(target))
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()
                
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

root_directory = './trained_models/swipe'
extract_parameters_from_zip(root_directory)