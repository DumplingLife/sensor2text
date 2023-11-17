import numpy as np
import os

# Function to calculate squared L2 norm between two numpy arrays
def squared_l2_norm(arr1, arr2):
    return np.sum((arr1 - arr2)**2)

# Path to the directories containing the npy files
dir_preprocessed = 'actionsense_data/S00_emg_chunks_preprocessed'
dir_original = 'actionsense_data/S00_emg_chunks'

# List the first 10 npy files from each directory
files_preprocessed = sorted(os.listdir(dir_preprocessed))[:10]
files_original = sorted(os.listdir(dir_original))[:10]

# Initialize an empty 10x10 matrix
distance_matrix = np.zeros((10, 10))

# Calculate pairwise distances
for i, file_preprocessed in enumerate(files_preprocessed):
    for j, file_original in enumerate(files_original):
        data_preprocessed = np.load(os.path.join(dir_preprocessed, file_preprocessed))
        data_original = np.load(os.path.join(dir_original, file_original))
        distance_matrix[i, j] = squared_l2_norm(data_preprocessed, data_original)

# Print or save the distance matrix
print(distance_matrix)
