import h5py
import numpy as np
import os

hdf5_file_path = 'hdf_data/2s_withlabel/S04.hdf5'
output_dir = 'actionsense_data/all_sensors_2s/S04'
os.makedirs(output_dir)

with h5py.File(hdf5_file_path, 'r') as hdf_file:
    example_matrices = hdf_file['example_matrices'][:]

    all_sensors_data = example_matrices[:, :, :]
    # emg_data = example_matrices[:, :, 2:18]

    for index, example in enumerate(all_sensors_data):
        filename = f'{output_dir}/{index:03}.npy'
        np.save(filename, example)