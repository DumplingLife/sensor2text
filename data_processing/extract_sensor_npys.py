import h5py
import numpy as np
import os

hdf_dir = 'data/actionsense_processed'
output_parent_dir = 'data/sensors'
hdf_files = os.listdir(hdf_dir)
# hdf_files = ["S00.hdf5"]
for hdf_file in hdf_files:
    print(hdf_file)
    hdf_file_path = f'{hdf_dir}/{hdf_file}'
    output_dir = f'{output_parent_dir}/{hdf_file.rsplit(".", 1)[0]}'
    os.makedirs(output_dir)

    with h5py.File(hdf_file_path, 'r') as hdf_file:
        example_matrices = hdf_file['example_matrices'][:]

        all_sensors_data = example_matrices[:, :, :]
        # emg_data = example_matrices[:, :, 2:18]

        for index, example in enumerate(all_sensors_data):
            filename = f'{output_dir}/{index:03}.npy'
            np.save(filename, example)