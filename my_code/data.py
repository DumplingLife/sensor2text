import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random

class ActionsenseDataset(Dataset):
    def __init__(self, data_dir, target_dir):
        self.data_files = []
        self.target_files = []
        self.filepaths = []

        # data and targets might not match: directories might not match, and # files in each might not match
        common_subdirs = set(os.listdir(data_dir)) & set(os.listdir(target_dir))
        for subdir in common_subdirs:
            common_files = set(os.listdir(f"{data_dir}/{subdir}")) & set(os.listdir(f"{target_dir}/{subdir}"))
            print(f"found {len(common_files)} files from {subdir}")
            for file in common_files:
                self.data_files.append(f"{data_dir}/{subdir}/{file}")
                self.target_files.append(f"{target_dir}/{subdir}/{file}")
                self.filepaths.append(f"{subdir}/{file}")
        
        # test: shuffle array to screw it up
        random.shuffle(self.data_files)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        target = np.load(self.target_files[idx])
        return torch.from_numpy(data).float(), torch.from_numpy(target).float(), self.filepaths[idx]