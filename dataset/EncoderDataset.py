import numpy as np
from torch.utils.data import Dataset
import os

class EncoderDataset(Dataset):
    def __init__(self, data_dir, target_dir, files_csv):
        self.data_dir = data_dir
        self.target_dir = target_dir
        with open(files_csv, "r") as f:
            self.files = [line.strip() for line in f]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.files[idx])).astype(np.float32)
        target = np.load(os.path.join(self.target_dir, self.files[idx])).astype(np.float32)
        return data, target, self.files[idx]