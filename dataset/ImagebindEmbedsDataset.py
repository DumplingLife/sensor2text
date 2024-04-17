import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from common.data_subdirs import subdirs

reworded_captions = {
    "Clean a pan with a sponge" : "A person is cleaning a pan with a sponge.",
    "Clean a pan with a towel" : "A person is cleaning a pan with a towel.",
    "Clean a plate with a sponge" : "A person is cleaning a plate with a sponge.",
    "Clean a plate with a towel" : "A person is cleaning a plate with a towel.",
    "Clear cutting board" : "A person is clearing the cutting board.",
    "Get items from cabinets: 3 each large small plates, bowls, mugs, glasses, sets of utensils" : "A person is getting plates, bowls, mugs, glasses, and utensils from the cabinets.",
    "Get items from refrigerator cabinets drawers" : "A person is retrieving items from the refrigerator, cabinets, and drawers.",
    "Get replace items from refrigerator cabinets drawers" : "A person is replacing items in the refrigerator, cabinets, and drawers.",
    "Load dishwasher: 3 each large small plates, bowls, mugs, glasses, sets of utensils" : "A person is loading the dishwasher with plates, bowls, mugs, glasses, and utensils.",
    "Open a jar of almond butter" : "A person is opening a jar of almond butter.",
    "Open close a jar of almond butter" : "A person is opening and closing a jar of almond butter.",
    "Peel a cucumber" : "A person is peeling a cucumber.",
    "Peel a potato" : "A person is peeling a potato.",
    "Pour water from a pitcher into a glass" : "A person is pouring water from a pitcher into a glass.",
    "Set table: 3 each large small plates, bowls, mugs, glasses, sets of utensils" : "A person is setting the table with plates, bowls, mugs, glasses, and utensils.",
    "Slice a cucumber" : "A person is slicing a cucumber.",
    "Slice a potato" : "A person is slicing a potato.",
    "Slice bread" : "A person is slicing the bread.",
    "Spread almond butter on a bread slice" : "A person is spreading almond butter on a bread slice.",
    "Spread jelly on a bread slice" : "A person is spreading jelly on a bread slice.",
    "Stack on table: 3 each large small plates, bowls" : "A person is putting plates and bowls onto the table.",
    "Unload dishwasher: 3 each large small plates, bowls, mugs, glasses, sets of utensils" : "A person is unloading the plates, bowls, mugs, glasses, and utensils from the dishwasher."
}
for key in reworded_captions:
    reworded_captions[key] += "</s>"


# gets (imagebind_embeds, captions)
# imagebind_embeds is 8 imagebind files, e.g. shape (batch_size, 8, 1024)
class ImagebindEmbedsDataset(Dataset):
    def __init__(self, data_dir, hdf_dir, files_csv):
        self.imagebind_embeds = []
        self.captions = []
        
        hdf_files = {}
        data_files = {}
        for subdir in subdirs:
            hdf_files[subdir] = h5py.File(f"{hdf_dir}/{subdir}.hdf5", 'r')
            data_files[subdir] = os.listdir(f"{data_dir}/{subdir}")
        
        with open(files_csv, "r") as f:
            filepaths = [line.strip() for line in f]
        for filepath in filepaths:
            subdir, idx = filepath.split("/")
            idx = int(idx.replace(".npy", ""))
            label = hdf_files[subdir]["example_labels"][idx]
            # if label and all(f"{i:03d}.npy" in data_files[subdir] for i in range(idx, idx+8)):
            if label and all(f"{subdir}/{i:03d}.npy" in filepaths for i in range(idx, idx+8)) and idx % 8 == 0: # pick 8s
                self.imagebind_embeds.append([f"{data_dir}/{subdir}/{i:03d}.npy" for i in range(idx, idx+8)])
                self.captions.append(label.decode('utf-8').replace("/", " "))

    def __len__(self):
        return len(self.imagebind_embeds)

    def __getitem__(self, idx):
        data = [np.load(imagebind_embed) for imagebind_embed in self.imagebind_embeds[idx]]
        caption = self.captions[idx]
        return torch.from_numpy(np.array(data)).float(), reworded_captions[caption]
