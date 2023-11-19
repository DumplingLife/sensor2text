import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from my_code.model import Model
import os
from tqdm import tqdm

class ActionSenseDataset(Dataset):
    def __init__(self):
        data_dir = "actionsense_data/emg_2s"
        target_dir = "actionsense_data/imagebind_targets_2s"
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

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        target = np.load(self.target_files[idx])
        return torch.from_numpy(data).float(), torch.from_numpy(target).float(), self.filepaths[idx]

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries, keys):
        queries = F.normalize(queries, p=2, dim=1)
        keys = F.normalize(keys, p=2, dim=1)
        similarity = torch.matmul(queries, keys.T) / self.temperature
        labels = torch.arange(queries.size(0)).to(queries.device)
        loss = F.cross_entropy(similarity, labels)
        return loss

d_model = 16
nhead = 4
num_layers = 2
learning_rate = 0.001
batch_size = 32
epochs = 30

dataset = ActionSenseDataset()
print("len(dataset):", len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model(d_model=d_model, nhead=nhead, num_layers=num_layers)
# loss_function = ContrastiveLoss()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):
    for i, (inputs, targets, _) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}")

print("Training Complete")


# testing stuff
outputs = [None] * len(dataset)
targets_list = [None] * len(dataset)
output_paths = [None] * len(dataset)
for i, (inputs, targets, filepath) in enumerate(DataLoader(dataset, batch_size=1, shuffle=True)):
    with torch.no_grad():
        outputs[i] = model(inputs)
        targets_list[i] = targets
        output_paths[i] = filepath

print(outputs[50])
print(outputs[100])
print(targets_list[50])
print(targets_list[100])

mean_output = sum(outputs) / len(dataset)
mean_targets = sum(targets_list) / len(dataset)
squared_error_to_target = 0
squared_error_to_mean = 0
targets_squared_error_to_mean = 0
for output, targets in zip(outputs, targets_list):
    squared_error_to_target += torch.mean((output - targets)**2)
    squared_error_to_mean += torch.mean((mean_output-output)**2)
    targets_squared_error_to_mean += torch.mean((targets-mean_targets)**2)
print(squared_error_to_target / len(dataset))
print(squared_error_to_mean / len(dataset))
print(targets_squared_error_to_mean / len(dataset))


for i in range(len(dataset)):
    np.save(f"actionsense_data/imagebind_preds_2s/{output_paths[i]}.npy", outputs[i])