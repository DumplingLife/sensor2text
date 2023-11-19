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
    def __init__(self, data_dir, target_dir):
        self.data_files = [f"{data_dir}/emg_{i:03d}.npy" for i in range(num_examples)]
        self.target_files = [f"{target_dir}/{i:03d}.npy" for i in range(num_examples)]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        target = np.load(self.target_files[idx])
        return torch.from_numpy(data).float(), torch.from_numpy(target).float()

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
epochs = 20

# important
num_examples = 544

dataset = ActionSenseDataset('actionsense_data/S00_emg_chunks_2s', 
                             'actionsense_data/S00_imagebind_embeds_2s')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model(d_model=d_model, nhead=nhead, num_layers=num_layers)
# loss_function = ContrastiveLoss()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):
    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}")

print("Training Complete")


# testing stuff
outputs = [None] * num_examples
targets_list = [None] * num_examples
for i, (inputs, targets) in enumerate(DataLoader(dataset, batch_size=1, shuffle=True)):
    with torch.no_grad():
        outputs[i] = model(inputs)
        targets_list[i] = targets

print(outputs[50])
print(outputs[100])
print(targets_list[50])
print(targets_list[100])

mean_output = sum(outputs) / num_examples
mean_targets = sum(targets_list) / num_examples
squared_error_to_target = 0
squared_error_to_mean = 0
targets_squared_error_to_mean = 0
for output, targets in zip(outputs, targets_list):
    squared_error_to_target += torch.mean((output - targets)**2)
    squared_error_to_mean += torch.mean((mean_output-output)**2)
    targets_squared_error_to_mean += torch.mean((targets-mean_targets)**2)
print(squared_error_to_target / num_examples)
print(squared_error_to_mean / num_examples)
print(targets_squared_error_to_mean / num_examples)

# save
for i in range(num_examples):
    np.save(f"actionsense_data/S00_imagebind_embeds_pred_2s/{i:03d}.npy", outputs[i])