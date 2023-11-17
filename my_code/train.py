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
        self.data_files = [f"{data_dir}/emg_{i:03d}.npy" for i in range(len(os.listdir(data_dir)))]
        self.target_files = [f"{target_dir}/{i:03d}.npy" for i in range(len(os.listdir(target_dir)))]

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
learning_rate = 0.0003
batch_size = 32
epochs = 30

dataset = ActionSenseDataset('actionsense_data/S00_emg_chunks_preprocessed', 
                             'actionsense_data/S00_imagebind_embeds')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model(d_model=d_model, nhead=nhead, num_layers=num_layers)
loss_function = ContrastiveLoss()
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

outputs = [None] * 230
targets_list = [None] * 230
for i, (inputs, targets) in enumerate(DataLoader(dataset, batch_size=1, shuffle=True)):
    with torch.no_grad():
        outputs[i] = model(inputs)
        targets_list[i] = targets
        print(torch.mean((outputs-targets)**2)) # MSE

mean_output = sum(outputs) / 230
for output in outputs:
    print(torch.mean((outputs-output)**2))