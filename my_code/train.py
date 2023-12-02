import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_code.model import Model, AllSensorsModel
from my_code.data import ActionsenseDataset


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


learning_rate = 0.001
batch_size = 32
epochs = 30

# dataset = ActionsenseDataset("actionsense_data/emg_2s", "actionsense_data/imagebind_targets_text_2s")
dataset = ActionsenseDataset("actionsense_data/all_sensors_2s", "actionsense_data/imagebind_targets_text_2s")
print("len(dataset):", len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model = Model()
model = AllSensorsModel()

loss_function = ContrastiveLoss()
# loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):
    total_loss = 0
    num_iters = 0
    for i, (inputs, targets, _) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_iters += 1

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/num_iters}")


# testing stuff
outputs = [None] * len(dataset)
targets_list = [None] * len(dataset)
output_paths = [None] * len(dataset)
for i, (inputs, targets, filepath) in enumerate(DataLoader(dataset, batch_size=1, shuffle=True)):
    with torch.no_grad():
        outputs[i] = model(inputs)[0]
        targets_list[i] = targets[0]
        output_paths[i] = filepath[0]

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
print("MSE:", squared_error_to_target / len(dataset))
print("average pred squared error to pred mean:", squared_error_to_mean / len(dataset))
print("average target squared error to target mean:", targets_squared_error_to_mean / len(dataset))


for i in range(len(dataset)):
    np.save(f"actionsense_data/imagebind_preds_2s/{output_paths[i]}", outputs[i])