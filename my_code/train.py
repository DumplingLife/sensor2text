import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from my_code.models.model import Model, AllSensorsModel, load_saved_model
from my_code.data import ActionsenseDataset
from my_code.losses import ContrastiveLoss


learning_rate = 0.0003
batch_size = 32
epochs = 30

dataset = ConcatDataset([
    ActionsenseDataset("actionsense_data/all_sensors_2s", "actionsense_data/imagebind_targets_2s", "video"),
    ActionsenseDataset("actionsense_data/all_sensors_2s", "actionsense_data/imagebind_targets_text_2s", "text"),
])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("len(dataset):", len(dataset))

model = AllSensorsModel()

load_saved_model(model, "my_code/best_model.pt")

contrastive_loss = ContrastiveLoss()
contrastive_loss_weight = 0.0001
mse_loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    epoch_mse_loss = 0
    num_iters = 0
    for i, (inputs, targets, _, flags) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = contrastive_loss(outputs, targets) * contrastive_loss_weight

        video_mask = (flags == "video")
        mse_loss_value = mse_loss(outputs[video_mask], targets[video_mask])
        loss += mse_loss_value

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_mse_loss += mse_loss_value
        num_iters += 1
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / num_iters}, MSE Loss: {epoch_mse_loss / num_iters}")

# testing stuff
outputs = [None] * len(dataset)
targets_list = [None] * len(dataset)
output_paths = [None] * len(dataset)
for i, (inputs, targets, filepath, _) in enumerate(DataLoader(dataset, batch_size=1, shuffle=True)):
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
print("MSE:", squared_error_to_target.item() / len(dataset))
print("average pred squared error to pred mean:", squared_error_to_mean.item() / len(dataset))
print("average target squared error to target mean:", targets_squared_error_to_mean.item() / len(dataset))


for i in range(len(dataset)):
    np.save(f"actionsense_data/imagebind_preds_2s/{output_paths[i]}", outputs[i])