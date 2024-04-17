import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.sensor_encoder import SensorEncoder
from dataset.EncoderDataset import EncoderDataset

device = torch.device("cuda")

# dataset = EncoderDataset("data/sensors", "data/imagebind_targets", "data/train.csv")
dataset = EncoderDataset("data/sensors", "data/imagebind_targets", "data/train_random_8.csv")
print("# training data:", len(dataset))

model = SensorEncoder(active_sensors=[True, True, True]).to(device)

learning_rate = 0.0002
batch_size = 32
epochs = 200
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    num_iters = 0
    for inputs, targets, _ in DataLoader(dataset, batch_size=batch_size, shuffle=True):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_iters += 1
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / num_iters}")

torch.save(model.state_dict(), "model_saves/sensor_encoder/model_random_split_8.pth")