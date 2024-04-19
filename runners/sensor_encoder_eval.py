import numpy as np
import os
import torch
from tqdm import tqdm
from models.sensor_encoder import SensorEncoder
from torch.utils.data import DataLoader
from dataset.EncoderDataset import EncoderDataset

device = torch.device("cuda")

save_path = "data/sensor_embeddings_muscle"
dataset = EncoderDataset("data/sensors", "data/imagebind_targets", "data/train_random_8.csv")
val_dataset = EncoderDataset("data/sensors", "data/imagebind_targets", "data/val_random_8.csv")
test_dataset = EncoderDataset("data/sensors", "data/imagebind_targets", "data/test_random_8.csv")
print("# training data:", len(dataset))
print("# validation data:", len(val_dataset))
print("# testing data:", len(test_dataset))

model = SensorEncoder(active_sensors=[False, True, False]).to(device)
model.load_state_dict(torch.load("model_saves/sensor_encoder/muscle_only.pth"))

def evaluate(dataset):
    outputs_list = []
    targets_list = []
    for inputs, targets, filepaths in tqdm(DataLoader(dataset, batch_size=32, shuffle=True)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        outputs_list += outputs
        targets_list += targets

        for output, filepath in zip(outputs, filepaths):
            directory = os.path.dirname(f"{save_path}/{filepath}")
            if not os.path.exists(directory):
                os.mkdir(directory)
            np.save(f"{save_path}/{filepath}", output.cpu().numpy())

    mean_output = sum(outputs_list) / len(dataset)
    mean_targets = sum(targets_list) / len(dataset)
    squared_error_to_target = 0
    squared_error_to_mean = 0
    targets_squared_error_to_mean = 0
    for output, targets in zip(outputs_list, targets_list):
        squared_error_to_target += torch.mean((output - targets)**2)
        squared_error_to_mean += torch.mean((mean_output-output)**2)
        targets_squared_error_to_mean += torch.mean((targets-mean_targets)**2)
    print("MSE:", squared_error_to_target.item() / len(dataset))
    print("average pred squared error to pred mean:", squared_error_to_mean.item() / len(dataset))
    print("average target squared error to target mean:", targets_squared_error_to_mean.item() / len(dataset))

print("Train ====")
evaluate(dataset)
print("Val ====")
evaluate(val_dataset)
print("Test ====")
evaluate(test_dataset)