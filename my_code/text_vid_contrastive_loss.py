import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_code.models.model import Model, AllSensorsModel
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

dataset = ActionsenseDataset("actionsense_data/imagebind_targets_text_2s", "actionsense_data/imagebind_targets_2s")
print("len(dataset):", len(dataset))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

loss_function = ContrastiveLoss()

for i, (text_targets, video_targets, _) in enumerate(dataloader):
    print(loss_function(text_targets, video_targets))