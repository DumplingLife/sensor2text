import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import nn
from tqdm import tqdm
import math

device = torch.device("cuda")

# Define the custom dataset
class EMGVideoDataset(Dataset):
    def __init__(self, emg_dir, video_embedding_dir):
        self.emg_dir = emg_dir
        self.video_embedding_dir = video_embedding_dir
        self.samples = [f"{i:03d}" for i in range(230)]  # 000 to 229

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples[index]
        emg_path = os.path.join(self.emg_dir, f"emg_{sample_id}.npy")
        video_embedding_path = os.path.join(self.video_embedding_dir, f"video_{sample_id}", "inputs_llama.npy")

        emg_data = np.load(emg_path)
        video_embedding = np.load(video_embedding_path).reshape(-1)

        return sample_id, torch.tensor(emg_data, dtype=torch.float32), torch.tensor(video_embedding, dtype=torch.float32)

# model
class EMG2VideoEmbeddingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(EMG2VideoEmbeddingModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead), num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len).unsqueeze(1).float().to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


# Custom collate function to handle padding
def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sample_id, sequences, labels = zip(*data)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return sample_id, padded_sequences, torch.stack(labels), lengths

# Hyperparameters
# input_size = 8
input_size = 16 # preprocessed has 16 (becuase left and right); original had 8
hidden_size = 512
output_size = 131072
num_layers = 4
nhead = 8

# batch_size = 32
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Load the dataset
# emg_dir = "actionsense_data/S00_emg_chunks"
emg_dir = "actionsense_data/S00_emg_chunks_preprocessed"
video_embedding_dir = "actionsense_data/S00_video_embeddings"
dataset = EMGVideoDataset(emg_dir, video_embedding_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
test_size = len(dataset) - train_size  # Remaining 20% for testing
print(f"{train_size=} {test_size=}")

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Initialize the model
model = EMG2VideoEmbeddingModel(input_size, hidden_size, output_size, num_layers, nhead).to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for sample_id, emg_data, video_embedding, lengths in progress_bar:
        emg_data, video_embedding = emg_data.to(device), video_embedding.to(device)
        
        # Forward pass
        outputs = model(emg_data, lengths)
        loss = criterion(outputs, video_embedding)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=epoch_loss / len(train_dataloader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")

# Test the model
model.eval()
total_loss = 0.0
save_count = 0
max_saves = 5  # Save results for first 5 examples

with torch.no_grad():
    for sample_id, emg_data, video_embedding, lengths in test_dataloader:
        emg_data, video_embedding = emg_data.to(device), video_embedding.to(device)
        
        outputs = model(emg_data, lengths)
        loss = criterion(outputs, video_embedding)
        total_loss += loss.item()

        # Save results for a few examples (first example of each batch, up to max_saves total)
        if save_count < max_saves:
            np.save(f"actionsense_data/pred_emg_video_embeddings/emg_pred_{sample_id[0]}.npy", outputs[0].cpu().numpy())
            save_count += 1

# Calculate mean loss
mean_loss = total_loss / len(test_dataloader)
print(f"Mean loss on test set: {mean_loss:.4f}")