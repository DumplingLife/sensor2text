import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import nn

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

        return torch.tensor(emg_data, dtype=torch.float32), torch.tensor(video_embedding, dtype=torch.float32)

# Define the model
class EMG2VideoEmbeddingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(EMG2VideoEmbeddingModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack padded sequence
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate LSTM
        packed_out, (hn, cn) = self.lstm(x)

        # Decode the hidden state of the last time step
        out = self.fc(hn[-1])
        return out

# Custom collate function to handle padding
def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*data)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.stack(labels), lengths

# Hyperparameters
input_size = 8  # Each EMG sample has 8 channels
hidden_size = 128  # Adjust this based on your requirements
output_size = 131072  # Flattened video embeddings
num_layers = 2  # Number of LSTM layers
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Load the dataset
emg_dir = "actionsense_data/S00_emg_chunks"
video_embedding_dir = "actionsense_data/S00_video_embeddings"
dataset = EMGVideoDataset(emg_dir, video_embedding_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize the model
model = EMG2VideoEmbeddingModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for data in dataloader:
        emg_data, video_embedding, lengths = data
        # Forward pass
        outputs = model(emg_data, lengths)
        loss = criterion(outputs, video_embedding)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
