import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import nn
from tqdm import tqdm

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

        return torch.tensor(emg_data, dtype=torch.float32), torch.tensor(video_embedding, dtype=torch.float32)

# Define the model
class EMGLanguageBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(EMGLanguageBranch, self).__init__()
        
        # LSTM-based EMG encoder
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        
        # Position embedding layer
        self.position_embedding = nn.Parameter(torch.randn(1, hidden_size))
        
        # EMG Q-former
        self.qformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,  # Number of attention heads
                dim_feedforward=2048,  # Dimension of feedforward network model
                dropout=dropout),
            num_layers=num_layers)
        
        # Linear layer to map EMG representation into the embedding space
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, lengths):
        # LSTM-based EMG encoder
        output, (hn, cn) = self.lstm(x)
        
        # Position embedding
        position = torch.arange(0, output.size(1)).unsqueeze(-1).to(output.device)
        position = self.position_embedding(position)
        output = output + position
        
        # EMG Q-former
        output = self.qformer(output)
        
        # Linear layer
        output = self.linear(output)
        
        # Average pooling over sequence length
        mask = (torch.arange(output.size(1)).unsqueeze(0).to(output.device) >= lengths.unsqueeze(1))
        output = output.masked_fill(mask.unsqueeze(-1), 0)
        output = output.sum(dim=1) / lengths.unsqueeze(-1).to(output.device)
        
        return output

# Custom collate function to handle padding
def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*data)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.stack(labels), lengths

# Hyperparameters
input_size = 8  # Size of EMG data input
hidden_size = 512  # Hidden size of LSTM and Q-former
output_size = 131072  # Size of output video embedding
num_layers = 4  # Number of layers in LSTM and Q-former
dropout = 0.1  # Dropout rate

batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Load the dataset
emg_dir = "actionsense_data/S00_emg_chunks"
video_embedding_dir = "actionsense_data/S00_video_embeddings"
dataset = EMGVideoDataset(emg_dir, video_embedding_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize the model
model = EMGLanguageBranch(input_size, hidden_size, output_size, num_layers, dropout)
model.to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for emg_data, video_embedding, lengths in progress_bar:
        emg_data, video_embedding = emg_data.to(device), video_embedding.to(device)
        
        # Forward pass
        outputs = model(emg_data, lengths)
        loss = criterion(outputs, video_embedding)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=epoch_loss / len(dataloader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
