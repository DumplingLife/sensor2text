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

train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
test_size = len(dataset) - train_size  # Remaining 20% for testing
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize the model
model = EMG2VideoEmbeddingModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
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
        progress_bar.set_postfix(loss=epoch_loss / len(train_dataloader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")

# Test the model
model.eval()
total_loss = 0.0
num_samples = 0
save_count = 0
max_saves = 5  # Save results for first 5 examples

with torch.no_grad():
    for emg_data, video_embedding, lengths in test_dataloader:
        emg_data, video_embedding = emg_data.to(device), video_embedding.to(device)
        
        # Forward pass
        outputs = model(emg_data, lengths)

        # Calculate the loss
        loss = criterion(outputs, video_embedding)

        # Update total loss and sample count
        total_loss += loss.item()
        num_samples += emg_data.size(0)

        # Save results for a few examples
        if save_count < max_saves:
            np.save(f"inference_results_{save_count}.npy", outputs[0].cpu().numpy())
            save_count += 1

# Calculate mean loss
mean_loss = total_loss / num_samples
print(f"Mean loss on test set: {mean_loss:.4f}")