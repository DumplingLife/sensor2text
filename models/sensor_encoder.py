import torch
import torch.nn as nn
import math

class Tokenizer(nn.Module):
    def __init__(self, sensor_dim, embed_dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.embed = nn.Linear(sensor_dim * chunk_size, embed_dim)

    def forward(self, x):
        x = x.unfold(1, self.chunk_size, self.chunk_size)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.embed(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SingleSensorEncoder(nn.Module):
    def __init__(self, sensor_dim, embed_dim, dim_feedforward, window_size, nhead, nlayers, dropout):
        super().__init__()
        self.tokenizer = Tokenizer(sensor_dim, embed_dim, window_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout, batch_first=True), nlayers)
        
    def forward(self, x):
        x = self.tokenizer(x)
        x = torch.cat([self.class_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x[:, 0, :]

class SensorEncoder(nn.Module):
    def __init__(self, sensor_dims=[2,16,66], active_sensors=[True,True,True], window_size=10, embed_dim=512, dim_feedforward=2048, nhead=8, nlayers=6, output_dim=1024, dropout=0.1):
        super().__init__()

        self.sensor_dims = sensor_dims
        self.active_sensors = active_sensors
        self.sensor_encoders = nn.ModuleList([SingleSensorEncoder(sensor_dim, embed_dim, dim_feedforward, window_size, nhead, nlayers, dropout) for sensor_dim in sensor_dims])
        encoders_output_dim = sum(active_sensors) * embed_dim
        self.fc = nn.Linear(encoders_output_dim, output_dim)

    def forward(self, x):
        single_sensor_outputs = []
        start_idx = 0
        for i in range(len(self.sensor_dims)):
            end_idx = start_idx + self.sensor_dims[i]
            if self.active_sensors[i]:
                x_ = x[:, :, start_idx:end_idx]
                x_ = self.sensor_encoders[i](x_)
                single_sensor_outputs.append(x_)
                start_idx = end_idx
        x = torch.cat(single_sensor_outputs, dim=-1)
        x = self.fc(x)
        return x