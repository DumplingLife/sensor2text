import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, input_size=16, d_model=256, nhead=8, num_layers=8, output_size=1024, dropout=0.1, use_cls_token=True, use_pos_encoder=True):
        super().__init__()
        self.input_size = input_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_size)) if use_cls_token else None
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout) if use_pos_encoder else None
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x):
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.input_projection(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.encoder(x)
        if self.cls_token is not None:
            x = x[:, 0, :]
        return self.output_projection(x)

class AllSensorsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_sizes = {'eye': 2, 'emg': 16, 'tactile': 32, 'body': 66}
        modality_d_hidden = 128
        self.encoders = nn.ModuleDict({
            modality: Model(input_size=input_size, d_model=modality_d_hidden, nhead=2, num_layers=2, output_size=modality_d_hidden, dropout=0.1,
                use_cls_token=False, use_pos_encoder=True)
            for modality, input_size in self.input_sizes.items()
        })
        self.final_encoder = Model(input_size=len(self.input_sizes) * modality_d_hidden, d_model=128, nhead=2, num_layers=2, output_size=1024, dropout=0.1,
            use_cls_token=True, use_pos_encoder=False)

    def forward(self, x):
        start = 0
        encoded_modalities = []
        for modality, size in self.input_sizes.items():
            end = start + size
            modality_input = x[:, :, start:end]
            encoded = self.encoders[modality](modality_input)
            encoded_modalities.append(encoded)
            print(encoded.size())
            start = end
        x = torch.cat(encoded_modalities, dim=-1)
        return self.final_encoder(x)