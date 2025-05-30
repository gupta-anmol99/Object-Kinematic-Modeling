import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # Add positional encoding to input
        

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=2048, model_dim=1024, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        self.pos_encoding = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.summary_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.input_proj(x)                     # (B, T, model_dim)
        x = self.pos_encoding(x)                   # (B, T, model_dim)
        encoded = self.transformer(x)              # (B, T, model_dim)
        summary = self.summary_pool(encoded.transpose(1, 2)).squeeze(-1)  # (B, model_dim)
        return encoded, summary                    # (B, T, model_dim), (B, model_dim)
