import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-(torch.log(torch.tensor(10000.0)) / dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):  # x: (B, S, D)
        S = x.size(1)
        return x + self.pe[:, :S, :]

class TemporalCLS(nn.Module):
    def __init__(self, input_dim=512, model_dim=512, num_layers=4, num_heads=8, dropout=0.1, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()

        # sequence-level CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encoding = PositionalEncoding(model_dim, max_len=max_len+1)  # +1 for CLS position

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, x, key_padding_mask: torch.Tensor = None):
        """
        x: (B, T, input_dim)  — per-frame embeddings (e.g., (B, 12, 512))
        key_padding_mask: optional (B, T) boolean mask, True = PAD (ignored)
                          If provided, we'll pad a False for CLS internally.
        returns:
          enc:    (B, T+1, model_dim)  — encoded sequence incl. CLS at pos 0
                                         (enc[:, 0] is the sequence embedding, enc[:, 1:] the frames)
        """
        B, T, _ = x.shape

        x = self.input_proj(x)  # (B, T, D)

        # prepend CLS
        cls = self.cls_token.expand(B, 1, -1)     # (B, 1, D)
        x = torch.cat([cls, x], dim=1)            # (B, T+1, D)

        # positional encoding
        x = self.pos_encoding(x)                  # (B, T+1, D)

        # build full key_padding_mask with CLS never masked
        kpm = None
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, T)
            kpm = torch.zeros((B, T+1), dtype=torch.bool, device=x.device)
            kpm[:, 1:] = key_padding_mask  # CLS at index 0 is always unmasked

        enc = self.encoder(x, src_key_padding_mask=kpm)  # (B, T+1, D)
        return self.out_norm(enc)                        # pre-norm layers leave the output un-normalized
