import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameCLS(nn.Module):
    def __init__(self, d_model=512, n_tokens=64, n_layers=2, n_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens

        # Learnable CLS token (1, 1, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Optional: learnable positional embeddings for 64 tokens + CLS (1, 65, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,          # (B, S, D)
            activation="gelu",
            norm_first=True             # pre-norm is usually stabler
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pre_ln = nn.LayerNorm(d_model)  # helps stability
        self.post_ln = nn.LayerNorm(d_model) # optional, for the CLS output

    def forward(self, sa3_feats):
        """
        sa3_feats: (B, 64, 512)  — tokens from PointNet++ SA3
        returns:
            frame_cls: (B, 512)  — frame-level embedding
        """
        B, N, D = sa3_feats.shape
        assert N == self.n_tokens and D == self.d_model

        x = self.pre_ln(sa3_feats)                # (B, 64, 512)

        # Expand and prepend CLS
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, 512)
        x = torch.cat([cls, x], dim=1)           # (B, 65, 512)

        # Add positional embeddings
        x = x + self.pos                          # (B, 65, 512)

        # Encode
        x = self.enc(x)                           # (B, 65, 512)

        # Take CLS (index 0) and finalize
        frame_cls = self.post_ln(x[:, 0, :])      # (B, 512)
        return frame_cls
