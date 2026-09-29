# SequenceEncoder.py
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Import your existing modules
# Adjust paths if needed
from models.Pointnet2Encoder import PointNet2_SA3
from models.FrameEncoder import FrameCLS


class SequenceEncoder(nn.Module):
    """
    SequenceEncoder
    ----------------
    Turns a sequence of point clouds (or precomputed SA3 tokens) into per-frame CLS embeddings.

    Inputs (one of):
      - Raw point clouds: (B, T, P, 3)  -> SA3 -> (B, T, 64, 512) -> FrameCLS -> (B, T, 512)
      - SA3 tokens:       (B, T, 64, 512) -> FrameCLS -> (B, T, 512)

    Args:
      use_backbone: If True, expect raw points and run PointNet++ SA3.
                    If False, expect precomputed SA3 tokens.
      backbone_kwargs: Dict of kwargs to build PointNet2_SA3 (only used if use_backbone=True).
      d_model: Feature dim (defaults to 512 to match SA3 output).
      n_tokens: Number of SA3 tokens per frame (defaults to 64).
      n_layers_frame: Transformer depth for FrameCLS.
      n_heads_frame:  Attention heads for FrameCLS.
      ff_mult_frame:  FFN expansion for FrameCLS.
      dropout_frame:  Dropout for FrameCLS.
      chunk_size:     If not None, process frames in chunks of this many (over B*T) to reduce memory.
    """

    def __init__(
        self,
        use_backbone: bool = True,
        backbone_kwargs: Optional[dict] = None,
        d_model: int = 512,
        n_tokens: int = 64,
        n_layers_frame: int = 2,
        n_heads_frame: int = 8,
        ff_mult_frame: int = 4,
        dropout_frame: float = 0.1,
        chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.use_backbone = use_backbone
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.n_tokens = n_tokens

        if self.use_backbone:
            if backbone_kwargs is None:
                backbone_kwargs = dict(
                    npoint_sa1=1024, radius_sa1=0.05, k_sa1=32, mlp_sa1=(64, 64, 128),
                    npoint_sa2=256,  radius_sa2=0.10, k_sa2=32, mlp_sa2=(128, 128, 256),
                    npoint_sa3=64,   radius_sa3=0.20, k_sa3=32, mlp_sa3=(256, 256, 512),
                )
            self.backbone = PointNet2_SA3(**backbone_kwargs)
        else:
            self.backbone = None

        self.frame_cls = FrameCLS(
            d_model=d_model,
            n_tokens=n_tokens,
            n_layers=n_layers_frame,
            n_heads=n_heads_frame,
            ff_mult=ff_mult_frame,
            dropout=dropout_frame,
        )

    def forward(self, seq_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
          seq_in:
            - If use_backbone=True:  (B, T, P, 3) float
            - If use_backbone=False: (B, T, 64, 512) float

        Returns:
          frame_cls_seq: (B, T, 512)
        """
        if self.use_backbone:
            assert seq_in.dim() == 4 and seq_in.size(-1) == 3, \
                f"Expected (B,T,P,3) when use_backbone=True, got {tuple(seq_in.shape)}"
            B, T, P, _ = seq_in.shape

            # Flatten over frames to reuse per-frame pipeline
            x = seq_in.reshape(B * T, P, 3)  # (B*T, P, 3)

            if self.chunk_size is None:
                sa3_xyz, sa3_tokens = self.backbone(x)         # (B*T, 64, 3), (B*T, 64, 512)
                frame_cls = self.frame_cls(sa3_tokens)          # (B*T, 512)
            else:
                frame_cls_chunks = []
                for s in range(0, B * T, self.chunk_size):
                    e = min(s + self.chunk_size, B * T)
                    sa3_xyz, sa3_tokens = self.backbone(x[s:e]) # (chunk, 64, 3), (chunk, 64, 512)
                    frame_cls_chunks.append(self.frame_cls(sa3_tokens))  # (chunk, 512)
                frame_cls = torch.cat(frame_cls_chunks, dim=0)  # (B*T, 512)

            frame_cls_seq = frame_cls.view(B, T, self.d_model)  # (B, T, 512)
            return frame_cls_seq

        else:
            # Expect precomputed SA3 tokens
            assert seq_in.dim() == 4 and seq_in.size(-2) == self.n_tokens and seq_in.size(-1) == self.d_model, \
                f"Expected (B,T,{self.n_tokens},{self.d_model}) when use_backbone=False, got {tuple(seq_in.shape)}"
            B, T, N, D = seq_in.shape
            x = seq_in.reshape(B * T, N, D)  # (B*T, 64, 512)

            if self.chunk_size is None:
                frame_cls = self.frame_cls(x)                   # (B*T, 512)
            else:
                frame_cls_chunks = []
                for s in range(0, B * T, self.chunk_size):
                    e = min(s + self.chunk_size, B * T)
                    frame_cls_chunks.append(self.frame_cls(x[s:e]))
                frame_cls = torch.cat(frame_cls_chunks, dim=0)  # (B*T, 512)

            frame_cls_seq = frame_cls.view(B, T, self.d_model)  # (B, T, 512)
            return frame_cls_seq


# --- Convenience factory for your current setting ---
def build_default_sequence_encoder(
    use_backbone: bool = True,
    chunk_size: Optional[int] = None,
) -> SequenceEncoder:
    return SequenceEncoder(
        use_backbone=use_backbone,
        chunk_size=chunk_size,
    )


