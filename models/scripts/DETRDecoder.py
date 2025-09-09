import torch
import torch.nn as nn
import torch.nn.functional as F

class JointSetDecoder(nn.Module):
    """
    Inputs:
      memory: (B, S, D) from your SequenceEncoder, where S = T+1 (incl. CLS), D=512
    Outputs:
      slots: (B, K, D) slot features (one per potential joint)
    """
    def __init__(
        self,
        d_model=512,
        num_queries=6,          # K (set to 6 now; you can bump to 8 later for headroom)
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)  # learned queries (K, D)

        # A small Transformer decoder (self-attn on queries + cross-attn to memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # use (B, L, D) everywhere
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, memory: torch.Tensor, memory_key_padding_mask=None):
        """
        memory: (B, S, D) = encoder tokens (you can include the CLS token; it’s fine)
        memory_key_padding_mask: optional (B, S) bool mask if you have variable lengths
        """
        B, S, D = memory.shape
        # Tile learned queries across batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)

        # No tgt mask (we're predicting a set); decoder handles self- and cross-attn
        slots = self.decoder(
            tgt=queries,                   # (B, K, D)
            memory=memory,                 # (B, S, D)
            memory_key_padding_mask=memory_key_padding_mask,  # optional
        )                                  # -> (B, K, D)
        return slots
    

class JointSlotHeads(nn.Module):
    """
    Inputs:
      slots: (B, K, D) from the JointSetDecoder
    Outputs (all per-slot):
      existence_logits: (B, K, 1)         # for BCEWithLogitsLoss
      type_logits:      (B, K, 2)         # {revolute, prismatic} (change as needed)
      axis_dir_unit:    (B, K, 3)         # L2-normalized direction
      anchor_point:     (B, K, 3)         # unconstrained 3D point
      order_score:      (B, K, 1)         # sigmoid in [0,1] (we output pre-sigmoid too if you prefer)
    """
    def __init__(self, d_model=512, hidden=256):
        super().__init__()
        # Small shared MLP trunk can help (optional); here we go head-specific for clarity.

        self.exist_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # logits
        )

        self.type_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # 2-class logits
        )

        self.axis_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)  # raw 3-vec -> will be normalized
        )

        self.point_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)  # 3D anchor
        )

        self.order_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # we'll pass through sigmoid at loss/inference time
        )

    def forward(self, slots: torch.Tensor):
        B, K, D = slots.shape

        existence_logits = self.exist_head(slots)         # (B, K, 1)
        type_logits      = self.type_head(slots)          # (B, K, 2)
        axis_raw         = self.axis_head(slots)          # (B, K, 3)
        anchor_point     = self.point_head(slots)         # (B, K, 3)
        order_logits     = self.order_head(slots)         # (B, K, 1)

        # Normalize axis direction (avoid divide-by-zero)
        axis_dir_unit = F.normalize(axis_raw, dim=-1, eps=1e-6)

        return {
            "existence_logits": existence_logits,   # use BCEWithLogitsLoss
            "type_logits":      type_logits,        # use CrossEntropyLoss
            "axis_dir":         axis_dir_unit,      # supervise with angular loss
            "anchor_point":     anchor_point,       # supervise with point-to-line distance
            "order_logits":     order_logits,       # pass through sigmoid when needed
        }
