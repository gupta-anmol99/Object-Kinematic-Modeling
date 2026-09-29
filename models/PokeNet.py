import torch
import torch.nn as nn

from models.SequenceEncoder import SequenceEncoder
from models.TemporalEncoder import TemporalCLS
from models.DETRDecoder import JointSetDecoder, JointSlotHeads, JointStateHead


class PokeNet(nn.Module):
    """
    Point-cloud sequence (B, T, P, 3) -> per-slot joint hypotheses.

    PointNet++ SA3 tokens -> frame [CLS] transformer -> temporal transformer (memory)
    -> K learnable joint queries (DETR decoder) -> slot heads + per-frame state head.
    """
    def __init__(self, d_model=512, num_queries=6, max_frames=64,
                 temporal_layers=4, decoder_layers=3, chunk_size=None):
        super().__init__()
        self.seq_encoder = SequenceEncoder(use_backbone=True, d_model=d_model, chunk_size=chunk_size)
        self.temporal = TemporalCLS(input_dim=d_model, model_dim=d_model,
                                    num_layers=temporal_layers, max_len=max_frames)
        self.decoder = JointSetDecoder(d_model=d_model, num_queries=num_queries,
                                       num_layers=decoder_layers)
        self.heads = JointSlotHeads(d_model=d_model)
        self.state_head = JointStateHead(d_model=d_model)

    def forward(self, seq):
        frames = self.seq_encoder(seq)      # (B, T, D)
        memory = self.temporal(frames)      # (B, T+1, D), CLS at index 0
        slots = self.decoder(memory)        # (B, K, D)
        out = self.heads(slots)
        out["states"] = self.state_head(slots, memory[:, 1:])  # (B, K, T, 3)
        return out

    @torch.no_grad()
    def predict(self, seq, center, scale, mu=0.5):
        """
        Inference as in the paper: keep slots with confidence > mu, sort by order score.
        center (B,3) / scale (B,) undo the dataset normalization, so anchors and prismatic
        states come back in camera metres (axis directions are unaffected).
        Returns a list (len B) of lists of joints, first-manipulated first.
        """
        out = self.forward(seq)
        return decode_predictions(out, center, scale, mu)


def decode_predictions(out, center, scale, mu=0.5):
    conf = torch.sigmoid(out["existence_logits"].squeeze(-1))   # (B,K)
    order = torch.sigmoid(out["order_logits"].squeeze(-1))      # (B,K)
    jtype = out["type_logits"].argmax(-1)                       # (B,K)
    states = out["states"]                                      # (B,K,T,3)
    results = []
    for b in range(conf.shape[0]):
        keep = torch.nonzero(conf[b] > mu).flatten()
        keep = keep[torch.argsort(order[b, keep])]
        joints = []
        for k in keep.tolist():
            t = int(jtype[b, k])
            s = states[b, k]
            value = torch.atan2(s[:, 0], s[:, 1]) if t == 0 else s[:, 2] * scale[b]
            joints.append({
                "slot": k,
                "confidence": float(conf[b, k]),
                "order_score": float(order[b, k]),
                "type": "revolute" if t == 0 else "prismatic",
                "axis": out["axis_dir"][b, k].cpu(),
                "anchor": (out["anchor_point"][b, k] * scale[b] + center[b]).cpu(),
                "state": value.cpu(),   # per frame, rad or metres, relative to frame 0
            })
        results.append(joints)
    return results
