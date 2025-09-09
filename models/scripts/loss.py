import torch
import torch.nn as nn
import torch.nn.functional as F

def point_to_line_distance(p, a, d, eps=1e-6):
    v = p - a
    proj = (v * d).sum(dim=-1, keepdim=True) * d
    ortho = v - proj
    return torch.sqrt((ortho * ortho).sum(dim=-1) + eps)

class JointSetLoss(nn.Module):
    def __init__(self,
                 w_exist=1.0, w_type=1.0, w_axis=1.0, w_point=1.0,
                 w_rank_l1=1.0, w_rank_pair=0.5, pair_margin=0.1):
        super().__init__()
        self.w_exist, self.w_type = w_exist, w_type
        self.w_axis, self.w_point = w_axis, w_point
        self.w_rank_l1, self.w_rank_pair = w_rank_l1, w_rank_pair
        self.pair_margin = pair_margin
        self.ce = nn.CrossEntropyLoss()
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, preds, gt_batch, assignments):
        B, K, _ = preds["axis_dir"].shape
        device = preds["axis_dir"].device

        exist_losses, type_losses, axis_losses = [], [], []
        point_losses, rank_l1_losses, rank_pair_losses = [], [], []

        for b in range(B):
            pairs = assignments[b]["pairs"]
            unmatched_k = assignments[b]["unmatched_k"]
            Mi = assignments[b]["M"]

            exist_log_b = preds["existence_logits"][b].squeeze(-1)  # (K,)
            type_log_b  = preds["type_logits"][b]                   # (K,2)
            axis_b      = F.normalize(preds["axis_dir"][b], dim=-1, eps=1e-6)
            anchor_b    = preds["anchor_point"][b]
            order_log_b = preds["order_logits"][b].squeeze(-1)      # (K,)

            gt_dir   = gt_batch["dir"][b]
            gt_point = gt_batch["point"][b]
            gt_type  = gt_batch["type"][b]
            gt_rank  = gt_batch["rank"][b]

            # existence: matched -> 1, unmatched -> 0
            ex_logits, ex_targets = [], []
            if pairs:
                k_pos = torch.tensor([k for k, _ in pairs], device=device)
                ex_logits.append(exist_log_b.index_select(0, k_pos))
                ex_targets.append(torch.ones_like(k_pos, dtype=torch.float32))
            if unmatched_k:
                k_neg = torch.tensor(unmatched_k, device=device)
                ex_logits.append(exist_log_b.index_select(0, k_neg))
                ex_targets.append(torch.zeros_like(k_neg, dtype=torch.float32))
            if ex_logits:
                ex_logits = torch.cat(ex_logits, 0); ex_targets = torch.cat(ex_targets, 0)
                exist_losses.append(self.bce_logits(ex_logits, ex_targets))

            if Mi == 0 or not pairs:
                continue

            k_idx = torch.tensor([k for k, _ in pairs], device=device)
            m_idx = torch.tensor([m for _, m in pairs], device=device)

            # type CE
            type_losses.append(self.ce(type_log_b.index_select(0, k_idx),
                                       gt_type.index_select(0, m_idx)))

            # axis: 1 - |cos|
            pred_dirs = axis_b.index_select(0, k_idx)
            gt_dirs   = F.normalize(gt_dir.index_select(0, m_idx), dim=-1, eps=1e-6)
            cos = (pred_dirs * gt_dirs).sum(-1).clamp(-1, 1)
            axis_losses.append((1.0 - cos.abs()).mean())

            # point-to-line
            pred_pts = anchor_b.index_select(0, k_idx)
            gt_pts   = gt_point.index_select(0, m_idx)
            point_losses.append(point_to_line_distance(pred_pts, gt_pts, gt_dirs).mean())

            # order: L1 + pairwise ranking
            s   = torch.sigmoid(order_log_b.index_select(0, k_idx))
            r_t = gt_rank.index_select(0, m_idx)
            rank_l1_losses.append(torch.abs(s - r_t).mean())

            if r_t.numel() >= 2:
                ti = r_t.unsqueeze(1); tj = r_t.unsqueeze(0)
                mask = (ti < tj)
                if mask.any():
                    si = s.unsqueeze(1).expand(-1, s.numel())
                    sj = s.unsqueeze(0).expand(s.numel(), -1)
                    pair_loss = F.relu(self.pair_margin - (sj - si))[mask].mean()
                    rank_pair_losses.append(pair_loss)

        def mean0(xs): 
            return torch.stack(xs).mean() if xs else torch.zeros((), device=device)

        L_exist = mean0(exist_losses)
        L_type  = mean0(type_losses)
        L_axis  = mean0(axis_losses)
        L_point = mean0(point_losses)
        L_rL1   = mean0(rank_l1_losses)
        L_rPair = mean0(rank_pair_losses)

        total = (self.w_exist*L_exist + self.w_type*L_type + self.w_axis*L_axis +
                 self.w_point*L_point + self.w_rank_l1*L_rL1 + self.w_rank_pair*L_rPair)

        stats = {"exist": L_exist.item(), "type": L_type.item(), "axis": L_axis.item(),
                 "point": L_point.item(), "rank_l1": L_rL1.item(),
                 "rank_pair": L_rPair.item(), "total": total.item()}
        return total, stats
