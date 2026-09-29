import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from models.matcher_utils import  build_cost_matrix, match_one_scipy

class HungarianJointMatcher:
    def __init__(self, w_type=1.0, w_angle=1.0, w_point=1.0, w_conf=1.0):
        self.w_type, self.w_angle, self.w_point, self.w_conf = w_type, w_angle, w_point, w_conf

    @torch.no_grad()
    def __call__(self, preds_b, gts_b):
        axis_dir = preds_b["axis_dir"]       # (B,K,3)
        anchor   = preds_b["anchor_point"]   # (B,K,3)
        type_log = preds_b["type_logits"]    # (B,K,2)

        B, K, _ = axis_dir.shape
        out = []
        for b in range(B):
            gt_dir   = gts_b["dir"][b]      # (Mb,3)
            gt_point = gts_b["point"][b]    # (Mb,3)
            gt_type  = gts_b["type"][b]     # (Mb,)
            Mb = gt_type.shape[0]

            if Mb == 0:
                out.append({"pairs": [], "unmatched_k": list(range(K)), "M": 0})
                continue

            pred_single = {
                "axis_dir": axis_dir[b], "anchor_point": anchor[b], "type_logits": type_log[b],
                "existence_logits": preds_b["existence_logits"][b],
            }
            C = build_cost_matrix(pred_single, gt_dir, gt_point, gt_type,
                                  w_type=self.w_type, w_angle=self.w_angle, w_point=self.w_point,
                                  w_conf=self.w_conf)  # (K,Mb)
            pairs, unmatched_k = match_one_scipy(C)
            out.append({"pairs": pairs, "unmatched_k": unmatched_k, "M": Mb})
        return out