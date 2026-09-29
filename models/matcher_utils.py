import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def point_to_line_distance(pred_point, gt_point, gt_dir_unit, eps=1e-6):
    v = pred_point - gt_point
    proj = (v * gt_dir_unit).sum(dim=-1, keepdim=True) * gt_dir_unit
    ortho = v - proj
    return torch.sqrt((ortho * ortho).sum(dim=-1) + eps)

def build_cost_matrix(pred, gt_dir, gt_point, gt_type, w_type=1.0, w_angle=1.0, w_point=1.0, w_conf=0.0, eps=1e-6):
    axis_dir = F.normalize(pred["axis_dir"], dim=-1, eps=eps)        # (K,3)
    anchor   = pred["anchor_point"]    # (K,3)
    type_log = pred["type_logits"]     # (K,2)

    K = axis_dir.shape[0]
    M = gt_dir.shape[0]

    gt_dir = F.normalize(gt_dir, dim=-1, eps=eps)

    # type nll
    logp = F.log_softmax(type_log, dim=-1)               # (K,2)
    gt_type_exp = gt_type.unsqueeze(0).expand(K, M)      # (K,M)
    logp_exp    = logp.unsqueeze(1).expand(K, M, 2)      # (K,M,2)
    type_cost   = -torch.gather(logp_exp, 2, gt_type_exp.unsqueeze(-1)).squeeze(-1)  # (K,M)

    # angle 1 - |cos|
    cos = torch.einsum('kd,md->km', axis_dir, gt_dir).clamp(-1, 1)
    ang_cost = 1.0 - cos.abs()

    # point-to-line
    dists = point_to_line_distance(
        anchor.unsqueeze(1).expand(K, M, 3),
        gt_point.unsqueeze(0).expand(K, M, 3),
        gt_dir.unsqueeze(0).expand(K, M, 3),
    )

    cost = w_type * type_cost + w_angle * ang_cost + w_point * dists  # (K,M)
    if w_conf > 0 and "existence_logits" in pred:
        # DETR-style: prefer confident slots, so duplicates are not matched at random
        conf = torch.sigmoid(pred["existence_logits"].reshape(K, 1))
        cost = cost - w_conf * conf
    return cost

def match_one_scipy(C: torch.Tensor):
    # Solve on CPU/NumPy
    k_idx, m_idx = linear_sum_assignment(C.detach().cpu().numpy())
    pairs = list(zip(k_idx.tolist(), m_idx.tolist()))
    unmatched_k = [k for k in range(C.shape[0]) if k not in k_idx]
    return pairs, unmatched_k
