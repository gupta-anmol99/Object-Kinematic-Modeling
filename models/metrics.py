import torch
import torch.nn.functional as F


def line_distance(a1, d1, a2, d2, eps=1e-6):
    """Minimum distance between the 3D lines (a1, d1) and (a2, d2); d1, d2 unit length."""
    n = torch.cross(d1, d2, dim=-1)
    n_norm = n.norm(dim=-1)
    v = a2 - a1
    skew = (v * n).sum(-1).abs() / n_norm.clamp_min(eps)
    ortho = v - (v * d1).sum(-1, keepdim=True) * d1           # parallel lines: point-to-line
    parallel = ortho.norm(dim=-1)
    return torch.where(n_norm > 1e-3, skew, parallel)


class JointMetrics:
    """
    Accumulates the paper's metrics over Hungarian-matched (pred slot, GT joint) pairs:
      axis_deg     angle between predicted and GT axis lines (sign-agnostic)
      disp_cm      line-to-line distance, revolute joints only, camera metres * 100
      state_deg    mean |dθ error| over frames, revolute
      state_cm     mean |dρ error| over frames, prismatic
      type_acc     joint type accuracy
    and per-sequence:
      count_acc    number of slots with confidence > mu equals the number of GT joints
      order_acc    predicted order of the matched joints equals the GT order (M >= 2)
    """
    def __init__(self, mu=0.5):
        self.mu = mu
        self.sums, self.counts = {}, {}

    def _add(self, key, values):
        values = torch.as_tensor(values, dtype=torch.float64).flatten()
        if values.numel() == 0:
            return
        self.sums[key] = self.sums.get(key, 0.0) + float(values.sum())
        self.counts[key] = self.counts.get(key, 0) + values.numel()

    @torch.no_grad()
    def update(self, preds, gt_batch, assignments, center, scale):
        conf = torch.sigmoid(preds["existence_logits"].squeeze(-1))
        for b, asg in enumerate(assignments):
            M = asg["M"]
            self._add("count_acc", [float(int((conf[b] > self.mu).sum()) == M)])
            if M == 0 or not asg["pairs"]:
                continue
            k_idx = torch.tensor([k for k, _ in asg["pairs"]], device=conf.device)
            m_idx = torch.tensor([m for _, m in asg["pairs"]], device=conf.device)

            d_p = F.normalize(preds["axis_dir"][b, k_idx], dim=-1)
            d_g = gt_batch["dir"][b][m_idx]
            cos = (d_p * d_g).sum(-1).abs().clamp(max=1.0)
            self._add("axis_deg", torch.rad2deg(torch.acos(cos)))

            g_type = gt_batch["type"][b][m_idx]
            self._add("type_acc", (preds["type_logits"][b, k_idx].argmax(-1) == g_type).float())

            # back to camera metres (normalization is translation + uniform scale)
            a_p = preds["anchor_point"][b, k_idx] * scale[b] + center[b]
            a_g = gt_batch["point"][b][m_idx] * scale[b] + center[b]
            rev = g_type == 0
            if rev.any():
                self._add("disp_cm", 100 * line_distance(a_p[rev], d_p[rev], a_g[rev], d_g[rev]))

            if "state" in gt_batch and "states" in preds:
                s_p = preds["states"][b, k_idx]                  # (M,T,3)
                s_g = gt_batch["state"][b][m_idx]
                if rev.any():
                    th_p = torch.atan2(s_p[rev, :, 0], s_p[rev, :, 1])
                    th_g = torch.atan2(s_g[rev, :, 0], s_g[rev, :, 1])
                    err = torch.atan2(torch.sin(th_p - th_g), torch.cos(th_p - th_g)).abs()
                    self._add("state_deg", torch.rad2deg(err).mean(-1))
                if (~rev).any():
                    err = (s_p[~rev, :, 2] - s_g[~rev, :, 2]).abs() * scale[b] * 100
                    self._add("state_cm", err.mean(-1))

            if M >= 2 and len(asg["pairs"]) == M:
                o_p = preds["order_logits"][b, k_idx].squeeze(-1)
                o_g = gt_batch["rank"][b][m_idx]
                self._add("order_acc", [float(torch.equal(torch.argsort(o_p), torch.argsort(o_g)))])

    def compute(self):
        return {k: self.sums[k] / self.counts[k] for k in sorted(self.sums)}

    @staticmethod
    def format(m):
        return "  ".join(f"{k} {v:.3f}" for k, v in m.items())
