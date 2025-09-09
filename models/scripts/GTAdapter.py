import torch
import torch.nn.functional as F

def parse_gt_batch(labels, device=None):
    """
    labels: dict with
      - "joint_params": (B, J, 7) padded, -1 rows mean 'no joint'
      - "joint_count":  (B,)  (optional, we can infer from mask)
    Returns:
      gt_batch: dict of lists (len B):
        - dir  : list[Tensor (Mi,3)]
        - point: list[Tensor (Mi,3)]
        - type : list[Tensor (Mi,)]  int64 in {0,1}
        - rank : list[Tensor (Mi,)]  float in [0,1], normalized order
    Notes:
      * Assumes anchor points are already in the same normalized space as inputs.
        If not, apply your dataset’s center/scale transform to the GT before calling this.
    """
    jp = labels["joint_params"]            # (B, J, 7)
    B, J, _ = jp.shape
    if device is None:
        device = jp.device

    dirs, points, types, ranks = [], [], [], []

    for b in range(B):
        row = jp[b]                        # (J,7)
        # valid mask: any coord not -1
        valid = (row[:, 0] != -1) & (row[:, 1] != -1) & (row[:, 2] != -1)

        row_valid = row[valid]             # (Mi,7)
        Mi = row_valid.shape[0]
        if Mi == 0:
            # empty sample
            dirs.append(torch.empty(0, 3, device=device))
            points.append(torch.empty(0, 3, device=device))
            types.append(torch.empty(0, dtype=torch.long, device=device))
            ranks.append(torch.empty(0, device=device))
            continue

        # split fields
        gt_dir   = row_valid[:, 0:3]                   # (Mi,3)
        gt_point = row_valid[:, 3:6]                   # (Mi,3)
        gt_type  = row_valid[:, 6].long()              # (Mi,)

        # normalize direction to unit
        gt_dir = F.normalize(gt_dir, dim=-1, eps=1e-6)

        # build ranks from present order (0..Mi-1) → [0,1]
        if Mi == 1:
            gt_rank = torch.zeros(Mi, device=device)   # avoid /0, single joint gets 0
        else:
            gt_rank = torch.arange(Mi, device=device, dtype=torch.float32) / float(Mi - 1)

        dirs.append(gt_dir)
        points.append(gt_point)
        types.append(gt_type)
        ranks.append(gt_rank)

    return {"dir": dirs, "point": points, "type": types, "rank": ranks}
