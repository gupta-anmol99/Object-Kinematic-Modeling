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
    jp = labels["joint_params"]          # (B, J, 7) on CPU from DataLoader
    B, J, _ = jp.shape

    dirs, points, types, ranks = [], [], [], []
    for b in range(B):
        row = jp[b].to(device)           # <-- move this sample to cuda
        valid = (row[:,0] != -1) & (row[:,1] != -1) & (row[:,2] != -1)
        row = row[valid]
        Mi = row.shape[0]

        if Mi == 0:
            dirs.append(torch.empty(0,3, device=device))
            points.append(torch.empty(0,3, device=device))
            types.append(torch.empty(0, dtype=torch.long, device=device))
            ranks.append(torch.empty(0, device=device))
            continue

        gt_dir   = F.normalize(row[:,0:3], dim=-1, eps=1e-6)          # (Mi,3) on cuda
        gt_point = row[:,3:6]                                          # (Mi,3) on cuda
        gt_type  = row[:,6].long()                                     # (Mi,)   on cuda

        if Mi == 1:
            gt_rank = torch.zeros(Mi, device=device)
        else:
            gt_rank = torch.arange(Mi, device=device, dtype=torch.float32) / float(Mi-1)

        dirs.append(gt_dir)
        points.append(gt_point)
        types.append(gt_type)
        ranks.append(gt_rank)

    return {"dir": dirs, "point": points, "type": types, "rank": ranks}
