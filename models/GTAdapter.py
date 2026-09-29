import torch
import torch.nn.functional as F


def manipulation_rank(deltas, eps=1e-4):
    """
    deltas: (T, M) joint values per frame for the M valid joints.
    Returns the normalized order r~ = r / max(1, M-1), where r is the rank of the
    first frame at which each joint starts moving (ties keep row order).
    """
    T, M = deltas.shape
    moved = (deltas - deltas[:1]).abs() > eps                       # (T, M)
    t_idx = torch.arange(T, device=deltas.device).unsqueeze(1).expand(T, M)
    onset = torch.where(moved, t_idx, torch.full_like(t_idx, T)).min(dim=0).values  # (M,)
    order = torch.argsort(onset * M + torch.arange(M, device=deltas.device))       # stable
    rank = torch.empty(M, device=deltas.device, dtype=torch.float32)
    rank[order] = torch.arange(M, device=deltas.device, dtype=torch.float32)
    return rank / float(max(1, M - 1))


def state_targets(deltas, types):
    """
    deltas: (T, M) joint values, types: (M,) 0 revolute / 1 prismatic.
    Returns (M, T, 3): revolute (sin dθ, cos dθ, 0), prismatic (0, 0, dρ), d = w.r.t. frame 0.
    """
    d = (deltas - deltas[:1]).transpose(0, 1)                   # (M, T)
    rev = torch.stack([torch.sin(d), torch.cos(d), torch.zeros_like(d)], dim=-1)
    pri = torch.stack([torch.zeros_like(d), torch.zeros_like(d), d], dim=-1)
    return torch.where((types == 0).view(-1, 1, 1), rev, pri)


def parse_gt_batch(labels, device=None):
    """
    labels: dict with
      - "joint_params": (B, J, 7) padded; only the first joint_count rows are valid
      - "joint_count":  (B,)
      - "joint_deltas": (B, T, J) optional; used to derive the manipulation order
    Returns:
      gt_batch: dict of lists (len B):
        - dir  : list[Tensor (Mi,3)]
        - point: list[Tensor (Mi,3)]
        - type : list[Tensor (Mi,)]  int64 in {0,1}
        - rank : list[Tensor (Mi,)]  float in [0,1], normalized order
        - state: list[Tensor (Mi,T,3)] per-frame state targets (only if joint_deltas given)
    Notes:
      * Anchor points must already be in the same normalized frame as the input points
        (SequencePointCloudDataset does this).
    """
    jp = labels["joint_params"]          # (B, J, 7) on CPU from DataLoader
    counts = labels["joint_count"]
    deltas = labels.get("joint_deltas")
    B, J, _ = jp.shape

    dirs, points, types, ranks, states = [], [], [], [], []
    for b in range(B):
        Mi = int(counts[b])
        row = jp[b, :Mi].to(device)      # counting rows (not "-1" checks) keeps axes like (-1,0,0)

        if Mi == 0:
            dirs.append(torch.empty(0,3, device=device))
            points.append(torch.empty(0,3, device=device))
            types.append(torch.empty(0, dtype=torch.long, device=device))
            ranks.append(torch.empty(0, device=device))
            if deltas is not None:
                states.append(torch.empty(0, deltas.shape[1], 3, device=device))
            continue

        gt_dir   = F.normalize(row[:,0:3], dim=-1, eps=1e-6)          # (Mi,3) on cuda
        gt_point = row[:,3:6]                                          # (Mi,3) on cuda
        gt_type  = row[:,6].long()                                     # (Mi,)   on cuda

        if deltas is not None:
            d_b = deltas[b, :, :Mi].to(device)
            gt_rank = manipulation_rank(d_b)
            states.append(state_targets(d_b, gt_type))
        else:
            gt_rank = torch.arange(Mi, device=device, dtype=torch.float32) / float(max(1, Mi-1))

        dirs.append(gt_dir)
        points.append(gt_point)
        types.append(gt_type)
        ranks.append(gt_rank)

    gt = {"dir": dirs, "point": points, "type": types, "rank": ranks}
    if deltas is not None:
        gt["state"] = states
    return gt
