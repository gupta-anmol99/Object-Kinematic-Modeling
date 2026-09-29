import glob, os, numpy as np, torch
import open3d as o3d

def voxel_downsample(xyz, voxel_size=0.005):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points, dtype=np.float32)

def fixed_resample(xyz, target_n=8192, seed=None):
    rng = np.random.default_rng(seed)
    N = xyz.shape[0]
    if N == target_n:
        return xyz
    if N > target_n:
        idx = rng.choice(N, target_n, replace=False)
    else:
        # upsample by repeating some indices
        extra = rng.choice(N, target_n - N, replace=True)
        idx = np.concatenate([np.arange(N), extra])
    return xyz[idx]

def unit_sphere_params(xyz):
    # center/scale of the unit-sphere normalization, so the same transform can be
    # applied to every frame of a sequence and to its GT
    center = xyz.mean(axis=0)
    scale = max(float(np.linalg.norm(xyz - center, axis=1).max()), 1e-6)
    return center.astype(np.float32), scale



def batched_index_points(points, idx):
    """
    points: (B, N, C)
    idx:    (B, M) or (B, M, K)
    returns:
      (B, M, C) or (B, M, K, C)
    """
    B, N, C = points.shape
    if idx.dim() == 2:
        # (B, M) -> gather along dim=1
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)          # (B, M, C)
        out = points.gather(dim=1, index=idx_expanded)               # (B, M, C)
        return out
    elif idx.dim() == 3:
        # (B, M, K) -> flatten MK, gather, then reshape back
        M, K = idx.shape[1], idx.shape[2]
        idx_flat = idx.reshape(B, M*K)                               # (B, M*K)
        idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, C)      # (B, M*K, C)
        out = points.gather(dim=1, index=idx_expanded)               # (B, M*K, C)
        return out.view(B, M, K, C)                                  # (B, M, K, C)
    else:
        raise ValueError(f"idx must be (B,M) or (B,M,K), got {tuple(idx.shape)}")


def farthest_point_sampling(xyz, n_samples):
    # xyz: (B, N, 3)
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=xyz.device)
    distances = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    batch_indices = torch.arange(B, device=xyz.device)
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B,1,3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, -1)              # (B,N)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, -1).indices
    return centroids  # (B, n_samples)

def ball_query(xyz, centroids, radius, k):
    # xyz: (B, N, 3), centroids: (B, M, 3) -> idx: (B, M, K)
    B, N, _ = xyz.shape
    M = centroids.shape[1]
    # squared distances (B, M, N)
    dist2 = torch.cdist(centroids, xyz, p=2) ** 2
    # K nearest; neighbours outside the radius are replaced by the nearest point
    # (standard PointNet++ padding) instead of pulling in far-away points
    dist2, idx = torch.topk(dist2, k, dim=-1, largest=False)    # (B,M,K), sorted ascending
    outside = dist2 > (radius ** 2)
    idx = torch.where(outside, idx[..., :1].expand_as(idx), idx)
    return idx

def group_points(features, idx):
    # features: (B, N, C), idx: (B, M, K) -> (B, M, K, C)
    return batched_index_points(features, idx)