import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_utils import batched_index_points, farthest_point_sampling, ball_query, group_points


class LocalPointNet(nn.Module):
    def __init__(self, in_ch, mlp_ch=(64, 64, 128)):
        super().__init__()
        layers = []
        last = in_ch
        for ch in mlp_ch:
            layers += [nn.Conv2d(last, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
            last = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, C_in, M, K)
        x = self.net(x)    # (B, C_out, M, K)
        x = torch.max(x, dim=-1).values  # (B, C_out, M)
        return x

class SetAbstraction(nn.Module):
    def __init__(self, n_samples, radius, k, in_ch, out_mlp=(64,64,128), include_xyz=True):
        super().__init__()
        self.n_samples = n_samples
        self.radius = radius
        self.k = k
        self.include_xyz = include_xyz
        c_in = in_ch + (3 if include_xyz else 0)
        self.local = LocalPointNet(c_in, out_mlp)

    def forward(self, xyz, feats):
        """
        xyz:   (B, N, 3)
        feats: (B, N, C_in) or None
        returns:
          new_xyz:   (B, M, 3)
          new_feats: (B, M, C_out)
        """
        B, N, _ = xyz.shape
        # FPS -> centroids
        fps_idx = farthest_point_sampling(xyz, self.n_samples)             # (B,M)
        new_xyz = batched_index_points(xyz, fps_idx)                       # (B,M,3)
        # group neighbors
        idx = ball_query(xyz, new_xyz, self.radius, self.k)                # (B,M,K)
        grouped_xyz = group_points(xyz, idx)                               # (B,M,K,3)
        # relative coords to centroid
        centroid = new_xyz.unsqueeze(2)                                    # (B,M,1,3)
        rel_xyz = grouped_xyz - centroid                                   # (B,M,K,3)

        if feats is not None:
            grouped_feats = group_points(feats, idx)                       # (B,M,K,C_in)
            if self.include_xyz:
                # concat rel xyz
                grouped = torch.cat([rel_xyz, grouped_feats], dim=-1)      # (B,M,K,3+C_in)
            else:
                grouped = grouped_feats
        else:
            grouped = rel_xyz                                              # (B,M,K,3)

        # to (B, C_in, M, K)
        grouped = grouped.permute(0, 3, 1, 2).contiguous()
        # local PointNet + max pool over K
        new_feats = self.local(grouped)                                    # (B,C_out,M)
        # to (B, M, C_out)
        new_feats = new_feats.permute(0, 2, 1).contiguous()
        return new_xyz, new_feats

# --------- PointNet++ backbone up to SA3 ----------

class PointNet2_SA3(nn.Module):
    def __init__(self,
                 npoint_sa1=1024, radius_sa1=0.05, k_sa1=32, mlp_sa1=(64,64,128),
                 npoint_sa2=256,  radius_sa2=0.10, k_sa2=32, mlp_sa2=(128,128,256),
                 npoint_sa3=64,   radius_sa3=0.20, k_sa3=32, mlp_sa3=(256,256,512),
                 include_xyz=True):
        super().__init__()
        # SA1 takes only xyz → in_ch=0 (we’ll concat xyz inside SA)
        self.sa1 = SetAbstraction(npoint_sa1, radius_sa1, k_sa1,
                                  in_ch=0, out_mlp=mlp_sa1, include_xyz=True)
        # SA2 takes features from SA1 → in_ch = mlp_sa1[-1] = 128
        self.sa2 = SetAbstraction(npoint_sa2, radius_sa2, k_sa2,
                                  in_ch=mlp_sa1[-1], out_mlp=mlp_sa2, include_xyz=include_xyz)
        # SA3 takes features from SA2 → in_ch = mlp_sa2[-1] = 256
        self.sa3 = SetAbstraction(npoint_sa3, radius_sa3, k_sa3,
                                  in_ch=mlp_sa2[-1], out_mlp=mlp_sa3, include_xyz=include_xyz)


    def forward(self, xyz):  # xyz: (B, N, 3)
        x = None
        xyz1, x1 = self.sa1(xyz, x)      # -> (B, 1024, 3), (B, 1024, 128)
        xyz2, x2 = self.sa2(xyz1, x1)    # -> (B, 256, 3),  (B, 256, 256)
        xyz3, x3 = self.sa3(xyz2, x2)    # -> (B, 64, 3),   (B, 64, 512)
        return xyz3, x3                  # SA3 tokens: (B, 64, 512)


