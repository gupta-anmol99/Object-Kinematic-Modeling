import os
import glob
import torch
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset

class PointCloudDepthDataset(Dataset):
    def __init__(self, root_dir, num_points=40000):
        self.root_dir = root_dir
        self.num_points = num_points

        self.samples = []
        folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for folder in folders:
            pc_paths = glob.glob(os.path.join(folder, "point_cloud_*.pcd"))
            for pc_path in pc_paths:
                number = os.path.basename(pc_path).split("_")[-1].split(".")[0]
                depth_path = os.path.join(folder, f"depth_frame_{number}.pt")
                if os.path.exists(depth_path):
                    self.samples.append((pc_path, depth_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc_path, depth_path = self.samples[idx]

        # Load point cloud
        pcd = o3d.io.read_point_cloud(pc_path)
        points = np.asarray(pcd.points)  # shape (N, 3)
        if points.shape[0] != self.num_points:
            raise ValueError(f"Expected {self.num_points} points, got {points.shape[0]} in {pc_path}")
        points = torch.tensor(points, dtype=torch.float32).T  # (3, N)

        # Load depth frame
        depth = torch.load(depth_path).float()  # (N,)
        if depth.numel() != self.num_points:
            raise ValueError(f"Expected depth of {self.num_points}, got {depth.shape} in {depth_path}")

        return points, depth
