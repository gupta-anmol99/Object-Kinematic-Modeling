import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
import json

class SequencePointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=40000, sequence_length=12):
        self.root_dir = root_dir
        self.num_points = num_points
        self.sequence_length = sequence_length

        self.sequence_folders = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.sequence_folders.sort()  
    def __len__(self):
        return len(self.sequence_folders)

    def __getitem__(self, idx):
        folder = self.sequence_folders[idx]
        pcd_files = sorted(glob.glob(os.path.join(folder, "point_cloud_*.pcd")))

        if len(pcd_files) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames in {folder}, got {len(pcd_files)}")

        # --- Load point cloud sequence ---
        sequence = []
        for pcd_path in pcd_files:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            if points.shape[0] != self.num_points:
                raise ValueError(f"{pcd_path} has {points.shape[0]} points, expected {self.num_points}")
            points = torch.tensor(points, dtype=torch.float32).T  # (3, N)
            sequence.append(points)
        sequence = torch.stack(sequence)  # (T, 3, N)

        # --- Load labels ---
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {folder}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        joint_count = torch.tensor(meta["joint_count"], dtype=torch.long)        # scalar
        joint_params = torch.tensor(meta["joint_params"], dtype=torch.float32)   # (3, 7)
        joint_deltas = torch.tensor(meta["joint_deltas"], dtype=torch.float32)   # (12, 3)

        labels = {
            "joint_count": joint_count,
            "joint_params": joint_params,
            "joint_deltas": joint_deltas
        }

        return sequence, labels
