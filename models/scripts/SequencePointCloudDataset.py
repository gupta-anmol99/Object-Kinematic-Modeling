import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
import json
from torch.utils.data import DataLoader
from models.scripts.pointnet_utils import fixed_resample, voxel_downsample, to_unit_sphere

class SequencePointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=8192, sequence_length=12, need_resampling= False, voxel_size=0.03, normalize=True):
        self.root_dir = root_dir
        self.num_points = num_points
        self.sequence_length = sequence_length
        self.voxel_size = voxel_size
        self.normalize = normalize
        self.need_resampling = need_resampling

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

        sequence = []
        for pcd_path in pcd_files:
            pcd = o3d.io.read_point_cloud(pcd_path)
            xyz = np.asarray(pcd.points, dtype=np.float32)  # (N, 3)

            # --- Preprocessing ---
            if self.voxel_size and self.voxel_size > 0:
                xyz = voxel_downsample(xyz, self.voxel_size)
            if self.need_resampling:
                xyz = fixed_resample(xyz, target_n=self.num_points)
            if self.normalize:
                xyz, center, scale = to_unit_sphere(xyz)
            else:
                center, scale = np.zeros((1,3), np.float32), 1.0

            points = torch.from_numpy(xyz).float()  # (N,3)
            sequence.append(points)

        sequence = torch.stack(sequence, dim=0)  # (T, N, 3)

        # --- Load labels ---
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {folder}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        labels = {
            "joint_count": torch.tensor(meta["joint_count"], dtype=torch.long),
            "joint_params": torch.tensor(meta["joint_params"], dtype=torch.float32),
            "joint_deltas": torch.tensor(meta["joint_deltas"], dtype=torch.float32),
        }

        return sequence, labels
    


def main():
    root_dir = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/data/data_sim"  # <-- change this
    dataset = SequencePointCloudDataset(root_dir, num_points=10000, sequence_length=12, need_resampling=True, voxel_size=0.03)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Grab one batch
    for batch in dataloader:
        sequences, labels = batch
        print("Sequences shape:", sequences.shape)  
        print("joint_count shape:", labels["joint_count"].shape)
        print("joint_params shape:", labels["joint_params"].shape)
        print("joint_deltas shape:", labels["joint_deltas"].shape)
        break  # just the first batch

if __name__ == "__main__":
    main()

