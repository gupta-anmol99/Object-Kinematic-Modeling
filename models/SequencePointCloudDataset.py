import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
import json
from torch.utils.data import DataLoader
from models.pointnet_utils import fixed_resample, voxel_downsample, unit_sphere_params


class SequencePointCloudDataset(Dataset):
    """
    One sample = one demonstration folder with point_cloud_000000.pcd ... and meta.json.

    Returns:
      sequence: (T, num_points, 3) points, normalized with ONE transform per sequence
                (center/scale of frame 0), so motion between frames is preserved.
      labels:   dict with GT expressed in the same normalized frame as the points:
        joint_count  (,)      number of valid joints M
        joint_params (J, 7)   [axis(3), anchor(3), type] ; rows >= M are padding (-1)
        joint_deltas (T, J)   joint value per frame (rad / normalized units) ; padding -1
        center (3,), scale () to map predictions back to camera metres: x * scale + center
    """
    def __init__(self, root_dir, num_points=8192, sequence_length=12, voxel_size=0.0,
                 far_plane=5.0, categories=None, deterministic=False):
        self.root_dir = root_dir
        self.num_points = num_points
        self.sequence_length = sequence_length
        self.voxel_size = voxel_size
        self.far_plane = far_plane
        self.deterministic = deterministic

        self.sequence_folders = sorted(
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        if categories:
            self.sequence_folders = [f for f in self.sequence_folders
                                     if category_of(f) in set(categories)]

    def __len__(self):
        return len(self.sequence_folders)

    def __getitem__(self, idx):
        folder = self.sequence_folders[idx]
        sequence, center, scale = load_sequence(
            folder, num_points=self.num_points, sequence_length=self.sequence_length,
            voxel_size=self.voxel_size, far_plane=self.far_plane,
            seed=idx if self.deterministic else None)

        # --- Load labels ---
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {folder}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        M = int(meta["joint_count"])
        params = np.array(meta["joint_params"], dtype=np.float32)   # (J, 7)
        deltas = np.array(meta["joint_deltas"], dtype=np.float32)   # (T, J)

        # same transform as the points: anchors are positions, prismatic values are lengths
        params[:M, 3:6] = (params[:M, 3:6] - center) / scale
        for j in range(M):
            if params[j, 6] == 1:
                deltas[:, j] = deltas[:, j] / scale

        labels = {
            "joint_count": torch.tensor(M, dtype=torch.long),
            "joint_params": torch.from_numpy(params),
            "joint_deltas": torch.from_numpy(deltas),
            "center": torch.from_numpy(center),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }

        return sequence, labels


def load_sequence(folder, num_points=8192, sequence_length=None, voxel_size=0.0,
                  far_plane=5.0, seed=None):
    """
    Read point_cloud_*.pcd from a folder and normalize the whole sequence with the
    center/scale of frame 0 (one transform per sequence, so motion is preserved).
    Returns sequence (T, num_points, 3) float tensor, center (3,) np.float32, scale float.
    """
    pcd_files = sorted(glob.glob(os.path.join(folder, "point_cloud_*.pcd")))
    if not pcd_files:
        raise FileNotFoundError(f"no point_cloud_*.pcd in {folder}")
    if sequence_length is not None and len(pcd_files) != sequence_length:
        raise ValueError(f"Expected {sequence_length} frames in {folder}, got {len(pcd_files)}")

    frames = []
    for path in pcd_files:
        xyz = np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float32)  # (N, 3)
        # depth pixels with no surface are clamped to the far plane -> drop them
        if far_plane is not None:
            fg = xyz[xyz[:, 2] < far_plane - 1e-3]
            if len(fg) >= 16:
                xyz = fg
        if voxel_size and voxel_size > 0:
            xyz = voxel_downsample(xyz, voxel_size)
        frames.append(xyz)

    center, scale = unit_sphere_params(frames[0])
    sequence = []
    for t, xyz in enumerate(frames):
        xyz = fixed_resample(xyz, target_n=num_points, seed=None if seed is None else seed * 1000 + t)
        sequence.append(torch.from_numpy((xyz - center) / scale).float())
    return torch.stack(sequence, dim=0), center, scale


def category_of(folder):
    # "fridge_000012" -> "fridge"
    return os.path.basename(folder.rstrip("/")).rsplit("_", 1)[0]


def main():
    import sys
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "data/data_sim"
    dataset = SequencePointCloudDataset(root_dir, num_points=8192, sequence_length=12)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Grab one batch
    for batch in dataloader:
        sequences, labels = batch
        print("Sequences shape:", sequences.shape)
        for k, v in labels.items():
            print(k, tuple(v.shape))
        break  # just the first batch

if __name__ == "__main__":
    main()
