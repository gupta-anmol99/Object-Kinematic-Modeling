# pip install open3d
import glob, os, numpy as np, torch
import open3d as o3d
from models.scripts.pointnet_utils import fixed_resample, voxel_downsample, to_unit_sphere
from models.scripts.Pointnet2Encoder import PointNet2_SA3
from models.scripts.FrameEncoder import FrameCLS

# ---- 1) PCD loading utilities ----
def load_pcd_xyz(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float32)  # (N, 3)
    print(f"Loaded objects with shape {xyz.shape}")
    if xyz.size == 0:
        raise ValueError(f"No points in {path}")
    return xyz

# ---- 2) Batch loader for a folder of .pcd ----
def load_batch_from_folder(folder, pattern="point_cloud_*.pcd",
                           voxel_size=0.0, target_n=8192, normalize=True, max_files=None):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if max_files is not None:
        paths = paths[:max_files]
    xyz_list = []
    meta = []
    for p in paths:
        xyz = load_pcd_xyz(p)
        if voxel_size and voxel_size > 0:
            xyz = voxel_downsample(xyz, voxel_size)
        xyz = fixed_resample(xyz, target_n=target_n)
        if normalize:
            xyz, center, scale = to_unit_sphere(xyz)
        else:
            center, scale = np.zeros((1,3), np.float32), 1.0
        xyz_list.append(xyz)  # (N,3)
        meta.append(dict(path=p, center=center, scale=scale))
    if not xyz_list:
        raise FileNotFoundError(f"No PCDs matched in {folder}")
    batch = np.stack(xyz_list, axis=0).astype(np.float32)  # (B,N,3)
    return torch.from_numpy(batch), meta  # (B,N,3), list of per-frame meta

# ---- 3) Example: load your fridge and run SA3 ----
folder = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/data/data_sim/fridge_000004"
xyz, meta = load_batch_from_folder(
    folder,
    pattern="point_cloud_*.pcd",
    voxel_size=0.0,     # set e.g. 0.003-0.01 if your clouds are very dense
    target_n=8192,      # set your fixed N here
    normalize=True,
    max_files=1         # change to None or an int to load more frames
)

print("Input xyz:", tuple(xyz.shape))  # (B,N,3)

# ---- 4) Run your SA3 backbone ----
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = PointNet2_SA3(
    npoint_sa1=1024, radius_sa1=0.05, k_sa1=32, mlp_sa1=(64,64,128),
    npoint_sa2=256,  radius_sa2=0.10, k_sa2=32, mlp_sa2=(128,128,256),
    npoint_sa3=64,   radius_sa3=0.20, k_sa3=32, mlp_sa3=(256,256,512),
).to(device).eval()

with torch.no_grad():
    sa3_xyz, sa3_tokens = backbone(xyz.to(device))  # (B,64,3), (B,64,512)


print("SA3 tokens   :", tuple(sa3_tokens.shape))

clsHead = FrameCLS(d_model=512, n_tokens=64, n_layers=2, n_heads=8).to(device)

with torch.no_grad():
    frame_cls = clsHead(sa3_tokens)
print("Frame cls    :", tuple(frame_cls.shape))  # (B, 512)
# If later you need to map predictions back to original coords:
# original_xyz = sa3_xyz.cpu().numpy() * meta[b]['scale'] + meta[b]['center']  (per frame b)


