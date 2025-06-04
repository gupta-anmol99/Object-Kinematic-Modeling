import torch
from torch.utils.data import DataLoader, random_split
from SequencePointCloudDataset import SequencePointCloudDataset
from PointEncoderDecoder import PointNetEncoder
from TransformerEncoder import TransformerEncoder  
from OutputLayer import JointParameterHead, JointDeltaPredictor  
from loss import JointEstimationLoss
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  



# --- Config ---
DATA_PATH = "/mount/scratch4/anmol/seq_kin/data/data_sim"
CHECKPOINT_PATH = "/home/local/ASURITE/agupt374/projects/obj_kin/checkpoint/point_encoder_epoch_10.pt"
SAVE_PATH = "/home/local/ASURITE/agupt374/projects/obj_kin/checkpoint/final_model"
NUM_POINTS = 40000
SEQUENCE_LENGTH = 12
LATENT_DIM = 2048
MODEL_DIM = 1024
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-5
MAX_JOINTS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- WandB ---
wandb.init(project="kinematic-model", name="full-pipeline-training:2")

# --- Dataset ---
dataset = SequencePointCloudDataset(DATA_PATH, num_points=NUM_POINTS, sequence_length=SEQUENCE_LENGTH)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- Models ---
encoder = PointNetEncoder(num_points=NUM_POINTS, latent_dim=LATENT_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
encoder = torch.nn.DataParallel(encoder)
encoder.eval()  # frozen
for param in encoder.parameters():
    param.requires_grad = False

transformer = TransformerEncoder(input_dim=LATENT_DIM, model_dim=MODEL_DIM).to(DEVICE)
joint_param_head = JointParameterHead(input_dim=MODEL_DIM, max_joints=MAX_JOINTS).to(DEVICE)
joint_delta_predictor = JointDeltaPredictor(model_dim=MODEL_DIM, num_joints=MAX_JOINTS, sequence_length=SEQUENCE_LENGTH).to(DEVICE)

# Move models to DataParallel if multiple GPUs are available
transformer = torch.nn.DataParallel(transformer)
joint_param_head = torch.nn.DataParallel(joint_param_head)
joint_delta_predictor = torch.nn.DataParallel(joint_delta_predictor)

# --- Optimizer & Loss ---
params = list(transformer.parameters()) + list(joint_param_head.parameters()) + list(joint_delta_predictor.parameters())
optimizer = torch.optim.Adam(params, lr=LR)
loss_fn = JointEstimationLoss(max_joints=MAX_JOINTS, sequence_length=SEQUENCE_LENGTH)

# --- Training Loop ---
step = 0
for epoch in range(1, EPOCHS + 1):
    transformer.train()
    joint_param_head.train()
    joint_delta_predictor.train()

    total_loss = 0.0
    for batch in train_loader:
        pc_seq, labels = batch  # pc_seq: (B, T, 3, N), labels: dict
        pc_seq = pc_seq.to(DEVICE)
        B, T, C, N = pc_seq.shape

        # --- Forward ---
        pc_flat = pc_seq.view(B * T, C, N)
        with torch.no_grad():
            latent_flat = encoder(pc_flat)
        latent_seq = latent_flat.view(B, T, LATENT_DIM)

        seq_out, summary = transformer(latent_seq)
        joint_logits, joint_params = joint_param_head(summary)
        joint_deltas = joint_delta_predictor(seq_out)

        # --- Labels ---
        target_joint_count = labels['joint_count'].to(DEVICE)           # (B,)
        target_joint_params = labels['joint_params'].to(DEVICE)         # (B, 3, 7)
        target_joint_deltas = labels['joint_deltas'].to(DEVICE)         # (B, 12, 3)

        # --- Loss ---
        loss, loss_dict = loss_fn(
            joint_logits, joint_params, joint_deltas,
            target_joint_count, target_joint_params, target_joint_deltas
        )

        # --- Backward ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step} - Loss: {loss.item():.4f}")
        
        if step % 10 == 0:
            wandb.log({f"loss/{k}": v for k, v in loss_dict.items()}, step=step)

    # --- Epoch Summary ---
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    wandb.log({"loss/epoch_avg": avg_loss}, step=step)

    # --- Save checkpoint ---
    if epoch % 5 == 0:
        ckpt_path = os.path.join(SAVE_PATH, f"full_model_{epoch}.pt")
        torch.save({
            "transformer": transformer.module.state_dict(),
            "param_head": joint_param_head.module.state_dict(),
            "delta_predictor": joint_delta_predictor.module.state_dict(),
        }, ckpt_path)



wandb.finish()


