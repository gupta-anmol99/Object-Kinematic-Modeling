import torch
from torch.utils.data import DataLoader
from SequencePointCloudDataset import SequencePointCloudDataset
from PointEncoderDecoder import PointNetEncoder
from TransformerEncoder import TransformerEncoder  
from OutputLayer import JointParameterHead, JointDeltaPredictor  
from loss import JointEstimationLoss
import wandb
import os

# --- Config ---
DATA_PATH = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/data/data_sim"
CHECKPOINT_PATH = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/checkpoint/point_encoder_epoch_10.pt"
SAVE_PATH = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/checkpoint/final_model"
NUM_POINTS = 40000
SEQUENCE_LENGTH = 12
LATENT_DIM = 2048
MODEL_DIM = 1024
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-4
MAX_JOINTS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- WandB ---
# wandb.init(project="kinematic-model", name="full-pipeline-training")

# --- Dataset ---
dataset = SequencePointCloudDataset(DATA_PATH, num_points=NUM_POINTS, sequence_length=SEQUENCE_LENGTH, return_labels=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- Models ---
encoder = PointNetEncoder(num_points=NUM_POINTS, latent_dim=LATENT_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
encoder.eval()  # frozen
for param in encoder.parameters():
    param.requires_grad = False

transformer = TransformerEncoder(input_dim=LATENT_DIM, model_dim=MODEL_DIM).to(DEVICE)
joint_param_head = JointParameterHead(input_dim=MODEL_DIM, max_joints=MAX_JOINTS).to(DEVICE)
joint_delta_predictor = JointDeltaPredictor(model_dim=MODEL_DIM, num_joints=MAX_JOINTS, sequence_length=SEQUENCE_LENGTH).to(DEVICE)

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
    for batch in dataloader:
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
        # wandb.log({f"loss/{k}": v for k, v in loss_dict.items()}, step=step)

    # --- Epoch Summary ---
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    # wandb.log({"loss/epoch_avg": avg_loss}, step=step)

    # --- Save checkpoint ---
    if epoch % 5 == 0:
        ckpt_path = os.path.join(SAVE_PATH, f"transformer_epoch_{epoch}.pt")
        torch.save({
            "transformer": transformer.state_dict(),
            "param_head": joint_param_head.state_dict(),
            "delta_predictor": joint_delta_predictor.state_dict(),
        }, ckpt_path)