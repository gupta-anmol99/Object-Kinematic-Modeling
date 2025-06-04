import torch
from torch.utils.data import DataLoader, random_split
from SequencePointCloudDataset import SequencePointCloudDataset
from PointEncoderDecoder import PointNetEncoder
from TransformerEncoder import TransformerEncoder
from OutputLayer import JointParameterHead, JointDeltaPredictor
from loss import JointEstimationLoss
import wandb
import os

# --- Config ---
DATA_PATH = "/mount/scratch4/anmol/seq_kin/data/data_sim"
CHECKPOINT_PATH = "/home/local/ASURITE/agupt374/projects/obj_kin/checkpoint/final_model/full_model_5.pt"
ENCODER_PATH = "/home/local/ASURITE/agupt374/projects/obj_kin/checkpoint/point_encoder_epoch_10.pt"
NUM_POINTS = 40000
SEQUENCE_LENGTH = 12
LATENT_DIM = 2048
MODEL_DIM = 1024
BATCH_SIZE = 8
MAX_JOINTS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- WandB (optional) ---
# wandb.init(project="kinematic-model", name="eval:epoch_5")

# --- Dataset ---
dataset = SequencePointCloudDataset(DATA_PATH, num_points=NUM_POINTS, sequence_length=SEQUENCE_LENGTH)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(42)
_, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Models ---
encoder = PointNetEncoder(num_points=NUM_POINTS, latent_dim=LATENT_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
encoder = torch.nn.DataParallel(encoder)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

transformer = TransformerEncoder(input_dim=LATENT_DIM, model_dim=MODEL_DIM).to(DEVICE)
joint_param_head = JointParameterHead(input_dim=MODEL_DIM, max_joints=MAX_JOINTS).to(DEVICE)
joint_delta_predictor = JointDeltaPredictor(model_dim=MODEL_DIM, num_joints=MAX_JOINTS, sequence_length=SEQUENCE_LENGTH).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
transformer.load_state_dict(checkpoint["transformer"])
joint_param_head.load_state_dict(checkpoint["param_head"])
joint_delta_predictor.load_state_dict(checkpoint["delta_predictor"])

transformer = torch.nn.DataParallel(transformer)
joint_param_head = torch.nn.DataParallel(joint_param_head)
joint_delta_predictor = torch.nn.DataParallel(joint_delta_predictor)

# --- Loss Function ---
loss_fn = JointEstimationLoss(max_joints=MAX_JOINTS, sequence_length=SEQUENCE_LENGTH)

# --- Evaluation ---
transformer.eval()
joint_param_head.eval()
joint_delta_predictor.eval()

total_loss = 0.0
with torch.no_grad():
    for batch in test_loader:
        pc_seq, labels = batch
        pc_seq = pc_seq.to(DEVICE)
        B, T, C, N = pc_seq.shape

        pc_flat = pc_seq.view(B * T, C, N)
        latent_flat = encoder(pc_flat)
        latent_seq = latent_flat.view(B, T, LATENT_DIM)

        seq_out, summary = transformer(latent_seq)
        joint_logits, joint_params = joint_param_head(summary)
        joint_deltas = joint_delta_predictor(seq_out)

        target_joint_count = labels['joint_count'].to(DEVICE)
        target_joint_params = labels['joint_params'].to(DEVICE)
        target_joint_deltas = labels['joint_deltas'].to(DEVICE)

        loss, loss_dict = loss_fn(
            joint_logits, joint_params, joint_deltas,
            target_joint_count, target_joint_params, target_joint_deltas
        )
        print(f"Batch Loss: {loss.item():.4f}, Count: {loss_dict['count_loss']:.4f}, "
              f"Direction: {loss_dict['direction_loss']:.4f}, Point: {loss_dict['point_loss']:.4f}, "
              f"Delta: {loss_dict['delta_loss']:.4f}")

        total_loss += loss.item()

# --- Report ---
avg_loss = total_loss / len(test_loader)
print(f"Test Avg Loss (Epoch 5): {avg_loss:.4f}")
# wandb.log({"test/loss_avg": avg_loss})
# wandb.finish()
