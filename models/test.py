import torch
from torch.utils.data import DataLoader
from SequencePointCloudDataset import SequencePointCloudDataset
from PointEncoderDecoder import PointNetEncoder
from TransformerEncoder import TransformerEncoder  # Assuming you saved it separately
from OutputLayer import JointParameterHead, JointDeltaPredictor  # Assuming you saved it separately
import os

# --- Config ---
DATA_PATH = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/data/data_sim"
CHECKPOINT_PATH = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic_Modelling/Sequential_Joint_Estimation/checkpoint/point_encoder_epoch_10.pt"
NUM_POINTS = 40000
SEQUENCE_LENGTH = 12
LATENT_DIM = 2048
MODEL_DIM = 1024
BATCH_SIZE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
dataset = SequencePointCloudDataset(DATA_PATH, num_points=NUM_POINTS, sequence_length=SEQUENCE_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Models ---
encoder = PointNetEncoder(num_points=NUM_POINTS, latent_dim=LATENT_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
encoder.eval()

transformer = TransformerEncoder(input_dim=LATENT_DIM, model_dim=MODEL_DIM).to(DEVICE)
transformer.eval()

# Initialize output layers
joint_param_head = JointParameterHead(input_dim=MODEL_DIM, max_joints=3).to(DEVICE)
joint_delta_predictor = JointDeltaPredictor(model_dim=MODEL_DIM, num_joints=3, sequence_length=SEQUENCE_LENGTH).to(DEVICE)
joint_param_head.eval()
joint_delta_predictor.eval()

# --- Run Inference ---
with torch.no_grad():
    for sequence, labels in dataloader:
        # sequence: (B, T, 3, N)
        # labels: dict of tensors
        sequence = sequence.to(DEVICE)
        B, T, C, N = sequence.shape

        input_reshaped = sequence.view(B * T, C, N)
        latents = encoder(input_reshaped)
        latents_seq = latents.view(B, T, LATENT_DIM)

        print(f"Encoded latent sequence shape: {latents_seq.shape}")

        seq_output, summary = transformer(latents_seq)
        print(f"Transformer sequence output shape: {seq_output.shape}")
        print(f"Transformer summary output shape: {summary.shape}")

        joint_logits, joint_params = joint_param_head(summary)
        print(f"Joint logits shape: {joint_logits.shape}")
        print(f"Joint parameters shape: {joint_params.shape}")

        joint_deltas = joint_delta_predictor(seq_output)
        print(f"Joint deltas shape: {joint_deltas.shape}")

        # --- Print Labels ---
        print("\n=== Ground Truth Labels ===")
        print("Joint count:", labels["joint_count"])
        print("Joint params:", labels["joint_params"].shape)     # (B, 3, 7)
        print("Joint deltas:", labels["joint_deltas"].shape)     # (B, 12, 3)
        break
