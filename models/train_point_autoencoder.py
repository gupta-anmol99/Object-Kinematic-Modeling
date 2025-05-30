import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from PointCloudDepthDataset import PointCloudDepthDataset
from PointEncoderDecoder import PointNetEncoder, DepthDecoder
import wandb
import os
from tqdm import tqdm

# --- wandb Init ---
wandb.init(project="Point-Encoder", name="PointNet-Encoder-Train-0")

# --- Hyperparameters ---
NUM_POINTS = 40000
LATENT_DIM = 2048
BATCH_SIZE = 12
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
data_path = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic Modelling/Sequential Joint Estimation/data/data_sim"
dataset = PointCloudDepthDataset(data_path, num_points=NUM_POINTS)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Model ---
encoder = PointNetEncoder(num_points=NUM_POINTS, latent_dim=LATENT_DIM).to(DEVICE)
decoder = DepthDecoder(latent_dim=LATENT_DIM).to(DEVICE)
model = torch.nn.Sequential(encoder, decoder)

# --- Loss + Optimizer ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# --- Checkpoint Path ---
save_path = "/home/local/ASUAD/agupt374/research_directory/Playground/Kinematic Modelling/Sequential Joint Estimation/checkpoint/"
os.makedirs(save_path, exist_ok=True)

# --- Training Loop ---
step_counter = 0
for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Epochs"):
    model.train()
    total_loss = 0.0

    for step, (pointclouds, depths) in enumerate(train_loader, start=1):
        pointclouds = pointclouds.to(DEVICE)
        depths = depths.to(DEVICE)

        preds = model(pointclouds)
        loss = criterion(preds, depths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pointclouds.size(0)

        step_counter += 1
        if step_counter % 100 == 0:
            wandb.log({"train/mse_loss": loss.item(), "step": step_counter})

        if step % 2500 == 0:
            print(f"  Step {step}: Batch MSE Loss = {loss.item():.6f}")

    avg_loss = total_loss / len(train_dataset)
    wandb.log({"train/avg_epoch_loss": avg_loss, "epoch": epoch})
    print(f"Epoch {epoch}: Avg MSE Loss = {avg_loss:.6f}")

    # Save encoder checkpoint
    if epoch % 2 == 0:
        print(f"Saving encoder checkpoint for epoch {epoch}...")
        torch.save(encoder.state_dict(), os.path.join(save_path, f"point_encoder_epoch_{epoch}.pt"))
        torch.save(decoder.state_dict(), os.path.join(save_path, f"point_decoder_epoch_{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join(save_path, f"point_model_full_epoch_{epoch}.pt"))
