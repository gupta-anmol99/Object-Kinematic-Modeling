import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, num_points=50000, latent_dim=2048):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv1d(3, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 512, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, 1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv6 = nn.Conv1d(1024, 1024, 1, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)

        self.pool = nn.MaxPool1d(kernel_size=num_points)
        self.dropout = nn.Dropout(p=0.3)

        self.linear1 = nn.Linear(1024, 1024, bias=False)
        self.ln1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, latent_dim, bias=False)

    def forward(self, x):  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.ln1(self.linear1(x)))
        x = self.linear2(x)
        return x


class DepthDecoder(nn.Module):
    def __init__(self, latent_dim=2048, output_size=(1, 200, 200)):
        super().__init__()
        self.output_size = output_size
        self.start_channels = 1024
        self.start_spatial_dim = 4  # 4x4 -> 256x256

        self.fc1 = nn.Linear(latent_dim, 4096)
        self.fc2 = nn.Linear(4096, self.start_channels * self.start_spatial_dim ** 2)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.start_channels, 512, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32 -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64 -> 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 128x128 -> 256x256
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 1, 1)      # 256x256 -> 256x256 (refine)
        )

    def forward(self, x):  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, self.start_channels, self.start_spatial_dim, self.start_spatial_dim)
        x = self.deconv(x)
        x = F.interpolate(x, size=self.output_size[1:], mode="bilinear", align_corners=False)
        return x.view(x.size(0), -1)
