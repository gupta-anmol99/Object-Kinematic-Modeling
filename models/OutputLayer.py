import torch
import torch.nn as nn
import torch.nn.functional as F

class JointParameterHead(nn.Module):
    def __init__(self, input_dim=1024, max_joints=6):
        super().__init__()
        self.max_joints = max_joints

        # --- Joint Count Prediction (1–6 joints) ---
        self.joint_count_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, max_joints),  # logits for classes 1-6
        )

        # --- Joint Parameter Prediction ---
        self.joint_param_mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, max_joints * 7),  # 7 params per joint
        )

    def forward(self, summary_latent):  # (B, input_dim)
        # Predict joint count
        joint_logits = self.joint_count_mlp(summary_latent)  # (B, 6)

        # Predict joint parameters
        joint_params = self.joint_param_mlp(summary_latent)  # (B, 6*7)
        joint_params = joint_params.view(-1, self.max_joints, 7)  # (B, 6, 7)

        # Normalize direction vector (first 3 values)
        direction = joint_params[:, :, 0:3]
        direction = F.normalize(direction, dim=-1)  # unit norm

        # Pass joint type through sigmoid to get probability
        joint_type = torch.sigmoid(joint_params[:, :, 6:7])

        # Reconstruct full params: [normalized dir, point, type prob]
        joint_params_out = torch.cat([
            direction,                  # (B, 6, 3)
            joint_params[:, :, 3:6],    # point (B, 6, 3)
            joint_type                  # (B, 6, 1)
        ], dim=-1)  # -> (B, 6, 7)

        return joint_logits, joint_params_out  # (B, 6), (B, 6, 7)
    

class JointDeltaPredictor(nn.Module):
    def __init__(self, model_dim=1024, num_joints=6, num_layers=2, num_heads=4, dropout=0.1, sequence_length=12):
        super().__init__()
        self.sequence_length = sequence_length
        self.query_embed = nn.Parameter(torch.randn(1, sequence_length, model_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(model_dim, num_joints)

    def forward(self, memory):  # memory = transformer encoder output → shape: (B, T, D)
        B = memory.size(0)
        query = self.query_embed.expand(B, -1, -1)  # (B, T, D)
        decoded = self.decoder(query, memory)       # (B, T, D)
        joint_deltas = self.output_head(decoded)    # (B, T, 6)
        return joint_deltas
