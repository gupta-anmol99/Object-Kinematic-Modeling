import torch
import torch.nn as nn
import torch.nn.functional as F

class JointEstimationLoss(nn.Module):
    def __init__(self, max_joints=3, sequence_length=12):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.max_joints = max_joints
        self.sequence_length = sequence_length

    def forward(self, joint_logits, joint_params, joint_deltas,
                target_joint_count, target_joint_params, target_joint_deltas):
        """
        Inputs:
        - joint_logits: (B, max_joints)
        - joint_params: (B, max_joints, 7)
        - joint_deltas: (B, T, max_joints)
        - target_joint_count: (B,)
        - target_joint_params: (B, max_joints, 7)
        - target_joint_deltas: (B, T, max_joints)
        """

        # 1. Classification Loss (joint count): logits vs class labels
        count_loss = self.ce_loss(joint_logits, target_joint_count - 1)

        # 2. Joint Parameter Loss (only for valid joints)
        param_mask = (target_joint_params[:, :, 0] != -1)  # (B, max_joints)
        param_mask_expanded = param_mask.unsqueeze(-1).expand_as(joint_params)

        joint_param_loss = self.mse_loss(
            joint_params[param_mask_expanded],
            target_joint_params[param_mask_expanded]
        )

        # Optional: Angle alignment loss for direction vectors
        dir_pred = joint_params[:, :, 0:3]
        dir_true = target_joint_params[:, :, 0:3]
        dir_true = F.normalize(dir_true, dim=-1)

        dot_product = F.cosine_similarity(dir_pred, dir_true, dim=-1)  # (B, max_joints)
        alignment_loss = 1 - dot_product[param_mask]  # want cos(θ) → 1
        direction_loss = alignment_loss.mean()

        # 3. Delta sequence loss (mask where GT is -1)
        delta_mask = (target_joint_deltas != -1)
        delta_loss = self.mse_loss(
            joint_deltas[delta_mask],
            target_joint_deltas[delta_mask]
        )

        # Final combined loss
        total_loss = (
            0.3 * count_loss +
            0.3 * joint_param_loss +
            0.2 * direction_loss +
            0.2 * delta_loss
        )

        return total_loss, {
            "count_loss": count_loss.item(),
            "joint_param_loss": joint_param_loss.item(),
            "direction_loss": direction_loss.item(),
            "delta_loss": delta_loss.item(),
            "total": total_loss.item()
        }
