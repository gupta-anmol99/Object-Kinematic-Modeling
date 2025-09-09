import torch
import torch.nn as nn
import torch.nn.functional as F

class JointEstimationLoss(nn.Module):
    def __init__(self, max_joints=3, sequence_length=12):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.max_joints = max_joints
        self.sequence_length = sequence_length

    def forward(self, joint_logits, joint_params, joint_deltas,
                target_joint_count, target_joint_params, target_joint_deltas):
        """
        Inputs:
        - joint_logits: (B, max_joints)
        - joint_params: (B, max_joints, 7) → [dir (3), point (3), type (1)]
        - joint_deltas: (B, T, max_joints)
        - target_joint_count: (B,)
        - target_joint_params: (B, max_joints, 7)
        - target_joint_deltas: (B, T, max_joints)
        """

        # 1. Classification Loss (joint count): logits vs class labels
        count_loss = self.ce_loss(joint_logits, target_joint_count - 1)

        # 2. Valid mask
        param_mask = (target_joint_params[:, :, 0] != -1)  # (B, max_joints)

        # 3. Direction Loss (cosine similarity)
        dir_pred = joint_params[:, :, 0:3]
        dir_true = target_joint_params[:, :, 0:3]
        dir_true = F.normalize(dir_true, dim=-1)
        dot_product = F.cosine_similarity(dir_pred, dir_true, dim=-1)  # (B, max_joints)
        direction_loss = (1 - dot_product[param_mask]).mean()

        # 4. Point Loss (MSE on 3:6)
        point_pred = joint_params[:, :, 3:6]
        point_true = target_joint_params[:, :, 3:6]
        point_loss = self.mse_loss(
            point_pred[param_mask],
            point_true[param_mask]
        )

        # 5. Joint Type Loss (BCE on 6)
        type_pred = joint_params[:, :, 6]
        type_true = target_joint_params[:, :, 6]
        type_loss = self.bce_loss(
            type_pred[param_mask],
            type_true[param_mask]
        )

        # 6. Delta sequence loss (MSE on non-masked entries)
        delta_mask = (target_joint_deltas != -1)
        delta_loss = self.mse_loss(
            joint_deltas[delta_mask],
            target_joint_deltas[delta_mask]
        )

        # Final combined loss (weights can be tuned)
        total_loss = (
            0.2 * count_loss +
            0.3 * direction_loss +
            0.25 * point_loss +
            0.1 * type_loss +
            0.15 * delta_loss
        )

        return total_loss, {
            "count_loss": count_loss.item(),
            "direction_loss": direction_loss.item(),
            "point_loss": point_loss.item(),
            "type_loss": type_loss.item(),
            "delta_loss": delta_loss.item(),
            "total": total_loss.item()
        }
