"""Evaluation metrics for human pose estimation (PCK and MPJPE)."""

import numpy as np
import torch

LSP_JOINT_NAMES: list[str] = [
    "R_Ankle",
    "R_Knee",
    "R_Hip",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "R_Wrist",
    "R_Elbow",
    "R_Shoulder",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "Thorax",
    "Head",
]


def compute_mpjpe(
    preds: torch.Tensor,
    gts: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Mean Per Joint Position Error (MPJPE) in Euclidean pixel distance.

    Args:
        preds: Predicted coordinates tensor of shape (B, J, 2).
        gts: Ground truth coordinates tensor or numpy array of shape (B, J, 2) or (B, 3, J).
        visibility: Optional visibility tensor (B, J).

    Returns:
        Tuple of (mean_mpjpe, per_joint_mpjpe).
    """
    device = preds.device
    gt_tensor = torch.as_tensor(gts, device=device, dtype=preds.dtype)

    # If gt is (B, 3, J) -> (coords, joints)
    if gt_tensor.dim() == 3 and gt_tensor.shape[1] == 3:
        if visibility is None:
            vis_tensor = (gt_tensor[:, 2, :] <= 1).float()
        else:
            vis_tensor = torch.as_tensor(visibility, device=device, dtype=torch.float32)
        gt_coords = gt_tensor[:, :2, :].permute(0, 2, 1)  # (B, J, 2)
    else:
        gt_coords = gt_tensor
        if visibility is not None:
            vis_tensor = torch.as_tensor(visibility, device=device, dtype=torch.float32)
        else:
            vis_tensor = torch.ones(preds.shape[:2], device=device, dtype=torch.float32)

    # Euclidean distance per joint
    dist = torch.sqrt(torch.sum((preds - gt_coords) ** 2, dim=-1))  # (B, J)
    dist = dist * vis_tensor

    sum_vis = torch.clamp(torch.sum(vis_tensor, dim=0), min=1e-6)
    per_joint_error = torch.sum(dist, dim=0) / sum_vis

    total_vis = torch.clamp(torch.sum(vis_tensor), min=1e-6)
    mean_error = torch.sum(dist) / total_vis

    return mean_error, per_joint_error


def compute_pck(
    preds: torch.Tensor,
    gts: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray | None = None,
    threshold: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Percentage of Correct Keypoints (PCK).

    If threshold < 1.0, calculates torso-relative PCK (standard PCK@0.2), where the
    tolerance is threshold * torso_length (midpoint of shoulders to midpoint of hips).
    If threshold >= 1.0, threshold is treated as absolute pixel radius.

    Args:
        preds: Predicted coordinates tensor of shape (B, J, 2).
        gts: Ground truth coordinates tensor or numpy array of shape (B, J, 2) or (B, 3, J).
        visibility: Optional visibility tensor (B, J).
        threshold: Relative torso ratio (< 1.0) or absolute pixels (>= 1.0).

    Returns:
        Tuple of (mean_pck, per_joint_pck).
    """
    device = preds.device
    gt_tensor = torch.as_tensor(gts, device=device, dtype=preds.dtype)

    # If gt is (B, 3, J) -> (coords, joints)
    if gt_tensor.dim() == 3 and gt_tensor.shape[1] == 3:
        if visibility is None:
            vis_tensor = (gt_tensor[:, 2, :] <= 1).float()
        else:
            vis_tensor = torch.as_tensor(visibility, device=device, dtype=torch.float32)
        gt_coords = gt_tensor[:, :2, :].permute(0, 2, 1)  # (B, J, 2)
    else:
        gt_coords = gt_tensor
        if visibility is not None:
            vis_tensor = torch.as_tensor(visibility, device=device, dtype=torch.float32)
        else:
            vis_tensor = torch.ones(preds.shape[:2], device=device, dtype=torch.float32)

    dist = torch.sqrt(torch.sum((preds - gt_coords) ** 2, dim=-1))  # (B, J)

    if threshold < 1.0:
        # Torso reference: midpoint of shoulders (8, 9) to midpoint of hips (2, 3)
        # Indices: 8: R_Shoulder, 9: L_Shoulder, 2: R_Hip, 3: L_Hip
        shoulder_mid = (gt_coords[:, 8, :] + gt_coords[:, 9, :]) / 2.0
        hip_mid = (gt_coords[:, 2, :] + gt_coords[:, 3, :]) / 2.0
        torso_dist = torch.sqrt(torch.sum((shoulder_mid - hip_mid) ** 2, dim=-1))
        torso_dist = torch.clamp(torso_dist, min=1e-6)
        effective_threshold = torso_dist.unsqueeze(-1) * threshold  # (B, 1)
    else:
        effective_threshold = torch.tensor(threshold, device=device)

    correct = (dist <= effective_threshold).float() * vis_tensor

    sum_vis = torch.clamp(torch.sum(vis_tensor, dim=0), min=1e-6)
    per_joint_pck = torch.sum(correct, dim=0) / sum_vis

    total_vis = torch.clamp(torch.sum(vis_tensor), min=1e-6)
    mean_pck = torch.sum(correct) / total_vis

    return mean_pck, per_joint_pck
