"""Loss functions for keypoint heatmap regression."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointHeatmapLoss(nn.Module):
    """Mean Squared Error (or Smooth L1) loss computed over multi-joint heatmaps.

    Supports optional joint-specific loss weighting.
    """

    def __init__(
        self,
        loss_type: Literal["mse", "smooth_l1"] = "mse",
        joint_weights: torch.Tensor | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type

        if joint_weights is not None:
            weights_tensor = torch.as_tensor(joint_weights, dtype=torch.float32)
            # Shape for broadcasting over (B, 14, H, W): (1, 14, 1, 1)
            self.register_buffer("weights", weights_tensor.view(1, -1, 1, 1))
        else:
            self.weights = None

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes loss between predicted and ground-truth heatmaps.

        Args:
            preds: Predicted heatmaps of shape (B, J, H, W).
            targets: Target Gaussian heatmaps of shape (B, J, H, W).

        Returns:
            Scalar loss tensor.
        """
        if self.loss_type == "mse":
            diff = (preds - targets) ** 2
        elif self.loss_type == "smooth_l1":
            diff = F.smooth_l1_loss(preds, targets, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.weights is not None:
            diff = diff * self.weights

        return diff.mean()
