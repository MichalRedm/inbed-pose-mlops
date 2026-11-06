"""Abstract base class for pose estimation architectures."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BasePoseModel(nn.Module, ABC):
    """Abstract interface for human pose estimation models."""

    @property
    @abstractmethod
    def output_type(self) -> str:
        """Returns the type of model prediction (e.g. 'heatmap' or 'coordinates')."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Args:
            x: Input image tensor of shape (B, in_channels, H, W).

        Returns:
            Output heatmaps of shape (B, num_joints, H_out, W_out).
        """
        pass
