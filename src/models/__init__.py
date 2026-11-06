"""Model architectures for in-bed pose estimation."""

from src.models.base import BasePoseModel
from src.models.resnet_pose import ResNetPose

__all__ = ["BasePoseModel", "ResNetPose"]
