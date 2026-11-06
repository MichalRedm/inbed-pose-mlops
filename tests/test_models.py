"""Unit tests for pose estimation model architectures."""

import torch

from src.models.resnet_pose import ResNetPose


def test_resnet_pose_forward_grayscale() -> None:
    """Tests ResNetPose forward pass with single-channel IR input."""
    model = ResNetPose(
        backbone="resnet18", num_joints=14, in_channels=1, pretrained=False
    )
    assert model.output_type == "heatmap"

    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    # Output should be (B, 14, H/4, W/4) = (2, 14, 32, 32)
    assert out.shape == (2, 14, 32, 32)


def test_resnet_pose_forward_multimodal() -> None:
    """Tests ResNetPose forward pass with 4-channel (RGB + IR) input."""
    model = ResNetPose(
        backbone="resnet18", num_joints=14, in_channels=4, pretrained=False
    )

    x = torch.randn(2, 4, 256, 256)
    out = model(x)
    assert out.shape == (2, 14, 64, 64)


def test_resnet_pose_backbone_selection() -> None:
    """Tests initialization of different ResNet backbones."""
    m18 = ResNetPose(backbone="resnet18", in_channels=1, pretrained=False)
    m34 = ResNetPose(backbone="resnet34", in_channels=1, pretrained=False)

    params18 = sum(p.numel() for p in m18.parameters())
    params34 = sum(p.numel() for p in m34.parameters())

    assert params34 > params18
    assert params18 < 16_000_000  # ResNet18 + deconv head is ~15.3M parameters
