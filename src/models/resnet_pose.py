"""ResNet backbone with deconvolutional head for 2D pose estimation.

Follows the SimpleBaseline pose architecture (Xiao et al., ECCV 2018).
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from src.models.base import BasePoseModel


class ResNetPose(BasePoseModel):
    """ResNet-based heatmap regression model for 14-keypoint pose estimation."""

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        num_joints: int = 14,
        in_channels: int = 1,
        pretrained: bool = False,
        deconv_channels: tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_joints = num_joints
        self.in_channels = in_channels

        # 1. Instantiate torchvision ResNet backbone
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            in_features = 512
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base = resnet34(weights=weights)
            in_features = 512
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = resnet50(weights=weights)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Adapt first convolution if in_channels != 3
        if in_channels != 3:
            orig_conv = base.conv1
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )
            if pretrained and orig_conv.weight is not None:
                with torch.no_grad():
                    if in_channels == 1:
                        # Grayscale: average RGB channel weights
                        new_conv.weight.copy_(
                            orig_conv.weight.mean(dim=1, keepdim=True)
                        )
                    elif in_channels == 4:
                        # RGB + IR: copy RGB and initialize IR channel with mean
                        new_conv.weight[:, :3, :, :].copy_(orig_conv.weight)
                        new_conv.weight[:, 3:, :, :].copy_(
                            orig_conv.weight.mean(dim=1, keepdim=True)
                        )
            base.conv1 = new_conv

        # Backbone layers up to layer4
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # 2. Deconvolutional Head (3x upsampling by 2 -> 8x total)
        deconv_layers: list[nn.Module] = []
        curr_in = in_features
        for out_c in deconv_channels:
            deconv_layers.extend(
                [
                    nn.ConvTranspose2d(
                        curr_in,
                        out_c,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            )
            curr_in = out_c

        self.deconv = nn.Sequential(*deconv_layers)

        # 3. Final 1x1 Conv predicting keypoint heatmaps
        self.final_layer = nn.Conv2d(
            curr_in,
            num_joints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes deconvolution and final prediction layers."""
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.final_layer.weight, std=0.001)
        if self.final_layer.bias is not None:
            nn.init.constant_(self.final_layer.bias, 0)

    @property
    def output_type(self) -> str:
        return "heatmap"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor (B, in_channels, H, W).

        Returns:
            Predicted heatmaps (B, num_joints, H/4, W/4).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv(x)
        out: torch.Tensor = self.final_layer(x)
        return out
