"""Augmentations for in-bed multimodal pose estimation."""

import math
import random

import numpy as np
import torch
import torchvision.transforms.v2.functional as F_v2
from PIL import Image

# LSP format joint flip symmetry mapping:
# 0: R_Ankle  <-> 5: L_Ankle
# 1: R_Knee   <-> 4: L_Knee
# 2: R_Hip    <-> 3: L_Hip
# 6: R_Wrist  <-> 11: L_Wrist
# 7: R_Elbow  <-> 10: L_Elbow
# 8: R_Shoulder <-> 9: L_Shoulder
# 12: Thorax (unchanged)
# 13: Head (unchanged)
LSP_FLIP_INDICES: list[int] = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]


def apply_tensor_cutout(
    img_tensor: torch.Tensor,
    scale_ratio: float = 0.35,
    probability: float = 0.5,
) -> torch.Tensor:
    """Applies random rectangular patch erasing (cutout) to an image tensor.

    Useful for consistency regularization during semi-supervised student training.

    Args:
        img_tensor: Tensor of shape (C, H, W) or (B, C, H, W).
        scale_ratio: Max fraction of height and width to erase.
        probability: Probability of applying cutout.

    Returns:
        Tensor with erased rectangular patch.
    """
    if random.random() > probability:
        return img_tensor

    is_batched = img_tensor.dim() == 4
    if not is_batched:
        img_tensor = img_tensor.unsqueeze(0)

    b, c, h, w = img_tensor.shape
    out = img_tensor.clone()

    cut_h = max(1, int(h * scale_ratio))
    cut_w = max(1, int(w * scale_ratio))

    for i in range(b):
        cy = random.randint(0, max(0, h - cut_h))
        cx = random.randint(0, max(0, w - cut_w))
        out[i, :, cy : cy + cut_h, cx : cx + cut_w] = 0.0

    if not is_batched:
        out = out.squeeze(0)

    return out


class PoseAugmenter:
    """Combines geometric transforms (flip, rotation, scale) for image and keypoints."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotation_degrees: float = 20.0,
        scale_range: tuple[float, float] = (0.85, 1.15),
        is_training: bool = True,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.is_training = is_training

    def __call__(
        self,
        image: Image.Image,
        joints: np.ndarray | None = None,
        image_size: tuple[int, int] = (256, 256),
    ) -> tuple[torch.Tensor, np.ndarray | None]:
        """Applies spatial transformations to image and 14 keypoint coordinates.

        Args:
            image: Input PIL Image.
            joints: Optional numpy array of shape (3, 14) with [x, y, visibility].
            image_size: Target (Height, Width).

        Returns:
            Tuple of transformed image tensor (C, H, W) and updated joints (3, 14).
        """
        w_orig, h_orig = image.size
        h_target, w_target = image_size

        # Resize image to target resolution
        img = image.resize((w_target, h_target), Image.Resampling.BILINEAR)

        # Scale joints according to resolution change
        kpts: np.ndarray | None = None
        if joints is not None:
            kpts = joints.copy()
            kpts[0, :] = kpts[0, :] * (w_target / w_orig)
            kpts[1, :] = kpts[1, :] * (h_target / h_orig)

        if not self.is_training:
            img_tensor = F_v2.to_image(img)
            img_tensor = F_v2.to_dtype(img_tensor, torch.float32, scale=True)
            return img_tensor, kpts

        # 1. Random Horizontal Flip
        if random.random() < self.flip_prob:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if kpts is not None:
                # Flip x coordinate
                kpts[0, :] = (w_target - 1) - kpts[0, :]
                # Swap symmetric left/right joints
                kpts = kpts[:, LSP_FLIP_INDICES]

        # 2. Random Rotation
        if self.rotation_degrees > 0 and random.random() < 0.5:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR)
            if kpts is not None:
                # Rotate keypoints around center
                rad = math.radians(-angle)
                cx, cy = w_target / 2.0, h_target / 2.0
                cos_a, sin_a = math.cos(rad), math.sin(rad)

                x = kpts[0, :] - cx
                y = kpts[1, :] - cy

                kpts[0, :] = x * cos_a - y * sin_a + cx
                kpts[1, :] = x * sin_a + y * cos_a + cy

        img_tensor = F_v2.to_image(img)
        img_tensor = F_v2.to_dtype(img_tensor, torch.float32, scale=True)

        return img_tensor, kpts
