"""Gaussian heatmap generation and peak decoding utilities."""

import numpy as np
import torch


def generate_gaussian_heatmaps(
    joints: torch.Tensor,
    heatmap_size: tuple[int, int] = (64, 64),
    image_size: tuple[int, int] = (256, 256),
    sigma: float = 2.0,
) -> torch.Tensor:
    """Generates 2D Gaussian heatmaps on PyTorch tensors directly on the target device.

    Args:
        joints: Tensor of shape (B, 3, 14) -> (coords, joints) with joints[:, :2, :] as (x, y)
                and joints[:, 2, :] as visibility.
        heatmap_size: Output resolution (H_out, W_out).
        image_size: Input image resolution (H_in, W_in).
        sigma: Standard deviation for Gaussian peaks.

    Returns:
        Tensor of shape (B, 14, H_out, W_out) with values in [0, 1].
    """
    b, _, j = joints.shape
    h_out, w_out = heatmap_size
    h_in, w_in = image_size
    device = joints.device

    # Scale joints from input image size to heatmap size
    scale_x = w_out / w_in
    scale_y = h_out / h_in

    mu_x = joints[:, 0, :] * scale_x  # (B, J)
    mu_y = joints[:, 1, :] * scale_y  # (B, J)
    visibility = joints[:, 2, :]  # (B, J)

    # Coordinate grids
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h_out, device=device, dtype=torch.float32),
        torch.arange(w_out, device=device, dtype=torch.float32),
        indexing="ij",
    )

    grid_x = grid_x.view(1, 1, h_out, w_out)
    grid_y = grid_y.view(1, 1, h_out, w_out)

    mu_x = mu_x.view(b, j, 1, 1)
    mu_y = mu_y.view(b, j, 1, 1)

    # 2D Gaussian
    dist_sq = (grid_x - mu_x) ** 2 + (grid_y - mu_y) ** 2
    heatmaps = torch.exp(-dist_sq / (2.0 * float(sigma) ** 2))

    # Mask out invisible or unannotated joints (visibility > 1 or coords (0,0))
    invalid_mask = (visibility > 1) | (
        (joints[:, 0, :] == 0) & (joints[:, 1, :] == 0)
    )  # (B, J)
    heatmaps = heatmaps.masked_fill(invalid_mask.view(b, j, 1, 1), 0.0)

    # Mask out joints outside boundaries
    out_of_bounds = (mu_x < 0) | (mu_y < 0) | (mu_x >= w_out) | (mu_y >= h_out)
    heatmaps = heatmaps.masked_fill(out_of_bounds, 0.0)

    return heatmaps


def decode_heatmaps(
    heatmaps: torch.Tensor | np.ndarray,
    image_size: tuple[int, int] = (256, 256),
    temperature: float = 10.0,
    soft_argmax: bool = False,
) -> torch.Tensor:
    """Decodes heatmaps (B, J, H, W) into 2D keypoint coordinates (B, J, 2).

    Args:
        heatmaps: Tensor of shape (B, J, H, W) containing predicted heatmaps.
        image_size: Target image size (H_in, W_in) to scale predicted coordinates.
        temperature: Softmax temperature when using soft-argmax.
        soft_argmax: If True, uses expected value; otherwise standard argmax.

    Returns:
        Tensor of shape (B, J, 2) representing (x, y) coordinates in image space.
    """
    if isinstance(heatmaps, np.ndarray):
        h_tensor = torch.from_numpy(heatmaps)
    else:
        h_tensor = heatmaps

    b, j, h, w = h_tensor.shape
    h_in, w_in = image_size
    device = h_tensor.device

    if soft_argmax:
        flat = h_tensor.view(b, j, -1)
        weights = torch.softmax(flat * temperature, dim=-1).view(b, j, h, w)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        coord_x = torch.sum(weights * grid_x.view(1, 1, h, w), dim=(-2, -1))
        coord_y = torch.sum(weights * grid_y.view(1, 1, h, w), dim=(-2, -1))
    else:
        flat = h_tensor.view(b, j, -1)
        max_idx = torch.argmax(flat, dim=-1)
        coord_y = (max_idx // w).to(torch.float32)
        coord_x = (max_idx % w).to(torch.float32)

    # Scale back to image coordinates
    scale_x = w_in / w
    scale_y = h_in / h

    coord_x = coord_x * scale_x
    coord_y = coord_y * scale_y

    return torch.stack([coord_x, coord_y], dim=-1)
