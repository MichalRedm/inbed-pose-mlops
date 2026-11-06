"""Unit tests for PCK and MPJPE evaluation metrics."""

import torch

from src.metrics.pose_metrics import compute_mpjpe, compute_pck


def test_mpjpe_perfect_match() -> None:
    """Verifies that identical predicted and ground truth keypoints yield zero MPJPE."""
    preds = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0]]], dtype=torch.float32
    )  # (1, 2, 2)
    gts = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]], dtype=torch.float32)

    mean_err, per_joint = compute_mpjpe(preds, gts)
    assert float(mean_err) == 0.0
    assert torch.all(per_joint == 0.0)


def test_mpjpe_known_distance() -> None:
    """Verifies MPJPE calculation with known geometric Euclidean offset."""
    # Joint 0 offset by (3, 4) -> distance = 5.0
    # Joint 1 offset by (0, 0) -> distance = 0.0
    preds = torch.tensor([[[3.0, 4.0], [10.0, 10.0]]], dtype=torch.float32)
    gts = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]], dtype=torch.float32)

    mean_err, per_joint = compute_mpjpe(preds, gts)
    assert abs(float(per_joint[0]) - 5.0) < 1e-5
    assert abs(float(per_joint[1]) - 0.0) < 1e-5
    assert abs(float(mean_err) - 2.5) < 1e-5


def test_pck_torso_relative() -> None:
    """Verifies torso-relative PCK@0.2 calculation."""
    # Create 1 sample with 14 joints
    coords = torch.zeros((1, 14, 2), dtype=torch.float32)

    # Set shoulders (8, 9) and hips (2, 3) to form a torso length of 100 pixels
    # Midpoint shoulders = (50, 100), Midpoint hips = (50, 0) -> torso_dist = 100
    # PCK@0.2 threshold = 0.2 * 100 = 20.0 pixels
    coords[0, 8] = torch.tensor([40.0, 100.0])
    coords[0, 9] = torch.tensor([60.0, 100.0])
    coords[0, 2] = torch.tensor([40.0, 0.0])
    coords[0, 3] = torch.tensor([60.0, 0.0])

    gts = coords.clone()
    preds = coords.clone()

    # Move joint 0 by 10 pixels (within 20px threshold -> correct)
    preds[0, 0] += torch.tensor([10.0, 0.0])
    # Move joint 1 by 30 pixels (outside 20px threshold -> incorrect)
    preds[0, 1] += torch.tensor([30.0, 0.0])

    mean_pck, per_joint_pck = compute_pck(preds, gts, threshold=0.2)

    assert float(per_joint_pck[0]) == 1.0  # within threshold
    assert float(per_joint_pck[1]) == 0.0  # outside threshold
    # Remaining 12 joints are identical (1.0), so 13 correct out of 14: 13 / 14 ~ 0.9285
    assert abs(float(mean_pck) - (13.0 / 14.0)) < 1e-3
