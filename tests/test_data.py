"""Unit tests for SLP dataset, augmentations, and DataModule."""

from pathlib import Path

import numpy as np
import torch

from src.data.augmentations import PoseAugmenter, apply_tensor_cutout
from src.data.datamodule import SLPDataModule
from src.data.dataset import SLPDataset
from src.utils.heatmaps import decode_heatmaps, generate_gaussian_heatmaps


def test_slp_dataset_loading(mock_slp_root: Path) -> None:
    """Tests basic indexing and tensor shape generation in SLPDataset."""
    dataset = SLPDataset(
        root=mock_slp_root,
        subjects=[1],
        modalities=["IR"],
        covers=["uncover"],
        split="train",
        image_size=(128, 128),
        heatmap_size=(32, 32),
        sigma=2.0,
    )

    assert len(dataset) == 3, f"Expected 3 frames, got {len(dataset)}"
    item = dataset[0]

    assert item["image"].shape == (1, 128, 128)
    assert item["target"] is not None
    assert item["target"].shape == (14, 32, 32)
    assert item["joints"] is not None
    assert item["joints"].shape == (3, 14)
    assert item["subject"] == 1
    assert item["cover"] == "uncover"


def test_slp_dataset_multimodal(mock_slp_root: Path) -> None:
    """Tests concatenation of multiple modalities (RGB + IR = 4 channels)."""
    dataset = SLPDataset(
        root=mock_slp_root,
        subjects=[1],
        modalities=["RGB", "IR"],
        covers=["uncover"],
        image_size=(64, 64),
        heatmap_size=(16, 16),
    )

    assert dataset.in_channels == 4
    item = dataset[0]
    assert item["image"].shape == (4, 64, 64)


def test_slp_dataset_unlabeled(mock_slp_root: Path) -> None:
    """Tests dataset mode where heatmaps are not generated (e.g. unlabeled stream)."""
    dataset = SLPDataset(
        root=mock_slp_root,
        subjects=[1],
        modalities=["IR"],
        covers=["cover1"],
        return_heatmaps=False,
    )

    item = dataset[0]
    assert item["target"] is None


def test_pose_augmenter_horizontal_flip() -> None:
    """Tests keypoint coordinate reflection and left/right joint swapping."""
    augmenter = PoseAugmenter(flip_prob=1.0, rotation_degrees=0.0, is_training=True)

    from PIL import Image

    dummy_img = Image.new("L", (100, 100), color=128)
    # Synthetic joints: joint 0 (R_Ankle) at x=20, joint 5 (L_Ankle) at x=80
    joints = np.zeros((3, 14), dtype=np.float32)
    joints[0, 0] = 20.0
    joints[1, 0] = 50.0
    joints[2, 0] = 1.0

    joints[0, 5] = 80.0
    joints[1, 5] = 50.0
    joints[2, 5] = 1.0

    _, flipped_joints = augmenter(dummy_img, joints, image_size=(100, 100))
    assert flipped_joints is not None

    # After horizontal flip of 100px width:
    # Joint 0 becomes old joint 5 flipped: (100 - 1 - 80) = 19
    # Joint 5 becomes old joint 0 flipped: (100 - 1 - 20) = 79
    np.testing.assert_almost_equal(flipped_joints[0, 0], 19.0, decimal=1)
    np.testing.assert_almost_equal(flipped_joints[0, 5], 79.0, decimal=1)


def test_tensor_cutout() -> None:
    """Tests random erasing of rectangular regions."""
    torch.manual_seed(42)
    tensor = torch.ones((1, 64, 64))
    cut = apply_tensor_cutout(tensor, scale_ratio=0.5, probability=1.0)

    # Some elements should now be zeroed
    assert torch.sum(cut == 0.0) > 0
    assert cut.shape == tensor.shape


def test_gaussian_heatmaps_and_decode() -> None:
    """Tests that generated Gaussian heatmaps decode accurately back to coordinates."""
    # Batch of 1 sample with 14 joints
    joints = torch.zeros((1, 3, 14))
    # Place joint 0 at (32.0, 48.0) in 128x128 space
    joints[0, 0, 0] = 32.0
    joints[0, 1, 0] = 48.0
    joints[0, 2, 0] = 1.0  # visible

    heatmaps = generate_gaussian_heatmaps(
        joints,
        heatmap_size=(32, 32),
        image_size=(128, 128),
        sigma=1.5,
    )
    assert heatmaps.shape == (1, 14, 32, 32)
    assert heatmaps[0, 0].max() > 0.9

    decoded = decode_heatmaps(heatmaps, image_size=(128, 128), soft_argmax=False)
    assert decoded.shape == (1, 14, 2)

    # Decoded coordinates should be close to (32.0, 48.0)
    pred_x, pred_y = decoded[0, 0, 0].item(), decoded[0, 0, 1].item()
    assert abs(pred_x - 32.0) <= 4.0
    assert abs(pred_y - 48.0) <= 4.0


def test_slp_datamodule_supervised(mock_slp_root: Path) -> None:
    """Tests SLPDataModule in standard supervised configuration."""
    dm = SLPDataModule(
        data_root=mock_slp_root,
        batch_size=2,
        num_workers=0,
        train_subjects=[1],
        val_subjects=[31],
        test_subjects=[41],
        semi_supervised=False,
    )

    dm.prepare_data()
    dm.setup("fit")

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert batch is not None
    assert "image" in batch
    assert "target" in batch
    assert batch["image"].shape[0] == 2
    assert batch["target"].shape == (2, 14, 64, 64)


def test_slp_datamodule_semi_supervised(mock_slp_root: Path) -> None:
    """Tests SLPDataModule in semi-supervised mode yielding labeled and unlabeled streams."""
    dm = SLPDataModule(
        data_root=mock_slp_root,
        batch_size=2,
        num_workers=0,
        train_subjects=[1],
        val_subjects=[31],
        test_subjects=[41],
        semi_supervised=True,
    )

    dm.setup("fit")
    loader = dm.train_dataloader()
    raw_batch = next(iter(loader))
    batch = raw_batch[0] if isinstance(raw_batch, tuple) else raw_batch

    assert isinstance(batch, dict)
    assert "labeled" in batch
    assert "unlabeled" in batch
    assert batch["labeled"]["target"] is not None
