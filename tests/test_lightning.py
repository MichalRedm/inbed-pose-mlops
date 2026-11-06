"""Integration tests for PyTorch Lightning Module training execution."""

from pathlib import Path

import pytorch_lightning as pl

from src.data.datamodule import SLPDataModule
from src.models.resnet_pose import ResNetPose
from src.training.lightning_module import InBedPoseLightningModule


def test_lightning_module_supervised_fast_dev_run(mock_slp_root: Path) -> None:
    """Verifies that supervised training step, validation step, and optimizer configure cleanly."""
    dm = SLPDataModule(
        data_root=mock_slp_root,
        batch_size=2,
        num_workers=0,
        train_subjects=[1],
        val_subjects=[31],
        test_subjects=[41],
        image_size=(64, 64),
        heatmap_size=(16, 16),
        semi_supervised=False,
    )

    model = ResNetPose(
        backbone="resnet18",
        num_joints=14,
        in_channels=1,
        pretrained=False,
        deconv_channels=(64, 64, 64),
    )

    module = InBedPoseLightningModule(
        model=model,
        mode="supervised",
        lr=1e-3,
        image_size=(64, 64),
        heatmap_size=(16, 16),
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm)


def test_lightning_module_semi_supervised_fast_dev_run(mock_slp_root: Path) -> None:
    """Verifies that semi-supervised Teacher-Student self-training executes without error."""
    dm = SLPDataModule(
        data_root=mock_slp_root,
        batch_size=2,
        num_workers=0,
        train_subjects=[1],
        val_subjects=[31],
        test_subjects=[41],
        image_size=(64, 64),
        heatmap_size=(16, 16),
        semi_supervised=True,
    )

    model = ResNetPose(
        backbone="resnet18",
        num_joints=14,
        in_channels=1,
        pretrained=False,
        deconv_channels=(64, 64, 64),
    )

    module = InBedPoseLightningModule(
        model=model,
        mode="semi_supervised",
        lr=1e-3,
        image_size=(64, 64),
        heatmap_size=(16, 16),
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=dm)
