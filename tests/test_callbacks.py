"""Unit tests for training callbacks and visual logging."""

from pathlib import Path
from unittest.mock import MagicMock

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.models.resnet_pose import ResNetPose
from src.training.callbacks import LogVisualPredictionsCallback, get_default_callbacks
from src.training.lightning_module import InBedPoseLightningModule


def test_get_default_callbacks(tmp_path: Path) -> None:
    """Verifies that default callbacks list contains all required monitors."""
    callbacks = get_default_callbacks(
        monitor="val_pck",
        mode="max",
        save_top_k=2,
        patience=5,
        log_predictions_every_n_epochs=2,
        save_dir=str(tmp_path / "predictions"),
    )

    types = [type(c) for c in callbacks]
    assert ModelCheckpoint in types
    assert EarlyStopping in types
    assert LearningRateMonitor in types
    assert LogVisualPredictionsCallback in types

    checkpoint_cb = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    assert checkpoint_cb.monitor == "val_pck"
    assert checkpoint_cb.mode == "max"
    assert checkpoint_cb.save_top_k == 2


def test_log_visual_predictions_callback(tmp_path: Path) -> None:
    """Verifies that LogVisualPredictionsCallback generates and saves visual figures."""
    save_dir = tmp_path / "visual_logs"
    callback = LogVisualPredictionsCallback(
        num_samples=2,
        log_every_n_epochs=1,
        save_dir=str(save_dir),
    )

    model = ResNetPose(backbone="resnet18", in_channels=1, num_joints=14)
    pl_module = InBedPoseLightningModule(model=model, mode="supervised")

    batch = {
        "image": torch.randn(2, 1, 256, 256),
        "target": torch.zeros(2, 14, 64, 64),
        "joints": torch.zeros(2, 3, 14),
    }

    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.current_epoch = 0
    mock_trainer.logger = None

    # 1. Batch start caches the images
    callback.on_validation_batch_start(
        trainer=mock_trainer,
        pl_module=pl_module,
        batch=batch,
        batch_idx=0,
    )
    assert callback._cached_batch is not None

    # 2. Epoch end renders and exports the plot to disk
    callback.on_validation_epoch_end(trainer=mock_trainer, pl_module=pl_module)

    # Verify cache cleared and image written to disk
    assert callback._cached_batch is None
    saved_files = list(save_dir.glob("*.png"))
    assert len(saved_files) == 1
    assert saved_files[0].name == "epoch_000.png"
