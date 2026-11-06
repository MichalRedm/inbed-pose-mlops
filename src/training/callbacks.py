"""PyTorch Lightning callbacks for model checkpointing and visual monitoring."""

import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.metrics.pose_metrics import LSP_SKELETON_PAIRS
from src.utils.heatmaps import decode_heatmaps

# Ensure headless matplotlib backend
matplotlib.use("Agg")


class LogVisualPredictionsCallback(Callback):
    """Callback to render and log pose prediction overlays to MLflow and disk."""

    def __init__(
        self,
        num_samples: int = 4,
        log_every_n_epochs: int = 1,
        save_dir: str | None = None,
    ) -> None:
        """Initializes the visual prediction callback.

        Args:
            num_samples: Number of validation samples to visualize in grid.
            log_every_n_epochs: Epoch frequency for logging prediction figures.
            save_dir: Optional local directory to save figures as PNG files.
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.save_dir = save_dir
        self._cached_batch: dict[str, Any] | None = None

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Caches the first validation batch of the epoch for visualization."""
        if (
            batch_idx == 0
            and (trainer.current_epoch + 1) % self.log_every_n_epochs == 0
        ):
            raw_batch = batch[0] if isinstance(batch, tuple) else batch
            if isinstance(raw_batch, dict) and "image" in raw_batch:
                images = raw_batch["image"][: self.num_samples].detach().cpu()
                joints = (
                    raw_batch["joints"][: self.num_samples].detach().cpu()
                    if "joints" in raw_batch
                    else None
                )
                self._cached_batch = {"image": images, "joints": joints}

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Generates, visualizes, and logs skeleton prediction figures."""
        if self._cached_batch is None:
            return

        images = self._cached_batch["image"].to(pl_module.device)
        gt_joints = self._cached_batch["joints"]

        pl_module.eval()
        with torch.no_grad():
            heatmaps = pl_module(images)

        image_size = getattr(pl_module, "image_size", (256, 256))
        pred_coords = (
            decode_heatmaps(heatmaps, image_size=image_size).detach().cpu().numpy()
        )
        images_cpu = images.detach().cpu().numpy()

        n = min(self.num_samples, images_cpu.shape[0])
        if n == 0:
            self._cached_batch = None
            return

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

        for i in range(n):
            ax = axes[0, i]
            img = images_cpu[i]

            # Render image background
            if img.shape[0] == 1:
                ax.imshow(np.clip(img[0], 0.0, 1.0), cmap="gray")
            elif img.shape[0] >= 3:
                rgb = np.clip(np.transpose(img[:3], (1, 2, 0)), 0.0, 1.0)
                ax.imshow(rgb)

            # Render Ground Truth skeleton (Green)
            if gt_joints is not None:
                gt = gt_joints[i].numpy()  # (3, 14)
                vis = gt[2, :] <= 1.0
                ax.scatter(
                    gt[0, vis],
                    gt[1, vis],
                    c="lime",
                    s=20,
                    edgecolors="black",
                    zorder=3,
                    label="GT" if i == 0 else "",
                )
                for j1, j2 in LSP_SKELETON_PAIRS:
                    if vis[j1] and vis[j2]:
                        ax.plot(
                            [gt[0, j1], gt[0, j2]],
                            [gt[1, j1], gt[1, j2]],
                            c="lime",
                            linewidth=1.8,
                            zorder=2,
                        )

            # Render Predicted skeleton (Red)
            pred = pred_coords[i]  # (14, 2)
            ax.scatter(
                pred[:, 0],
                pred[:, 1],
                c="red",
                s=20,
                edgecolors="white",
                zorder=5,
                label="Pred" if i == 0 else "",
            )
            for j1, j2 in LSP_SKELETON_PAIRS:
                ax.plot(
                    [pred[j1, 0], pred[j2, 0]],
                    [pred[j1, 1], pred[j2, 1]],
                    c="red",
                    linewidth=1.8,
                    linestyle="--",
                    zorder=4,
                )

            ax.set_title(f"Sample {i + 1}", fontsize=10)
            ax.axis("off")

        if n > 0:
            axes[0, 0].legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"Epoch {trainer.current_epoch:02d} Predictions (Green: GT, Red: Pred)",
            fontsize=12,
        )
        fig.tight_layout()

        # Log figure to MLflow logger if present
        if trainer.logger is not None:
            if hasattr(trainer.logger, "experiment") and hasattr(
                trainer.logger.experiment, "log_figure"
            ):
                run_id = getattr(trainer.logger, "run_id", None)
                if run_id:
                    try:
                        trainer.logger.experiment.log_figure(
                            run_id=run_id,
                            figure=fig,
                            artifact_file=f"predictions/epoch_{trainer.current_epoch:03d}.png",
                        )
                    except Exception:
                        pass

        # Save to disk if save_dir specified
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            fig_path = os.path.join(
                self.save_dir, f"epoch_{trainer.current_epoch:03d}.png"
            )
            fig.savefig(fig_path, bbox_inches="tight")

        plt.close(fig)
        self._cached_batch = None


def get_default_callbacks(
    monitor: str = "val_pck",
    mode: str = "max",
    save_top_k: int = 1,
    patience: int = 10,
    log_predictions_every_n_epochs: int = 1,
    save_dir: str | None = None,
) -> list[Callback]:
    """Constructs standard training callbacks for checkpointing and monitoring.

    Args:
        monitor: Metric to monitor for checkpointing and early stopping.
        mode: Direction of metric optimization ('min' or 'max').
        save_top_k: Number of best checkpoints to retain.
        patience: Epoch patience before early stopping triggers.
        log_predictions_every_n_epochs: Frequency of visual prediction logging.
        save_dir: Directory to save visual prediction plots.

    Returns:
        List of PyTorch Lightning Callback instances.
    """
    callbacks: list[Callback] = [
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            filename=f"best-checkpoint-{{{monitor}:.4f}}",
            save_last=True,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        LogVisualPredictionsCallback(
            num_samples=4,
            log_every_n_epochs=log_predictions_every_n_epochs,
            save_dir=save_dir,
        ),
    ]
    return callbacks
