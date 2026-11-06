"""PyTorch LightningModule for in-bed human pose estimation.

Supports both standard supervised baseline training and SOTA Teacher-Student
semi-supervised self-training utilizing unlabeled covered examples.
"""

import copy
from typing import Any, Literal, cast

import pytorch_lightning as pl
import torch
import torch.optim as optim

from src.data.augmentations import apply_tensor_cutout
from src.metrics.pose_metrics import compute_mpjpe, compute_pck
from src.models.base import BasePoseModel
from src.training.losses import JointHeatmapLoss
from src.utils.heatmaps import decode_heatmaps


class InBedPoseLightningModule(pl.LightningModule):
    """Encapsulates pose estimation science, optimization, and metric tracking."""

    teacher: BasePoseModel | None

    def __init__(
        self,
        model: BasePoseModel,
        mode: Literal["supervised", "semi_supervised"] = "supervised",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_scheduler: Literal["cosine", "plateau", "none"] = "cosine",
        loss_type: Literal["mse", "smooth_l1"] = "mse",
        optimizer_name: Literal["adam", "adamw"] = "adamw",
        ema_alpha: float = 0.999,
        confidence_threshold: float = 0.35,
        lambda_unlabeled: float = 1.0,
        cutout_prob: float = 0.5,
        cutout_scale: float = 0.35,
        image_size: tuple[int, int] = (256, 256),
        heatmap_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.mode = mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.optimizer_name = optimizer_name
        self.ema_alpha = ema_alpha
        self.confidence_threshold = confidence_threshold
        self.lambda_unlabeled = lambda_unlabeled
        self.cutout_prob = cutout_prob
        self.cutout_scale = cutout_scale
        self.image_size = image_size
        self.heatmap_size = heatmap_size

        self.criterion = JointHeatmapLoss(loss_type=loss_type)

        # Set up EMA Teacher if semi-supervised
        if self.mode == "semi_supervised":
            self.teacher = copy.deepcopy(model)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()
        else:
            self.teacher = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student model."""
        return cast(torch.Tensor, self.model(x))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Computes training step loss."""
        if self.mode == "supervised":
            images = batch["image"]
            targets = batch["target"]

            preds = self.model(images)
            loss = self.criterion(preds, targets)

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return cast(torch.Tensor, loss)

        # Semi-supervised Teacher-Student step
        # CombinedLoader can return (batch_dict, batch_idx, dataloader_idx) in Lightning 2.x
        if isinstance(batch, tuple):
            batch_data = batch[0]
        else:
            batch_data = batch

        batch_l = batch_data.get("labeled")
        batch_u = batch_data.get("unlabeled")

        if batch_l is None or batch_u is None:
            raise ValueError(
                "Semi-supervised mode expects both 'labeled' and 'unlabeled' batches."
            )

        # 1. Supervised loss on labeled data
        img_l = batch_l["image"]
        target_l = batch_l["target"]
        pred_l = self.model(img_l)
        loss_labeled = self.criterion(pred_l, target_l)

        # 2. Unlabeled Teacher-Student consistency loss
        img_u = batch_u["image"]
        b, _, _, _ = img_u.shape

        assert self.teacher is not None, (
            "Teacher model must exist for semi-supervised mode"
        )

        with torch.no_grad():
            self.teacher.eval()
            pred_teacher = self.teacher(img_u)  # (B, 14, H_out, W_out)

        # Confidence gating per joint: peak intensity in predicted heatmap
        flat_teacher = pred_teacher.view(b, 14, -1)
        conf, _ = flat_teacher.max(dim=-1)  # (B, 14)
        mask = (conf >= self.confidence_threshold).float()  # (B, 14)

        # Student receives strong perturbation view (cutout)
        img_u_strong = apply_tensor_cutout(
            img_u, scale_ratio=self.cutout_scale, probability=self.cutout_prob
        )
        pred_student = self.model(img_u_strong)

        # Masked consistency loss
        diff = (pred_student - pred_teacher.detach()) ** 2
        masked_diff = diff * mask.view(b, 14, 1, 1)
        denom = torch.clamp(
            mask.sum() * self.heatmap_size[0] * self.heatmap_size[1], min=1.0
        )
        loss_unlabeled = masked_diff.sum() / denom

        total_loss = loss_labeled + self.lambda_unlabeled * loss_unlabeled

        # 3. Update EMA Teacher weights
        with torch.no_grad():
            for s_param, t_param in zip(
                self.model.parameters(), self.teacher.parameters(), strict=True
            ):
                t_param.data.mul_(self.ema_alpha).add_(
                    s_param.data, alpha=1.0 - self.ema_alpha
                )

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_labeled", loss_labeled, on_step=False, on_epoch=True)
        self.log("train_loss_unlabeled", loss_unlabeled, on_step=False, on_epoch=True)
        self.log("teacher_conf", conf.mean(), on_step=False, on_epoch=True)

        return cast(torch.Tensor, total_loss)

    def validation_step(
        self, batch: dict[str, Any] | None, batch_idx: int
    ) -> dict[str, torch.Tensor] | None:
        """Validation evaluation step."""
        if batch is None:
            return None

        images = batch["image"]
        targets = batch["target"]
        joints = batch["joints"]

        preds = self.model(images)
        val_loss = self.criterion(preds, targets)

        # Decode heatmaps to 2D image coordinates
        pred_coords = decode_heatmaps(preds, image_size=self.image_size)

        # Compute PCK@0.2 and MPJPE
        val_pck, _ = compute_pck(pred_coords, joints, threshold=0.2)
        val_mpjpe, _ = compute_mpjpe(pred_coords, joints)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pck", val_pck, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mpjpe", val_mpjpe, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": val_loss, "val_pck": val_pck, "val_mpjpe": val_mpjpe}

    def test_step(
        self, batch: dict[str, Any] | None, batch_idx: int
    ) -> dict[str, torch.Tensor] | None:
        """Test evaluation step."""
        if batch is None:
            return None

        images = batch["image"]
        targets = batch["target"]
        joints = batch["joints"]

        preds = self.model(images)
        test_loss = self.criterion(preds, targets)

        pred_coords = decode_heatmaps(preds, image_size=self.image_size)
        test_pck, _ = compute_pck(pred_coords, joints, threshold=0.2)
        test_mpjpe, _ = compute_mpjpe(pred_coords, joints)

        self.log("test_loss", test_loss, on_epoch=True)
        self.log("test_pck", test_pck, on_epoch=True)
        self.log("test_mpjpe", test_mpjpe, on_epoch=True)

        return {"test_loss": test_loss, "test_pck": test_pck, "test_mpjpe": test_mpjpe}

    def configure_optimizers(self) -> Any:
        """Constructs optimizer and learning rate scheduler."""
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.optimizer_name == "adam":
            optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs or 30, eta_min=1e-6
            )
            return [optimizer], [scheduler]
        elif self.lr_scheduler == "plateau":
            scheduler_dict = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3
                ),
                "monitor": "val_pck",
            }
            return [optimizer], [scheduler_dict]

        return optimizer
