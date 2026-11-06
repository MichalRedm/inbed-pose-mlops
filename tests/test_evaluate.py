"""Unit tests for standalone evaluation script and metrics aggregation."""

import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate import evaluate_model, print_evaluation_summary
from src.metrics.pose_metrics import LSP_JOINT_NAMES
from src.models.resnet_pose import ResNetPose
from src.training.lightning_module import InBedPoseLightningModule


class DictDataset(TensorDataset):
    """Wraps tensors into dictionary format matching SLP collate output."""

    def __init__(self, images: torch.Tensor, joints: torch.Tensor) -> None:
        super().__init__(images, joints)
        self.images = images
        self.joints = joints

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
            "joints": self.joints[idx],
        }

    def __len__(self) -> int:
        return int(self.images.size(0))


def test_evaluate_model_and_summary() -> None:
    """Verifies evaluate_model produces expected metric dictionary structure."""
    model = ResNetPose(backbone="resnet18", in_channels=1, num_joints=14)
    lit_module = InBedPoseLightningModule(model=model, mode="supervised")

    # Synthetic dataset with 4 samples
    images = torch.randn(4, 1, 256, 256)
    joints = torch.full((4, 3, 14), 100.0)
    joints[:, 2, :] = 1.0  # visible

    dataset = DictDataset(images, joints)
    dataloader = DataLoader(dataset, batch_size=2)  # type: ignore[arg-type]

    results = evaluate_model(lit_module, dataloader, device=torch.device("cpu"))

    assert "pck@0.2" in results
    assert "pck@0.1" in results
    assert "mpjpe" in results
    assert results["num_samples"] == 4
    assert len(results["per_joint_pck@0.2"]) == 14
    for name in LSP_JOINT_NAMES:
        assert name in results["per_joint_pck@0.2"]
        assert name in results["per_joint_mpjpe"]

    # Verify summary print doesn't raise exception
    print_evaluation_summary(results, cover_label="TEST - UNCOVER")


def test_evaluate_cli_smoke(mock_slp_root: Path, tmp_path: Path) -> None:
    """Tests evaluate.py CLI invocation against a saved checkpoint."""
    model = ResNetPose(backbone="resnet18", in_channels=1, num_joints=14)
    lit_module = InBedPoseLightningModule(model=model, mode="supervised")

    ckpt_path = tmp_path / "model.ckpt"
    import pytorch_lightning as pl

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        max_epochs=1,
    )
    trainer.strategy.connect(lit_module)
    trainer.save_checkpoint(str(ckpt_path))

    out_json = tmp_path / "metrics.json"
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--checkpoint",
        str(ckpt_path),
        "--config",
        "configs/baseline.yaml",
        "--data_dir",
        str(mock_slp_root),
        "--split",
        "test",
        "--covers",
        "uncover",
        "--batch_size",
        "2",
        "--device",
        "cpu",
        "--output_json",
        str(out_json),
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res.returncode == 0, f"Evaluation CLI failed:\n{res.stderr}\n{res.stdout}"
    assert out_json.exists()
