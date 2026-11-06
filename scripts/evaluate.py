"""Evaluation script for In-Bed Pose Estimation models on SLP test splits."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path for direct CLI execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SLPDataset, collate_pose
from src.metrics.pose_metrics import LSP_JOINT_NAMES, compute_mpjpe, compute_pck
from src.models.resnet_pose import ResNetPose
from src.training.lightning_module import InBedPoseLightningModule
from src.utils.heatmaps import decode_heatmaps


def evaluate_model(
    model: InBedPoseLightningModule,
    dataloader: DataLoader[Any],
    device: torch.device,
) -> dict[str, Any]:
    """Runs evaluation over a DataLoader and computes pose estimation metrics.

    Args:
        model: InBedPoseLightningModule in eval mode.
        dataloader: DataLoader yielding SLP batch dictionaries.
        device: PyTorch device to execute evaluation on.

    Returns:
        Dictionary of computed overall and per-joint metrics.
    """
    model.eval()
    model.to(device)

    all_preds: list[torch.Tensor] = []
    all_gts: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            joints = batch["joints"]  # (B, 3, 14)

            heatmaps = model(images)
            pred_coords = decode_heatmaps(heatmaps, image_size=model.image_size)

            all_preds.append(pred_coords.cpu())
            all_gts.append(joints.cpu())

    preds_cat = torch.cat(all_preds, dim=0)  # (N, 14, 2)
    gts_cat = torch.cat(all_gts, dim=0)  # (N, 3, 14)

    # Compute PCK at thresholds 0.2, 0.1, 0.05
    pck_02, per_joint_pck_02 = compute_pck(preds_cat, gts_cat, threshold=0.2)
    pck_01, per_joint_pck_01 = compute_pck(preds_cat, gts_cat, threshold=0.1)
    pck_005, _ = compute_pck(preds_cat, gts_cat, threshold=0.05)

    # Compute MPJPE
    mpjpe, per_joint_mpjpe = compute_mpjpe(preds_cat, gts_cat)

    results: dict[str, Any] = {
        "pck@0.2": float(pck_02.item()),
        "pck@0.1": float(pck_01.item()),
        "pck@0.05": float(pck_005.item()),
        "mpjpe": float(mpjpe.item()),
        "per_joint_pck@0.2": {
            name: float(val.item())
            for name, val in zip(LSP_JOINT_NAMES, per_joint_pck_02, strict=True)
        },
        "per_joint_mpjpe": {
            name: float(val.item())
            for name, val in zip(LSP_JOINT_NAMES, per_joint_mpjpe, strict=True)
        },
        "num_samples": int(preds_cat.shape[0]),
    }
    return results


def print_evaluation_summary(results: dict[str, Any], cover_label: str) -> None:
    """Prints a formatted evaluation metrics table to stdout.

    Args:
        results: Dictionary returned by evaluate_model.
        cover_label: Description of the evaluated cover condition.
    """
    print("\n" + "=" * 65)
    print(f"SLP POSE ESTIMATION EVALUATION RESULTS ({cover_label})")
    print("=" * 65)
    print(f"Evaluated Samples: {results['num_samples']}")
    print(f"Overall PCK@0.20 : {results['pck@0.2'] * 100:.2f}%")
    print(f"Overall PCK@0.10 : {results['pck@0.1'] * 100:.2f}%")
    print(f"Overall PCK@0.05 : {results['pck@0.05'] * 100:.2f}%")
    print(f"Overall MPJPE    : {results['mpjpe']:.2f} px")
    print("-" * 65)
    print(f"{'Joint':<15} | {'PCK@0.20 (%)':<15} | {'MPJPE (px)':<15}")
    print("-" * 65)
    for name in LSP_JOINT_NAMES:
        pck_val = results["per_joint_pck@0.2"].get(name, 0.0) * 100
        mpjpe_val = results["per_joint_mpjpe"].get(name, 0.0)
        print(f"{name:<15} | {pck_val:<15.2f} | {mpjpe_val:<15.2f}")
    print("=" * 65 + "\n")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate In-Bed Pose Estimation Model from Checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch Lightning model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to SLP dataset root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--covers",
        nargs="+",
        default=None,
        help="Cover conditions to evaluate (e.g. uncover, cover1, cover2).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to export evaluation metrics to JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for evaluation."""
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    data_dir = args.data_dir or data_cfg.get("data_dir", "data/SLP/danaLab")
    modalities = data_cfg.get("modalities", ["IR"])
    covers = args.covers or data_cfg.get("covers", ["uncover"])
    image_size = tuple(data_cfg.get("image_size", (256, 256)))
    heatmap_size = tuple(data_cfg.get("heatmap_size", (64, 64)))
    sigma = data_cfg.get("sigma", 2.0)

    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", len(modalities))
    if "RGB" in [m.upper() for m in modalities]:
        in_channels = 3 if len(modalities) == 1 else 3 + len(modalities) - 1

    model = ResNetPose(
        backbone=model_cfg.get("backbone", "resnet18"),
        in_channels=in_channels,
        num_joints=model_cfg.get("num_joints", 14),
        pretrained=False,
    )

    # Load model from checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    lit_module = InBedPoseLightningModule.load_from_checkpoint(
        args.checkpoint,
        model=model,
        map_location=args.device,
    )

    subjects = (
        range(1, 31)
        if args.split == "train"
        else (range(31, 41) if args.split == "val" else range(41, 46))
    )

    # Initialize evaluation dataset & loader
    dataset = SLPDataset(
        root=data_dir,
        subjects=subjects,
        split=args.split,
        modalities=modalities,
        covers=covers,
        image_size=image_size,
        heatmap_size=heatmap_size,
        sigma=sigma,
        augmenter=None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pose,
    )

    device = torch.device(args.device)
    cover_desc = "+".join(covers)
    results = evaluate_model(lit_module, dataloader, device=device)
    print_evaluation_summary(
        results, cover_label=f"{args.split.upper()} - {cover_desc}"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Exported metrics to {args.output_json}")


if __name__ == "__main__":
    main()
