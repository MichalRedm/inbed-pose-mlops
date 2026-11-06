"""Training script for In-Bed Pose Estimation with PyTorch Lightning and MLflow."""

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path for direct CLI execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import MLFlowLogger

from src.data.datamodule import SLPDataModule
from src.models.resnet_pose import ResNetPose
from src.training.callbacks import get_default_callbacks
from src.training.lightning_module import InBedPoseLightningModule


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def run_training(
    config: dict[str, Any],
    fast_dev_run: bool = False,
    override_logger: Any = None,
) -> tuple[pl.Trainer, InBedPoseLightningModule, list[dict[str, float]]]:
    """Configures and executes the training pipeline.

    Args:
        config: Nested configuration dictionary.
        fast_dev_run: Whether to run a single batch smoke test.
        override_logger: Optional logger instance (e.g., for HPO or testing).

    Returns:
        Tuple of (Trainer, trained LightningModule, test_results).
    """
    # 1. Set global random seed
    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # 2. Extract configuration sections
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    mlf_cfg = config.get("mlflow", {})

    mode = train_cfg.get("mode", "supervised")

    # 3. Initialize DataModule
    data_root = data_cfg.get("data_dir", data_cfg.get("data_root", "data/SLP/danaLab"))
    train_covers = data_cfg.get("covers", data_cfg.get("train_covers", ["uncover"]))
    val_covers = data_cfg.get("val_covers", ["uncover", "cover1", "cover2"])
    test_covers = data_cfg.get("test_covers", ["cover1", "cover2"])
    unlabeled_covers = data_cfg.get(
        "covers_unlabeled", data_cfg.get("unlabeled_covers", ["cover1", "cover2"])
    )

    datamodule = SLPDataModule(
        data_root=data_root,
        modalities=data_cfg.get("modalities", ["IR"]),
        train_covers=train_covers,
        val_covers=val_covers,
        test_covers=test_covers,
        unlabeled_covers=unlabeled_covers,
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        image_size=tuple(data_cfg.get("image_size", (256, 256))),
        heatmap_size=tuple(data_cfg.get("heatmap_size", (64, 64))),
        sigma=data_cfg.get("sigma", 2.0),
        semi_supervised=(mode == "semi_supervised"),
    )

    # 4. Initialize Pose Model
    in_channels = model_cfg.get(
        "in_channels",
        len(data_cfg.get("modalities", ["IR"])),
    )
    # If RGB is included, adjust channels appropriately
    if "RGB" in [m.upper() for m in data_cfg.get("modalities", ["IR"])]:
        in_channels = (
            3
            if len(data_cfg.get("modalities", [])) == 1
            else 3 + len(data_cfg.get("modalities", [])) - 1
        )

    model = ResNetPose(
        backbone=model_cfg.get("backbone", "resnet18"),
        in_channels=in_channels,
        num_joints=model_cfg.get("num_joints", 14),
        pretrained=model_cfg.get("pretrained", False),
    )

    # 5. Initialize LightningModule
    lit_module = InBedPoseLightningModule(
        model=model,
        mode=mode,
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        optimizer_name=train_cfg.get("optimizer", "adamw"),
        lr_scheduler=train_cfg.get("lr_scheduler", "cosine"),
        loss_type=train_cfg.get("loss_type", "mse"),
        ema_alpha=train_cfg.get("ema_alpha", 0.999),
        confidence_threshold=train_cfg.get(
            "confidence_threshold", train_cfg.get("conf_threshold", 0.35)
        ),
        lambda_unlabeled=train_cfg.get("lambda_unlabeled", 1.0),
        image_size=tuple(data_cfg.get("image_size", (256, 256))),
    )

    # 6. Configure Callbacks
    save_dir = train_cfg.get("visual_save_dir", "artifacts/predictions")
    callbacks = get_default_callbacks(
        monitor=train_cfg.get("monitor", "val_pck"),
        mode="max",
        save_top_k=1,
        patience=train_cfg.get("patience", 10),
        log_predictions_every_n_epochs=train_cfg.get(
            "log_predictions_every_n_epochs", 1
        ),
        save_dir=save_dir if not fast_dev_run else None,
    )

    # 7. Configure MLflow Logger
    if fast_dev_run:
        logger: Any = False
    elif override_logger is not None:
        logger = override_logger
    else:
        tracking_uri = mlf_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        experiment_name = mlf_cfg.get("experiment_name", "inbed-pose-baseline")
        run_name = mlf_cfg.get("run_name", None)

        logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name=run_name,
            log_model=True,
        )
        logger.log_hyperparams(config)

    # 8. Configure Trainer
    trainer = pl.Trainer(
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        precision=train_cfg.get("precision", "16-mixed"),
        max_epochs=train_cfg.get("max_epochs", 25),
        callbacks=callbacks if not fast_dev_run else [],
        logger=logger,
        fast_dev_run=fast_dev_run,
        enable_checkpointing=not fast_dev_run,
    )

    # 9. Train Model
    trainer.fit(lit_module, datamodule=datamodule)

    # 10. Evaluate on Test Set
    test_results: list[dict[str, float]] = []
    if not fast_dev_run:
        raw_results = trainer.test(lit_module, datamodule=datamodule, ckpt_path="best")
    else:
        raw_results = trainer.test(lit_module, datamodule=datamodule)
    test_results = [dict(r) for r in raw_results]

    return trainer, lit_module, test_results


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train In-Bed Pose Estimation Model with PyTorch Lightning and MLflow."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root path to SLP dataset directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run 1 batch smoke test without saving artifacts.",
    )
    parser.add_argument(
        "--semi_supervised",
        action="store_true",
        help="Run in semi-supervised self-training mode.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name.",
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db).",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help="Hardware accelerator ('auto', 'cpu', 'gpu').",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of devices / GPUs.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision ('16-mixed', '32-true').",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for training."""
    args = parse_args()
    config = load_config(args.config)

    # Apply command-line overrides
    if args.data_dir:
        config.setdefault("data", {})["data_dir"] = args.data_dir
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["lr"] = args.lr
    if args.max_epochs:
        config.setdefault("training", {})["max_epochs"] = args.max_epochs
    if args.semi_supervised:
        config.setdefault("training", {})["mode"] = "semi_supervised"
    if args.accelerator:
        config.setdefault("training", {})["accelerator"] = args.accelerator
    if args.devices:
        config.setdefault("training", {})["devices"] = args.devices
    if args.precision:
        config.setdefault("training", {})["precision"] = args.precision
    if args.experiment_name:
        config.setdefault("mlflow", {})["experiment_name"] = args.experiment_name
    if args.run_name:
        config.setdefault("mlflow", {})["run_name"] = args.run_name
    if args.tracking_uri:
        config.setdefault("mlflow", {})["tracking_uri"] = args.tracking_uri

    print("=" * 60)
    print("In-Bed Pose Estimation MLOps Training Pipeline")
    print(f"Config: {args.config}")
    print(f"Mode: {config.get('training', {}).get('mode', 'supervised')}")
    print(f"Fast Dev Run: {args.fast_dev_run}")
    print("=" * 60)

    run_training(config, fast_dev_run=args.fast_dev_run)


if __name__ == "__main__":
    main()
