"""Hyperparameter Optimization (HPO) pipeline with Optuna, PyTorch Lightning, and MLflow."""

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path for direct CLI execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import optuna
import pytorch_lightning as pl
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna_integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from scripts.train import load_config
from src.data.datamodule import SLPDataModule
from src.models.resnet_pose import ResNetPose
from src.training.lightning_module import InBedPoseLightningModule


def sample_hyperparameters(
    trial: optuna.Trial,
    search_space: dict[str, Any],
    semi_supervised: bool = False,
    semi_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Samples hyperparameters from the search space specification.

    Args:
        trial: Active Optuna trial.
        search_space: Dictionary specifying parameter types, ranges, or choices.
        semi_supervised: Whether to also sample semi-supervised parameters.
        semi_space: Dictionary specifying semi-supervised parameter search ranges.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    params: dict[str, Any] = {}
    combined_space = dict(search_space)
    if semi_supervised and semi_space:
        combined_space.update(semi_space)

    for name, spec in combined_space.items():
        param_type = spec.get("type")
        if param_type == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported parameter type: {param_type} for {name}")

    return params


def create_objective(
    config: dict[str, Any],
    data_dir: str | None = None,
    max_epochs: int = 15,
    enable_pruning: bool = True,
    semi_supervised: bool = False,
    fast_dev_run: bool = False,
) -> Any:
    """Factory function creating the Optuna optimization objective.

    Args:
        config: Full HPO configuration dictionary.
        data_dir: Optional override for dataset root.
        max_epochs: Maximum epochs per trial.
        enable_pruning: Whether to enable PyTorch Lightning trial pruning.
        semi_supervised: Whether to optimize semi-supervised self-training.
        fast_dev_run: Whether to run 1 batch smoke test per trial.

    Returns:
        Callable Optuna objective function.
    """
    base_cfg_path = config.get("base_config", "configs/baseline.yaml")
    base_cfg = load_config(base_cfg_path)

    study_cfg = config.get("study", {})
    monitor = study_cfg.get("monitor", "val_pck")
    search_space = config.get("search_space", {})
    semi_space = config.get("semi_supervised_space", {})
    mlf_cfg = config.get("mlflow", {})
    tracking_uri = mlf_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlf_cfg.get("experiment_name", "inbed-pose-hpo")

    def objective(trial: optuna.Trial) -> float:
        # 1. Sample trial hyperparameters
        sampled = sample_hyperparameters(
            trial,
            search_space,
            semi_supervised=semi_supervised,
            semi_space=semi_space,
        )

        # 2. Build trial configuration
        data_cfg = dict(base_cfg.get("data", {}))
        if data_dir:
            data_cfg["data_dir"] = data_dir
        if "batch_size" in sampled:
            data_cfg["batch_size"] = sampled["batch_size"]
        if "sigma" in sampled:
            data_cfg["sigma"] = sampled["sigma"]

        model_cfg = dict(base_cfg.get("model", {}))
        train_cfg = dict(base_cfg.get("training", {}))

        train_cfg["mode"] = "semi_supervised" if semi_supervised else "supervised"
        if "lr" in sampled:
            train_cfg["lr"] = sampled["lr"]
        if "weight_decay" in sampled:
            train_cfg["weight_decay"] = sampled["weight_decay"]
        if "loss_type" in sampled:
            train_cfg["loss_type"] = sampled["loss_type"]
        if "optimizer" in sampled:
            train_cfg["optimizer"] = sampled["optimizer"]
        if "lr_scheduler" in sampled:
            train_cfg["lr_scheduler"] = sampled["lr_scheduler"]
        if "conf_threshold" in sampled:
            train_cfg["conf_threshold"] = sampled["conf_threshold"]
        if "lambda_unlabeled" in sampled:
            train_cfg["lambda_unlabeled"] = sampled["lambda_unlabeled"]
        if "ema_alpha" in sampled:
            train_cfg["ema_alpha"] = sampled["ema_alpha"]

        # 3. Instantiate DataModule
        datamodule = SLPDataModule(
            data_root=data_cfg.get("data_dir", "data/SLP/danaLab"),
            modalities=data_cfg.get("modalities", ["IR"]),
            train_covers=data_cfg.get("covers", ["uncover"]),
            val_covers=data_cfg.get("val_covers", ["uncover", "cover1", "cover2"]),
            test_covers=data_cfg.get("test_covers", ["cover1", "cover2"]),
            unlabeled_covers=data_cfg.get("covers_unlabeled", ["cover1", "cover2"]),
            batch_size=data_cfg.get("batch_size", 16),
            num_workers=0,
            image_size=tuple(data_cfg.get("image_size", (256, 256))),
            heatmap_size=tuple(data_cfg.get("heatmap_size", (64, 64))),
            sigma=data_cfg.get("sigma", 2.0),
            semi_supervised=semi_supervised,
        )

        # 4. Instantiate Model & LightningModule
        in_channels = len(data_cfg.get("modalities", ["IR"]))
        model = ResNetPose(
            backbone=model_cfg.get("backbone", "resnet18"),
            in_channels=in_channels,
            num_joints=model_cfg.get("num_joints", 14),
            pretrained=False,
        )

        lit_module = InBedPoseLightningModule(
            model=model,
            mode="semi_supervised" if semi_supervised else "supervised",
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
            optimizer_name=train_cfg.get("optimizer", "adamw"),
            lr_scheduler=train_cfg.get("lr_scheduler", "cosine"),
            loss_type=train_cfg.get("loss_type", "mse"),
            ema_alpha=train_cfg.get("ema_alpha", 0.999),
            confidence_threshold=train_cfg.get("conf_threshold", 0.35),
            lambda_unlabeled=train_cfg.get("lambda_unlabeled", 1.0),
            image_size=tuple(data_cfg.get("image_size", (256, 256))),
        )

        # 5. Callbacks
        callbacks: list[pl.Callback] = [
            ModelCheckpoint(monitor=monitor, mode="max", save_top_k=1)
        ]
        if enable_pruning and not fast_dev_run:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=monitor))

        # 6. Logger
        if fast_dev_run:
            logger: Any = False
        else:
            logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                run_name=f"trial_{trial.number:03d}",
            )
            logger.log_hyperparams(sampled)

        # 7. Trainer
        trainer = pl.Trainer(
            accelerator=train_cfg.get("accelerator", "auto"),
            devices=1,
            precision=train_cfg.get("precision", "16-mixed"),
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=logger,
            fast_dev_run=fast_dev_run,
            enable_progress_bar=False,
        )

        trainer.fit(lit_module, datamodule=datamodule)

        # 8. Retrieve target evaluation metric
        metric_val = trainer.callback_metrics.get(monitor)
        if metric_val is None:
            return 0.0

        score = float(metric_val.item())
        return score

    return objective


def run_hpo_study(
    config_path: str = "configs/hpo.yaml",
    n_trials: int = 20,
    timeout: float | None = None,
    data_dir: str | None = None,
    max_epochs: int = 15,
    semi_supervised: bool = False,
    enable_pruning: bool = True,
    baseline_score: float = 0.72,
    fast_dev_run: bool = False,
) -> optuna.Study:
    """Creates and runs an Optuna study with median pruning.

    Args:
        config_path: Path to HPO YAML configuration.
        n_trials: Number of trials to run.
        timeout: Optional wall-clock timeout in seconds.
        data_dir: SLP dataset directory.
        max_epochs: Epochs per trial.
        semi_supervised: Whether to optimize semi-supervised mode.
        enable_pruning: Whether to prune unpromising trials.
        baseline_score: Baseline metric for delta comparison.
        fast_dev_run: Smoke test flag.

    Returns:
        Completed Optuna Study instance.
    """
    config = load_config(config_path)
    study_cfg = config.get("study", {})
    pruning_cfg = config.get("pruning", {})

    pruner = (
        MedianPruner(
            n_startup_trials=pruning_cfg.get("n_warmup_steps", 3),
            n_warmup_steps=pruning_cfg.get("n_warmup_steps", 3),
            interval_steps=pruning_cfg.get("interval_steps", 1),
        )
        if enable_pruning
        else optuna.pruners.NopPruner()
    )

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_cfg.get("name", "inbed-pose-optuna-study"),
        direction=study_cfg.get("direction", "maximize"),
        pruner=pruner,
        sampler=sampler,
        storage=study_cfg.get("storage") if not fast_dev_run else None,
        load_if_exists=study_cfg.get("load_if_exists", True),
    )

    objective = create_objective(
        config=config,
        data_dir=data_dir,
        max_epochs=max_epochs,
        enable_pruning=enable_pruning,
        semi_supervised=semi_supervised,
        fast_dev_run=fast_dev_run,
    )

    print("=" * 65)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION (HPO)")
    print(f"Study Name    : {study.study_name}")
    print(f"Target Metric : {study_cfg.get('monitor', 'val_pck')}")
    print(f"Trials        : {n_trials}")
    print(
        f"Pruning       : {'Enabled (MedianPruner)' if enable_pruning else 'Disabled'}"
    )
    print("=" * 65)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Summarize results
    best_trial = study.best_trial
    best_val = best_trial.value if best_trial.value is not None else 0.0
    gain = best_val - baseline_score
    pct_gain = (gain / baseline_score) * 100 if baseline_score > 0 else 0.0

    print("\n" + "=" * 65)
    print("OPTUNA OPTIMIZATION COMPLETE")
    print("=" * 65)
    print(
        f"Best Trial #{best_trial.number}: {study_cfg.get('monitor')}: {best_trial.value:.4f}"
    )
    print(f"Baseline Score   : {baseline_score:.4f}")
    print(f"Absolute Gain    : +{gain * 100:.2f}% ({pct_gain:+.2f}% relative)")
    print(
        f"Pass Criterion (>= 3.0% Gain): {'PASSED [OK]' if gain >= 0.03 else 'NEEDS MORE TUNING'}"
    )
    print("-" * 65)
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k:<18}: {v}")
    print("=" * 65 + "\n")

    return study


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for tune.py."""
    parser = argparse.ArgumentParser(
        description="Run Optuna Hyperparameter Optimization with PyTorch Lightning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hpo.yaml",
        help="Path to HPO YAML configuration file.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=15,
        help="Maximum training epochs per trial.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for the entire optimization study.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="SLP dataset root directory.",
    )
    parser.add_argument(
        "--baseline_score",
        type=float,
        default=0.72,
        help="Baseline validation metric for gain calculation (default: 0.72).",
    )
    parser.add_argument(
        "--semi_supervised",
        action="store_true",
        help="Tune semi-supervised Teacher-Student hyperparameters.",
    )
    parser.add_argument(
        "--no_pruning",
        action="store_true",
        help="Disable trial pruning.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run fast smoke test without external storage.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for tune.py."""
    args = parse_args()
    run_hpo_study(
        config_path=args.config,
        n_trials=args.n_trials,
        timeout=args.timeout,
        data_dir=args.data_dir,
        max_epochs=args.max_epochs,
        semi_supervised=args.semi_supervised,
        enable_pruning=not args.no_pruning,
        baseline_score=args.baseline_score,
        fast_dev_run=args.fast_dev_run,
    )


if __name__ == "__main__":
    main()
