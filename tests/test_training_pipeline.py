"""Integration and smoke tests for the PyTorch Lightning training pipeline."""

import os
import subprocess
import sys
from pathlib import Path

from scripts.train import load_config, run_training


def test_load_config() -> None:
    """Verifies that default config files are syntactically valid YAML."""
    baseline_cfg = load_config("configs/baseline.yaml")
    assert "data" in baseline_cfg
    assert "model" in baseline_cfg
    assert "training" in baseline_cfg
    assert baseline_cfg["training"]["mode"] == "supervised"

    semi_cfg = load_config("configs/semi_supervised.yaml")
    assert semi_cfg["training"]["mode"] == "semi_supervised"
    assert "covers_unlabeled" in semi_cfg["data"]


def test_run_training_supervised_fast_dev_run(mock_slp_root: Path) -> None:
    """Smoke tests end-to-end training pipeline in supervised mode."""
    config = load_config("configs/baseline.yaml")
    config["data"]["data_dir"] = str(mock_slp_root)
    config["data"]["batch_size"] = 2
    config["training"]["accelerator"] = "cpu"
    config["training"]["precision"] = "32-true"

    trainer, model, test_results = run_training(config, fast_dev_run=True)

    assert trainer is not None
    assert model is not None
    assert isinstance(test_results, list)
    assert len(test_results) > 0
    assert "test_loss" in test_results[0]


def test_run_training_semi_supervised_fast_dev_run(mock_slp_root: Path) -> None:
    """Smoke tests end-to-end training pipeline in semi-supervised mode."""
    config = load_config("configs/semi_supervised.yaml")
    config["data"]["data_dir"] = str(mock_slp_root)
    config["data"]["batch_size"] = 2
    config["training"]["accelerator"] = "cpu"
    config["training"]["precision"] = "32-true"

    trainer, model, test_results = run_training(config, fast_dev_run=True)

    assert trainer is not None
    assert model is not None
    assert model.mode == "semi_supervised"
    assert model.teacher is not None
    assert len(test_results) > 0


def test_train_cli_smoke(mock_slp_root: Path) -> None:
    """Verifies scripts/train.py CLI flags and execution."""
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        "configs/baseline.yaml",
        "--data_dir",
        str(mock_slp_root),
        "--batch_size",
        "2",
        "--accelerator",
        "cpu",
        "--precision",
        "32-true",
        "--fast_dev_run",
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res.returncode == 0, f"CLI train failed:\n{res.stderr}\n{res.stdout}"
    assert "In-Bed Pose Estimation MLOps Training Pipeline" in res.stdout


def test_populate_mlflow_smoke(tmp_path: Path) -> None:
    """Smoke tests populate_dashboard script with temporary SQLite backend."""
    from scripts.populate_mlflow import populate_dashboard

    temp_db = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{temp_db}"

    populate_dashboard(tracking_uri=tracking_uri)

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exps = [e.name for e in client.search_experiments()]
    assert "inbed-pose-baseline" in exps
    assert "inbed-pose-semi-supervised" in exps
    assert "inbed-pose-hpo" in exps
