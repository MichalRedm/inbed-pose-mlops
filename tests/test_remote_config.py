"""Unit tests for remote execution configuration and Kaggle runner notebook."""

import json
from pathlib import Path

from scripts.train import load_config


def test_remote_kaggle_config_valid() -> None:
    """Verifies configs/remote_kaggle.yaml structure and multi-GPU settings."""
    cfg = load_config("configs/remote_kaggle.yaml")

    assert "data" in cfg
    assert "training" in cfg
    assert "mlflow" in cfg

    train_cfg = cfg["training"]
    assert train_cfg.get("accelerator") == "gpu"
    assert train_cfg.get("devices") == 2
    assert train_cfg.get("strategy") == "ddp"
    assert train_cfg.get("precision") == "16-mixed"

    data_cfg = cfg["data"]
    assert data_cfg.get("batch_size") == 32
    assert data_cfg.get("num_workers") == 2


def test_kaggle_notebook_structure() -> None:
    """Verifies notebooks/kaggle_runner.ipynb exists and is valid Jupyter JSON."""
    nb_path = Path("notebooks/kaggle_runner.ipynb")
    assert nb_path.exists(), "Kaggle runner notebook missing"

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 5
    assert nb.get("nbformat") == 4
