"""Unit and integration tests for Optuna HPO, pruning, and study analysis."""

import os
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState, create_trial

from scripts.analyze_hpo import analyze_study, print_analysis
from scripts.tune import run_hpo_study, sample_hyperparameters


def test_sample_hyperparameters() -> None:
    """Verifies that sample_hyperparameters respects bounds and types."""
    search_space = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "optimizer": {"type": "categorical", "choices": ["adamw", "adam"]},
        "batch_size": {"type": "categorical", "choices": [8, 16]},
    }
    semi_space = {
        "conf_threshold": {"type": "float", "low": 0.3, "high": 0.6, "step": 0.1},
    }

    study = optuna.create_study()
    trial = study.ask()

    sampled = sample_hyperparameters(
        trial,
        search_space,
        semi_supervised=True,
        semi_space=semi_space,
    )

    assert 1e-4 <= sampled["lr"] <= 1e-2
    assert sampled["optimizer"] in ["adamw", "adam"]
    assert sampled["batch_size"] in [8, 16]
    assert 0.3 <= sampled["conf_threshold"] <= 0.6


def test_run_hpo_study_smoke(mock_slp_root: Path) -> None:
    """Verifies that run_hpo_study runs multiple trials in fast_dev_run mode."""
    study = run_hpo_study(
        config_path="configs/hpo.yaml",
        n_trials=2,
        data_dir=str(mock_slp_root),
        max_epochs=1,
        enable_pruning=False,
        fast_dev_run=True,
    )

    assert len(study.trials) == 2
    assert study.best_trial is not None


def test_analyze_study_and_tryhard(tmp_path: Path) -> None:
    """Verifies analyze_study calculates gain, parameter importance, and exports plots."""
    study = optuna.create_study(direction="maximize")

    dists = {
        "lr": FloatDistribution(1e-4, 1e-2, log=True),
        "sigma": FloatDistribution(1.5, 3.5, step=0.5),
        "loss_type": CategoricalDistribution(["mse", "smooth_l1"]),
    }

    # Trial 0: Baseline level
    t0 = create_trial(
        params={"lr": 0.001, "sigma": 2.0, "loss_type": "mse"},
        distributions=dists,
        value=0.72,
    )
    study.add_trial(t0)

    # Trial 1: Worst trial (overshoot lr, tiny sigma)
    t1 = create_trial(
        params={"lr": 0.005, "sigma": 1.5, "loss_type": "smooth_l1"},
        distributions=dists,
        value=0.45,
    )
    study.add_trial(t1)

    # Trial 2: Best trial (> 3% gain)
    t2 = create_trial(
        params={"lr": 0.0008, "sigma": 2.5, "loss_type": "mse"},
        distributions=dists,
        value=0.78,
    )
    study.add_trial(t2)

    # Trial 3: Pruned trial
    t3 = create_trial(
        params={"lr": 0.0001, "sigma": 1.5, "loss_type": "mse"},
        distributions=dists,
        state=TrialState.PRUNED,
    )
    study.add_trial(t3)

    out_dir = tmp_path / "hpo_artifacts"
    summary = analyze_study(study, baseline_score=0.72, output_dir=str(out_dir))

    assert summary["best_trial"]["value"] == 0.78
    assert summary["worst_trial"]["value"] == 0.45
    assert summary["delta"]["meets_3pct_gain_requirement"] is True
    assert summary["n_pruned"] == 1

    tryhard = summary["tryhard_analysis"]
    assert "worst_trial_breakdown" in tryhard
    assert "one_sentence_takeaway" in tryhard
    assert len(summary["parameter_importances"]) > 0

    # Verify generated artifact files
    assert (out_dir / "optimization_history.png").exists()
    assert (out_dir / "param_importances.png").exists()
    assert (out_dir / "hpo_summary.json").exists()

    # Terminal report output smoke test
    print_analysis(summary)


def test_tune_cli_smoke(mock_slp_root: Path) -> None:
    """Verifies tune.py CLI execution with fast_dev_run."""
    cmd = [
        sys.executable,
        "scripts/tune.py",
        "--config",
        "configs/hpo.yaml",
        "--data_dir",
        str(mock_slp_root),
        "--n_trials",
        "1",
        "--max_epochs",
        "1",
        "--fast_dev_run",
        "--no_pruning",
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res.returncode == 0, f"tune.py CLI failed:\n{res.stderr}\n{res.stdout}"
    assert "OPTUNA OPTIMIZATION COMPLETE" in res.stdout
