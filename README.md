# In-Bed Pose Estimation MLOps (`inbed-pose-mlops`)

[![Python CI](https://github.com/MichalRedm/inbed-pose-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/MichalRedm/inbed-pose-mlops/actions/workflows/ci.yml)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.x-792ee5.svg)](https://lightning.ai/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Optuna HPO](https://img.shields.io/badge/Optuna-HPO-005B94.svg)](https://optuna.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)

Production-grade MLOps repository for **In-Bed Human Pose Estimation** on the **Simultaneously-collected Multimodal Lying Pose (SLP)** dataset (VIP Cup 2021). Features modular **PyTorch Lightning** pipelines, **MLflow** visual experiment tracking, automated **Optuna** hyperparameter optimization with trial pruning ($\ge 3\%$ PCK gain), and dual-compute support for local laptops (NVIDIA RTX 3050 4GB) and remote Kaggle runners (2x Tesla T4 30GB).

---

## 🏛️ System Architecture

```
                                    +-----------------------+
                                    |      SLP Dataset      |
                                    | (IR, RGB / Uncovered, |
                                    |    Cover1, Cover2)    |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    |     SLPDataModule     |
                                    | (Dual-Stream Loader,  |
                                    |   Pose Augmentations) |
                                    +-----------+-----------+
                                                |
                                                v
               +-----------------------------------------------------------------+
               |                   InBedPoseLightningModule                      |
               |                                                                 |
               |   [Supervised Mode]               [Semi-Supervised Mode]        |
               |   Student (ResNetPose) <-------+  Student Network (ResNetPose)  |
               |   MSE Heatmap Loss             |  EMA Teacher (Requires Grad=F) |
               |                                |  Cutout Consistency + Gating   |
               +--------------------------------+--------------------------------+
                                                |
                                                v
                                    +-----------------------+
                                    |    MLflow Tracking    |
                                    | (val_pck, val_mpjpe,  |
                                    |   Visual Skeletons)   |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    |      Optuna HPO       |
                                    |   (Median Pruning,    |
                                    |    >= 3% PCK Gain)    |
                                    +-----------------------+
```

---

## ⚡ Quickstart

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies using `uv` or `pip`:
```bash
git clone https://github.com/MichalRedm/inbed-pose-mlops.git
%cd inbed-pose-mlops

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Verify Local Installation (Smoke Test)
Run the test suite across all modules:
```bash
pytest tests/ -v
```

---

## 📊 Model Monitoring with MLflow UI

All training runs and Optuna trials log scalar metrics (`train_loss`, `val_loss`, `val_pck`, `val_mpjpe`), model checkpoints, and visual prediction figures to a local SQLite database (`sqlite:///mlflow.db`).

### Launch the Local MLflow Dashboard
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser to view:
- **Real-time Training & Validation Curves**: Monitor convergence and overfitting across epochs.
- **Visual Prediction Artifacts**: Inspect skeletal keypoint overlays (green = ground truth, red = model prediction) logged at `predictions/epoch_XXX.png`.
- **Nested HPO Trial Tree**: Compare individual Optuna trials under the parent optimization study.

---

## 🏋️ Training & Evaluation

### Train Supervised Baseline Model
```bash
# Smoke test (1 batch dev run)
python scripts/train.py --config configs/baseline.yaml --fast_dev_run

# Full local training on IR uncover data (RTX 3050 friendly)
python scripts/train.py \
    --config configs/baseline.yaml \
    --data_dir data/SLP/danaLab \
    --batch_size 16 \
    --precision 16-mixed \
    --max_epochs 25
```

### Train SOTA Semi-Supervised Model (Teacher-Student)
Utilizes unannotated blanket-covered poses (`cover1`, `cover2`) via EMA pseudo-labeling:
```bash
python scripts/train.py \
    --config configs/semi_supervised.yaml \
    --data_dir data/SLP/danaLab \
    --batch_size 16 \
    --semi_supervised \
    --max_epochs 30
```

### Standalone Checkpoint Evaluation
Evaluate a trained model checkpoint on specific cover conditions (`uncover`, `cover1`, `cover2`):
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best-checkpoint-val_pck=0.7812.ckpt \
    --config configs/baseline.yaml \
    --data_dir data/SLP/danaLab \
    --split test \
    --covers cover1 cover2 \
    --output_json artifacts/evaluation_results.json
```

---

## 🎯 Hyperparameter Optimization (Optuna HPO)

The HPO pipeline tunes 7+ key hyperparameters across optimization, architecture, and Gaussian target width with automatic trial pruning:

### Run the Optimization Study
```bash
python scripts/tune.py \
    --config configs/hpo.yaml \
    --data_dir data/SLP/danaLab \
    --n_trials 20 \
    --max_epochs 15 \
    --baseline_score 0.72
```

### Analyze Results & Generate Tryhard Visualizations
```bash
python scripts/analyze_hpo.py \
    --config configs/hpo.yaml \
    --baseline_score 0.72 \
    --output_dir artifacts/hpo
```
Outputs:
- `artifacts/hpo/optimization_history.png`: Complete study progression highlighting pruned trials.
- `artifacts/hpo/param_importances.png`: Parameter sensitivity bar chart.
- `artifacts/hpo/hpo_summary.json`: Detailed breakdown with $\ge 3\%$ gain validation and worst-run failure mode diagnostics.

---

## 🚀 Homework 2: Model Serving & Dynamic Batching Benchmark (BentoML)

The trained Homework 1 `ResNetPose` (`resnet18`, 1-channel IR) model is served as a REST API using **BentoML** (`src/serving/service.py`) with configurable **adaptive dynamic batching** (`batchable=True`, `max_batch_size=32`, `max_latency_ms=3000`).

### 1. Package the Trained HW1 Model
Export the trained PyTorch Lightning checkpoint into the local BentoML Model Store:
```bash
python scripts/export_model.py --config configs/serving.yaml
```

### 2. Launch the BentoML REST API Service
Start the service with **adaptive dynamic batching** enabled (default):
```bash
python -m bentoml serve src.serving.service:InBedPoseService --host 127.0.0.1 --port 3000
```
Start the service **without dynamic batching** (`batchable=False`):
```bash
POSE_ENABLE_BATCHING=0 python -m bentoml serve src.serving.service:InBedPoseService --host 127.0.0.1 --port 3000
```

### 3. Client Request Example & Working Output
Query the running service using the Python client:
```bash
python scripts/client_example.py --config configs/serving.yaml
```
Or via `curl` (sending a base64-encoded 256×256 IR image):
```bash
curl -X POST "http://127.0.0.1:3000/predict" \
  -H "Content-Type: application/json" \
  -d '{"images_b64": ["<BASE64_ENCODED_IMAGE>"]}'
```
Example output returned by `scripts/client_example.py`:
```text
Service URL          : http://127.0.0.1:3000
Model / Backbone     : inbed_pose_resnet18:latest (resnet18)
Device               : cpu
Batching Enabled     : True
Batch Size Processed : 1
Inference Time       : 125.64 ms
Joint          |   X (px) |   Y (px) | Confidence
R_Ankle        |   156.00 |    52.00 |     7.2999
...
Head           |   156.00 |    52.00 |     6.5228
```

### 4. Dynamic Batching Benchmark Results
Run the full asynchronous load test (`15s` warm-up, `60s` measurement window, `3` repeats per concurrency level):
```bash
python scripts/benchmark_serving.py --config configs/serving.yaml --auto-spawn
```

- **Hardware**: AMD Ryzen 7 6800H (8C/16T), PyTorch `2.15.0+cpu` (`cpu` execution)
- **Batching Parameters**: `max_batch_size = 32`, `max_latency_ms = 3000` (`0` failed/dropped requests across all 18 runs)

| Concurrency | Throughput no batching [samples/s] | Throughput with batching [samples/s] | p95 latency no batching [ms] | p95 latency with batching [ms] |
|---|---|---|---|---|
| 1 | 23.23 ± 0.52 | 22.55 ± 2.21 | 49.9 ± 2.6 | 53.7 ± 16.7 |
| 8 | 28.25 ± 0.06 | 33.10 ± 5.74 | 298.1 ± 2.9 | 287.0 ± 69.4 |
| 32 | 27.43 ± 0.65 | 31.78 ± 0.53 | 1275.9 ± 58.1 | 1120.2 ± 32.2 |

> **Observation**: With `max_latency_ms = 3000` (giving the CPU enough headroom to finish batched forward passes without `503` load shedding), adaptive batching forms average batches of `~4.1` samples at concurrency 8 and `~16.3` samples at concurrency 32, raising throughput by **+17.2%** (concurrency 8) and **+15.9%** (concurrency 32) while also lowering `p95` latency (`1275.9 ms` → `1120.2 ms` at concurrency 32).

---

## ☁️ Compute Strategy: Local GPU vs. Remote Kaggle

- **Local Machine (NVIDIA RTX 3050 Laptop, 4 GB VRAM)**:
  - Highly optimized memory footprint using `precision="16-mixed"` and `batch_size=16`.
  - Recommended for unit testing, fast dev runs, and single-model training.
- **Remote Cloud (Kaggle Dual Tesla T4, 30 GB VRAM)**:
  - Distributed Data Parallel (`ddp`) multi-GPU training with `batch_size=32-64`.
  - Recommended for full Optuna studies (20–50 trials).
  - See [`docs/remote_gpu_guide.md`](docs/remote_gpu_guide.md) and [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb).

---

## 🎤 Presentation & Exam Guide

Preparing for the course defense? Consult [`docs/presentation_guide.md`](docs/presentation_guide.md) for:
- 5-minute timed slide structure.
- Narrative script for MLflow training curves.
- Step-by-step Optuna tuning justification.
- **Tryhard Section**: Complete architectural breakdown of the worst run vs best run, parameter importance ranking, and key one-sentence takeaways.

---

## 🛠️ Developer Tooling & Quality Checks

Run local CI gates before committing:
```bash
# Linting & Formatting Check
ruff check . && ruff format --check .

# Static Type Checking
mypy src scripts --config-file mypy.ini

# Test Suite
pytest tests/ -v
```

---

## 📄 License
MIT License. Developed for MLOps Coursework & Research.
