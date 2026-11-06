# Remote GPU Execution Guide (Kaggle Dual Tesla T4)

This guide provides instructions for training and optimizing models on remote dual-GPU hardware using Kaggle's free compute environment.

---

## 💻 Hardware Comparison & Strategy

| Feature | Local Laptop GPU | Remote Kaggle GPU |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop | **2x NVIDIA Tesla T4** |
| **Total VRAM** | 4,096 MiB (4 GB) | **30,720 MiB (30 GB: 2x 15 GB)** |
| **Compute Capability**| 8.6 (Ampere) | 7.5 (Turing) |
| **Execution Mode** | Single GPU (`devices=1`) | **Distributed Data Parallel (`strategy="ddp"`, `devices=2`)** |
| **Recommended Use** | Fast dev runs, unit tests, single model training (`batch_size=16`) | **Multi-trial Optuna HPO (20–50 trials), semi-supervised self-training (`batch_size=32–64`)** |

---

## 🚀 Running on Kaggle in 5 Steps

### 1. Create a Kaggle Notebook
1. Open [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right-hand **Notebook Options** panel:
   - Set **Accelerator** to **GPU T4 x2**.
   - Ensure **Internet** is toggled **ON**.

### 2. Import Notebook or Clone Repository
You can import [`notebooks/kaggle_runner.ipynb`](../notebooks/kaggle_runner.ipynb) directly into Kaggle:
```bash
!git clone https://github.com/MichalRedm/inbed-pose-mlops.git
%cd inbed-pose-mlops
!pip install -r requirements.txt
```

### 3. Attach the SLP Dataset
Attach the Simultaneous Multimodal Lying Pose (SLP) dataset to the Kaggle notebook via **Add Input**, or download it into `/kaggle/input/slp-dataset/danaLab`.

### 4. Execute Multi-GPU Baseline Training
Run PyTorch Lightning across both Tesla T4 GPUs with PyTorch Distributed Data Parallel (`ddp`):
```bash
python scripts/train.py \
    --config configs/remote_kaggle.yaml \
    --data_dir /kaggle/input/slp-dataset/danaLab \
    --devices 2 \
    --strategy ddp \
    --batch_size 32 \
    --max_epochs 30
```

### 5. Run Parallel Optuna HPO & Analysis
```bash
# 1. Run Optuna HPO study (20 trials with median pruning)
python scripts/tune.py \
    --config configs/hpo.yaml \
    --data_dir /kaggle/input/slp-dataset/danaLab \
    --n_trials 20 \
    --max_epochs 15 \
    --baseline_score 0.72

# 2. Generate Tryhard analysis plots and metrics summary
python scripts/analyze_hpo.py \
    --config configs/hpo.yaml \
    --baseline_score 0.72 \
    --output_dir artifacts/hpo
```

### 6. Download Artifacts to Local Machine
The notebook packages all artifacts and the SQLite MLflow database into `/kaggle/working/inbed_pose_results.tar.gz`. Download this file and extract it into your local repository to inspect everything in your local MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
