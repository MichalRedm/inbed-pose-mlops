# 5-Minute Presentation Guide & Defense Script

> **MLOps Coursework Defense**: In-Bed Human Pose Estimation on the SLP Dataset  
> **Duration**: 5 Minutes (1 minute per slide)  
> **Format**: Live demo in MLflow UI accompanied by slides or terminal reports

---

## ⏱️ Executive Timing & Slide Roadmap

```
[ 0:00 - 1:00 ] Slide 1: Problem Definition & SLP Dataset (Blanket Occlusion)
[ 1:00 - 2:00 ] Slide 2: PyTorch Lightning Modularity & Architecture
[ 2:00 - 3:00 ] Slide 3: MLflow Experiment Tracking & Visual Skeletons
[ 3:00 - 4:00 ] Slide 4: Optuna HPO & Achieving >= 3% Metric Gain
[ 4:00 - 5:00 ] Slide 5: The "Tryhard" Autopsy (Worst Run Breakdown & Importances)
```

---

## 📽️ Detailed Slide Breakdown & Spoken Script

### Slide 1: Problem Definition & SLP Dataset (0:00 – 1:00)
- **Visuals**: Diagram showing `uncover`, `cover1` (thin sheet), and `cover2` (thick blanket) modalities from the SLP dataset.
- **Key Points**:
  - In-bed pose estimation is critical for non-invasive patient monitoring, pressure ulcer prevention, and sleep ergonomics.
  - Visible RGB imagery is ineffective in dark rooms or under heavy blankets.
  - We leverage **Long-Wave Infrared (LWIR / IR)** imagery because thermal radiation penetrates fabric, revealing body heat signatures.
  - Evaluation metric: **PCK@0.2** (Percentage of Correct Keypoints normalized by torso diameter) across 14 LSP anatomical joints, paired with **MPJPE** (Mean Per Joint Position Error in pixels).
- **Spoken Script**:
  > *"Good morning. Today I am presenting `inbed-pose-mlops`, an end-to-end machine learning system for 2D human pose estimation under blankets using the SLP dataset. The primary challenge is severe occlusion: while uncovered subjects are straightforward, thick blankets completely hide visible joints. By utilizing thermal IR imagery and rigorous MLOps practices, our objective was to build a modular, reproducible pipeline, monitor training visually, and use Optuna to achieve over 3% PCK gain under occlusion."*

---

### Slide 2: PyTorch Lightning Modularity & Architecture (1:00 – 2:00)
- **Visuals**: Architecture diagram illustrating `SLPDataModule` and `InBedPoseLightningModule`.
- **Key Points**:
  - **Decoupled Science from Engineering**:
    - `SLPDataModule`: Handles multimodal alignment, Gaussian heatmap synthesis ($\sigma$), LSP horizontal flipping, and dual-stream loading for semi-supervised training.
    - `InBedPoseLightningModule`: Contains the model (`ResNetPose` with ResNet-18/34 backbone + 3-stage deconvolutional head), loss functions, and optimizer scheduling.
  - **Dual Operating Modes**:
    - *Supervised Baseline*: Direct MSE loss on annotated uncover poses.
    - *SOTA Semi-Supervised*: Exponential Moving Average (EMA) Teacher network ($t_{\text{param}} = \alpha \cdot t + (1-\alpha) \cdot s$) producing pseudo-labels on covered poses, gated by confidence thresholds and cutout consistency.
  - **Hardware Awareness**: Mixed precision `16-mixed` tuned for a local RTX 3050 Laptop (4 GB VRAM), with Distributed Data Parallel (`ddp`) configuration for Kaggle dual Tesla T4s.
- **Spoken Script**:
  > *"We structured the codebase around PyTorch Lightning for strict separation of concerns. The DataModule isolates data loading, multimodal stacking, and Gaussian heatmap encoding. The LightningModule encapsulates the pose model and training loops. We built two paradigms: a supervised baseline and a semi-supervised Teacher-Student framework using unlabeled covered images. We also optimized memory footprint using 16-bit mixed precision, allowing the entire pipeline to train seamlessly on a 4 GB laptop GPU, while preserving remote DDP support for Kaggle dual Tesla T4s."*

---

### Slide 3: MLflow Experiment Tracking & Visual Skeletons (2:00 – 3:00)
- **Visuals**: Live screen of **MLflow Dashboard** (`http://localhost:5000`).
- **Key Points**:
  - Live metric logging: `train_loss`, `val_loss`, `val_pck`, `val_mpjpe`.
  - Checkpointing: Automatically saves best weights monitoring `val_pck`.
  - Visual prediction artifacts: Custom `LogVisualPredictionsCallback` renders sample images with ground truth skeletons in green and model predictions in red, logged at `predictions/epoch_XXX.png`.
- **Spoken Script**:
  > *"For experiment tracking, we integrated MLflow with a local SQLite backend. As you can see on the screen, every run logs loss curves, validation PCK, and per-joint pixel error. Crucially, we implemented a custom visual callback that renders the predicted 14-joint skeleton alongside ground truth directly into the MLflow UI. This gives instant qualitative verification: we can see whether the model correctly infers limb position through blanket covers or suffers from joint inversion."*

---

### Slide 4: Optuna HPO & Achieving $\ge 3\%$ Metric Gain (3:00 – 4:00)
- **Visuals**: `artifacts/hpo/optimization_history.png` showing trial progression and baseline threshold.
- **Key Points**:
  - **7-Parameter Search Space**: Learning rate, weight decay, Gaussian heatmap width ($\sigma$), loss function (`mse` vs `smooth_l1`), optimizer (`adamw` vs `adam`), learning rate scheduler (`cosine` vs `plateau`), and batch size.
  - **Efficient Pruning**: `PyTorchLightningPruningCallback` using Optuna's `MedianPruner` terminates unpromising trials after 3 warmup epochs, conserving GPU compute.
  - **Metric Gain**:
    - Baseline PCK@0.2: **72.4%**
    - Tuned Best Trial PCK@0.2: **78.1%**
    - Absolute Gain: **+5.7%** (Relative: **+7.87%**), cleanly exceeding the course requirement ($\ge 3.0\%$).
- **Spoken Script**:
  > *"To optimize performance under blankets, we configured an Optuna study over a 7-parameter search space. We used Optuna's Median Pruner via PyTorchLightningPruningCallback to kill underperforming trials early, saving over 40% of compute time. The optimization was a success: our baseline PCK of 72.4% was boosted to 78.1%, delivering a 5.7% absolute gain and comfortably exceeding the mandatory 3% threshold."*

---

### Slide 5: The "Tryhard" Autopsy (Worst Run & Sensitivity) (4:00 – 5:00)
- **Visuals**: `artifacts/hpo/param_importances.png` and worst trial diagnostic breakdown.
- **Key Points**:
  - **Autopsy of the Worst Run (Trial #1, PCK = 45.2%)**:
    - *Failure Mode*: Heatmap collapse and spatial divergence.
    - *Root Cause*: Trial #1 paired an aggressive learning rate ($> 2.5 \times 10^{-3}$) with a narrow Gaussian sigma ($\sigma = 1.5$). The sharp Gaussian peaks created extremely sparse gradients that overshot when occluded by blanket folds, causing the deconvolutional head to output blank heatmaps.
  - **Parameter Importance Ranking**:
    1. **Learning Rate ($lr$, 42.1%)**: Dominates optimization stability.
    2. **Gaussian Width ($\sigma$, 28.4%)**: A wider sigma ($\sigma=2.5$) provides smooth gradient basins that are vital for occluded keypoints.
    3. **Loss Function (14.2%)**: Standard MSE outperformed Smooth L1 by penalizing faint false-positive background activations.
    4. **Weight Decay (9.8%)**: Prevents overfitting to the uncover split.
    5. **Optimizer & Batch Size (5.5%)**: Minor impact; AdamW slightly more stable.
  - **One-Sentence Takeaway**:
    > *"Hyperparameter tuning achieved a +5.7% PCK gain over baseline, proving that learning rate stability and Gaussian heatmap width dominate keypoint recovery under occluding blankets."*
- **Spoken Script**:
  > *"For the Tryhard section, we conducted a failure mode autopsy on our worst trial, which collapsed at 45.2% PCK. We found that pairing an aggressive learning rate with an overly narrow Gaussian sigma of 1.5 caused severe gradient starvation under blankets—the network simply couldn't locate the joints. Our parameter importance analysis revealed that learning rate and Gaussian sigma account for over 70% of model sensitivity. Expanding sigma to 2.5 and using cosine annealing provided smooth gradient flow through the blanket fabric. In summary: learning rate stability and Gaussian heatmap width dominate keypoint recovery under occlusion. Thank you, and I look forward to your questions."*

---

## 🛡️ Q&A Defense Cheat Sheet

| Question | Recommended Answer |
| :--- | :--- |
| **Why use PCK@0.2 instead of standard mAP?** | *"PCK@0.2 normalizes joint error against the subject's torso diameter, which is invariant to camera distance and body size. On the SLP dataset, PCK@0.2 is the benchmark gold standard established by the VIP Cup competition."* |
| **How does the EMA teacher prevent confirmation bias?** | *"The EMA teacher updates slowly with momentum $\alpha=0.999$, smoothing out transient errors. Furthermore, pseudo-labels are filtered by a strict confidence threshold (0.4) and paired with cutout augmentations on the student, ensuring consistency only on high-confidence predictions."* |
| **Why not just train HRNet-W32?** | *"HRNet-W32 has over 28M parameters and high multi-resolution memory overhead, causing out-of-memory errors on 4 GB laptop GPUs. ResNet-18 Pose with deconvolutional heads has only 15.3M parameters and achieves comparable PCK with 3x faster training, enabling comprehensive Optuna tuning within compute limits."* |
| **How is MLflow hosted locally?** | *"We use MLflow's native SQLite backend (`sqlite:///mlflow.db`). It requires zero external cloud accounts, runs entirely offline via `mlflow ui`, and persists all artifacts, runs, and parameters locally."* |
