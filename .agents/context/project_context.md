# Project Context & Roadmap

## Overview
**inbed-pose-mlops** is a machine learning repository designed for Homework 1 of the MLOps course. It tackles in-bed human pose estimation using the Simultaneous Multimodal Lying Pose (SLP) dataset, integrating PyTorch Lightning, MLflow experiment tracking, and Optuna hyperparameter optimization.

---

## Dataset: SLP (Simultaneously-collected Multimodal Lying Pose)
- **Modalities**: RGB, Long-Wave Infrared (LWIR / IR), Depth.
- **Occlusion Covers**:
  - `uncover`: Person lying in bed without blanket (annotated ground truth).
  - `cover1`: Thin sheet cover (partially occluded, unannotated or pseudo-labeled).
  - `cover2`: Thick blanket cover (heavily occluded, unannotated or pseudo-labeled).
- **Keypoints**: 14 LSP-format joints:
  1. R_Ankle, 2. R_Knee, 3. R_Hip, 4. L_Hip, 5. L_Knee, 6. L_Ankle,
  7. R_Wrist, 8. R_Elbow, 9. R_Shoulder, 10. L_Shoulder, 11. L_Elbow, 12. L_Wrist,
  13. Thorax, 14. Head.
- **Evaluation Metric**: PCK@0.2 (Percentage of Correct Keypoints normalized by torso diameter) and MPJPE (Mean Per Joint Position Error).

---

## Phased Pull Request Roadmap
- [x] **PR 1**: `chore/repo-scaffolding-and-ci` (Repository layout, agent governance, developer tooling, GitHub Actions CI)
- [x] **PR 2**: `feat/slp-data-pipeline` (Dataset, augmentations, `SLPDataModule`, unit tests)
- [x] **PR 3**: `feat/pose-models-and-lightning-module` (`ResNetPose`, `InBedPoseLightningModule`, PCK/MPJPE metrics, unit tests)
- [ ] **PR 4**: `feat/mlflow-logging-and-training-pipeline` (CLI training, MLflow visual callbacks, checkpoints, smoke test)
- [ ] **PR 5**: `feat/optuna-hpo-pipeline` (Optuna runner, pruning callback, $\ge 3\%$ gain verification, parameter importance plots)
- [ ] **PR 6**: `feat/remote-gpu-support` (Streamlined Kaggle notebook, remote runner, multi-GPU DDP config)
- [ ] **PR 7**: `docs/documentation-and-presentation-guide` (`README.md`, 5-minute presentation notes & Tryhard materials)
