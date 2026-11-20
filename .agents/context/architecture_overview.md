# Architecture Overview

## Subsystem Architecture

```
inbed-pose-mlops/
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch SLP Dataset with Gaussian heatmap generation
│   │   ├── datamodule.py       # SLPDataModule (pl.LightningDataModule)
│   │   └── augmentations.py    # Spatial & occlusion augmentations
│   ├── models/
│   │   ├── base.py             # BasePoseModel interface
│   │   ├── resnet_pose.py      # ResNet-18/34 with deconvolutional head
│   │   └── hrnet.py            # Optional High-Resolution Network
│   ├── training/
│   │   ├── lightning_module.py # InBedPoseLightningModule (supervised + semi-supervised)
│   │   ├── losses.py           # Heatmap MSE and joint-weighted losses
│   │   └── callbacks.py        # MLflow visual prediction callback & checkpointing
│   ├── metrics/
│   │   └── pose_metrics.py     # PCK@0.2, MPJPE, heatmap peak decoders
│   ├── serving/
│   │   ├── predictor.py        # PoseInferenceEngine (BentoML/ckpt loading & batched inference)
│   │   ├── schemas.py          # Pydantic v2 API request/response schemas
│   │   └── service.py          # BentoML InBedPoseService with adaptive dynamic batching
│   └── utils/
│       ├── config.py           # YAML loader
│       └── visualization.py    # Matplotlib skeletal overlay rendering
└── scripts/
    ├── train.py                # Standalone training script with MLflow logging
    ├── tune.py                 # Standalone Optuna HPO runner with pruning
    ├── analyze_hpo.py          # Parameter importance and >=3% delta verification
    ├── evaluate.py             # Model checkpoint evaluation
    ├── export_model.py         # Package HW1 checkpoint into BentoML Model Store
    ├── client_example.py       # Example HTTP client for BentoML Pose API
    └── benchmark_serving.py    # Asyncio+httpx dynamic batching load benchmark
```

### Key Interfaces & Contracts
1. **`SLPDataModule`**:
   - `setup(stage)`: Initializes datasets.
   - `train_dataloader()`: Returns either standard DataLoader or `CombinedLoader` with `labeled` and `unlabeled` batches.
2. **`InBedPoseLightningModule`**:
   - `training_step(batch, batch_idx)`: Computes labeled heatmap MSE loss (+ optional EMA teacher pseudo-label loss on unlabeled data).
   - `validation_step(batch, batch_idx)`: Decodes predicted heatmaps to coordinates, computes `val_loss`, `val_pck`, and `val_mpjpe`.
3. **MLflow Tracking**:
   - Connected via `MLFlowLogger` with local SQLite storage `sqlite:///mlflow.db`.
