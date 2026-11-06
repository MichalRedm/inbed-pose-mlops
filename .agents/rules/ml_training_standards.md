# Machine Learning & Training Standards

> [!IMPORTANT]
> **Trigger Paths**: `src/data/**`, `src/models/**`, `src/training/**`, `scripts/**`, `configs/**`
> **When to Read**: MUST be read before modifying dataset loaders, model architectures, Lightning modules, or optimization scripts.

This document establishes engineering invariants for PyTorch Lightning, MLflow monitoring, and Optuna HPO in `inbed-pose-mlops`.

---

## 1. PyTorch Lightning Modularity

### Separation of Concerns
1. **Data Logic (`src/data/datamodule.py`)**:
   - Encapsulated strictly within `pl.LightningDataModule`.
   - Never inject model inference or optimizer logic into `SLPDataModule`.
   - Handles `setup()`, batch collations, and DataLoader creation.
2. **Model & Science Logic (`src/training/lightning_module.py`)**:
   - Encapsulated strictly within `InBedPoseLightningModule(pl.LightningModule)`.
   - Defines `forward()`, `training_step()`, `validation_step()`, `test_step()`, and `configure_optimizers()`.
   - All metrics (`train_loss`, `val_loss`, `val_pck`, `val_mpjpe`) MUST be logged via `self.log()`.
3. **Execution Logic (`scripts/train.py`, `scripts/tune.py`)**:
   - Thin CLI orchestrators that assemble the config, DataModule, LightningModule, and `pl.Trainer`.

---

## 2. Training Paradigms: Supervised vs. SOTA Semi-Supervised

### Supervised Mode (`mode="supervised"`)
- Trains the student network on labeled batches (`uncover` images with ground-truth joint coordinates).
- Loss: Mean Squared Error (MSE) between predicted heatmaps and Gaussian ground-truth heatmaps generated with standard deviation $\sigma$.

### Semi-Supervised Self-Training (`mode="semi_supervised"`)
- Employs an Exponential Moving Average (EMA) Teacher network ($t_{\text{param}} = \alpha \cdot t_{\text{param}} + (1-\alpha) \cdot s_{\text{param}}$).
- The Teacher generates pseudo-labels on weak views of unannotated covered samples (`cover1`, `cover2`).
- Pseudo-labels are gated by joint-specific confidence thresholds.
- The Student receives strong augmentations (patch cutout/erasing) and minimizes consistency loss against the gated pseudo-labels.

---

## 3. Model Monitoring & MLflow Integration

- Every training run connects to an `MLFlowLogger`:
  ```python
  from lightning.pytorch.loggers import MLFlowLogger

  mlf_logger = MLFlowLogger(
      experiment_name="inbed-pose-estimation",
      tracking_uri="sqlite:///mlflow.db",
  )
  ```
- Visual Prediction Callbacks:
  - At validation epoch end, sample predictions (skeletal connections overlaid on images) are generated and saved as MLflow image artifacts.
- Checkpointing:
  - `ModelCheckpoint` monitors `val_pck` (`mode="max"`).

---

## 4. Hyperparameter Optimization with Optuna

- Run via `scripts/tune.py`.
- Incorporates `PyTorchLightningPruningCallback(trial, monitor="val_pck")` to terminate unpromising runs early.
- Each Optuna trial is logged as a child MLflow run under a parent study experiment (`nested=True`).
- Target: $\ge 3\%$ relative/absolute gain on validation PCK@0.2 over baseline.
