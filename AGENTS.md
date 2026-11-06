# AGENTS.md - Master Instructions & Context for AI Assistants

Welcome to the **inbed-pose-mlops** repository! This file serves as the primary entry point, high-level context map, and deterministic rule routing matrix for AI coding assistants.

---

## 🏛️ Project Architecture & Overview

**inbed-pose-mlops** is a production-grade machine learning and MLOps repository for In-Bed Human Pose Estimation on the Simultaneously-collected Multimodal Lying Pose (**SLP**) dataset (VIP Cup 2021). The system trains 2D keypoint heatmap regression models (`ResNetPose`) across multimodal inputs (`RGB`, `IR`) under varying levels of blanket occlusion (`uncover`, `cover1`, `cover2`).
- **Core ML Framework**: PyTorch Lightning (`SLPDataModule` + `InBedPoseLightningModule`).
- **Training Paradigms**: Supervised baseline + SOTA semi-supervised Teacher-Student self-training utilizing unlabeled covered examples.
- **Model Monitoring & Logging**: MLflow tracking dashboard (`mlflow ui`) logging scalar metrics (`val_pck`, `val_mpjpe`, losses) and visual prediction heatmaps.
- **Hyperparameter Optimization**: Optuna HPO study with pruning callback achieving $\ge 3\%$ validation PCK gain.
- **Compute Execution**: Optimized for local GPU (RTX 3050 4GB) with remote GPU runner for Kaggle dual Tesla T4s.

---

## 🚦 Mandatory Rule Routing Matrix

Before writing or modifying any code, identify your target area and **read the corresponding rule file FIRST**:

| When working on / modifying... | Target Paths / Globs | Mandatory File to Read FIRST | Key Invariants & Pitfalls to Check |
| :--- | :--- | :--- | :--- |
| **All Tasks / Source Control** | All paths (`**`) | [`.agents/rules/git_and_pr_standards.md`](.agents/rules/git_and_pr_standards.md)<br>[`.agents/skills/git-pr-workflow/SKILL.md`](.agents/skills/git-pr-workflow/SKILL.md) | • **Branch FIRST**: Check branch & branch out (`git checkout -b ...`) BEFORE editing<br>• Atomic Conventional Commits (`feat(scope): ...`)<br>• Use `.tmp_pr_body.md` for PRs |
| **Code Quality, Linting, Typing** | `**/*.py`, `pyproject.toml`, `mypy.ini` | [`.agents/rules/code_quality_and_ci_standards.md`](.agents/rules/code_quality_and_ci_standards.md) | • Ruff lint & format check with 0 warnings<br>• Strict Mypy type-hints on all signatures<br>• Comprehensive pytest coverage |
| **Data Pipelines & Datasets** | `src/data/**`, `configs/**` | [`.agents/rules/ml_training_standards.md`](.agents/rules/ml_training_standards.md)<br>[`.agents/context/project_context.md`](.agents/context/project_context.md) | • Modalities (`RGB`, `IR`), covers (`uncover`, `cover1`, `cover2`)<br>• Gaussian heatmap encoding ($\sigma$) & joint visibility masking |
| **Models & PyTorch Lightning** | `src/models/**`, `src/training/**` | [`.agents/rules/ml_training_standards.md`](.agents/rules/ml_training_standards.md)<br>[`.agents/context/architecture_overview.md`](.agents/context/architecture_overview.md) | • Decoupled LightningModule with `self.log()`<br>• EMA Teacher with `requires_grad=False`<br>• PCK@0.2 evaluation metric invariance |
| **HPO & Tuning Scripts** | `scripts/tune.py`, `configs/hpo.yaml` | [`.agents/rules/ml_training_standards.md`](.agents/rules/ml_training_standards.md) | • `PyTorchLightningPruningCallback`<br>• Nested MLflow runs for Optuna trials<br>• Assert $\ge 3\%$ metric gain over baseline |

---

## 🔄 Operational Phase Gates

Every task must progress sequentially through these 5 lifecycle gates:

```
[ Gate 1: Rule Intake & Branch Creation ] ➔ [ Gate 2: Implementation ] ➔ [ Gate 3: Local CI Verification ] ➔ [ Gate 4: Context Maintenance ] ➔ [ Gate 5: Git Commit & PR Protocol ]
```

1. **Gate 1: Rule Intake & Branch Creation (MANDATORY)**:
   - Check current branch (`git branch --show-current`). If on `main`, pull latest and branch out (`git checkout -b <type>/<description>`) BEFORE editing.
   - Read required rule files from the *Rule Routing Matrix*.
2. **Gate 2: Implementation**:
   - Write code adhering strictly to repository standards, preserving docstrings, and enforcing complete type hints.
3. **Gate 3: Local CI Verification**:
   - Execute: `ruff check .`, `ruff format --check .`, `mypy src scripts`, `pytest tests/`.
4. **Gate 4: Context Self-Maintenance**:
   - Follow [`.agents/skills/agent-maintenance/SKILL.md`](.agents/skills/agent-maintenance/SKILL.md) if models, configs, or roadmap evolve.
5. **Gate 5: Git Commit & PR Protocol**:
   - Stage and commit with atomic Conventional Commits (`feat(scope): ...`); push branch, monitor CI, and create PR via temporary `.tmp_pr_body.md`.

---

## ⚙️ Core CLI Tools & Build Commands

Always run these commands from the repository root:

| Purpose | Working Directory | Command |
| :--- | :--- | :--- |
| **Lint & Format Check** | Root (`/`) | `ruff check . && ruff format --check .` |
| **Auto-Format Code** | Root (`/`) | `ruff format . && ruff check --fix .` |
| **Static Type Check** | Root (`/`) | `mypy src scripts --config-file mypy.ini` |
| **Run Unit Tests** | Root (`/`) | `pytest tests/ -v` |
| **Train Baseline Model** | Root (`/`) | `python scripts/train.py --config configs/baseline.yaml` |
| **Fast Dev Run (Smoke Test)** | Root (`/`) | `python scripts/train.py --config configs/baseline.yaml --fast_dev_run` |
| **Launch MLflow UI** | Root (`/`) | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| **Run Optuna HPO Study** | Root (`/`) | `python scripts/tune.py --config configs/hpo.yaml --n_trials 20` |
| **Analyze HPO & Plot** | Root (`/`) | `python scripts/analyze_hpo.py` |

---

## 🚨 Operational Boundaries & Escalation

- **Always**:
  - Check current branch and ensure you are on a dedicated branch (`git checkout -b ...`) BEFORE editing or creating files.
  - Run full local verification commands (`ruff`, `mypy`, `pytest`) prior to pushing.
  - Follow [git-pr-workflow](.agents/skills/git-pr-workflow/SKILL.md) for branch naming, atomic commits, and PR creation.
- **Never**:
  - Never write, modify, or stage code directly on `main` branch.
  - Never commit `.env` files, credentials, MLflow databases (`mlflow.db`), or large checkpoint weights.
  - Never bypass compiler or linter errors using `# type: ignore` or `# noqa` unless formally documented and verified.
