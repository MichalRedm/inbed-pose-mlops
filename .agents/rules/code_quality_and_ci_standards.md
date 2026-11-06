# Code Quality & CI Standards

> [!IMPORTANT]
> **Trigger Paths**: `**/*.py`, `pyproject.toml`, `mypy.ini`, `.github/workflows/**`
> **When to Read**: MUST be read before writing or refactoring Python code, configuring static analysis, or writing automated tests.

This document defines the Python coding conventions, static analysis gates, and test suite requirements for `inbed-pose-mlops`.

---

## 1. Static Analysis & Linting (Ruff)

We use **Ruff** for high-speed linting and code formatting:
- Target Python Version: Python 3.10+
- Line Length: 88 characters
- Rule Sets: `E` (pycodestyle errors), `F` (Pyflakes), `I` (isort), `B` (flake8-bugbear), `UP` (pyupgrade).
- Formatting: 4-space indentation, double quotes, trailing commas in multiline collections.

### Commands:
```bash
# Check code for lint violations
python -m ruff check .

# Check formatting compliance
python -m ruff format --check .

# Auto-fix lint and auto-format
python -m ruff check --fix .
python -m ruff format .
```

---

## 2. Static Typing & Type Annotations (Mypy)

- All public functions, methods, class attributes, and module interfaces **must be strictly type-annotated**.
- Do not use `Any` where a specific type, union, or generic can be expressed.
- Use Python standard typing imports (`typing.Dict`, `typing.List`, `typing.Optional`, `typing.Tuple`, `typing.Union`, `typing.cast`).
- Tensor shapes should be documented in docstrings (e.g. `(B, 14, H, W)`).
- Run Mypy without ignoring warnings:
```bash
python -m mypy src scripts --config-file mypy.ini
```

---

## 3. Testing Standards (Pytest)

- Test Framework: `pytest`.
- Test Directory: `tests/`.
- Every feature or component added must have automated tests:
  - `test_data.py`: Verifies dataset indexing, modality/cover filtering, augmentations, and `SLPDataModule`.
  - `test_models.py`: Verifies model forward passes, parameter counts, and output heatmap dimensions.
  - `test_metrics.py`: Verifies PCK@0.2 and MPJPE calculations against synthetic inputs with known ground truth.
  - `test_lightning.py`: Verifies `fast_dev_run=True` execution of `InBedPoseLightningModule`.
- Tests must be deterministic and fast, relying on synthetic mock data fixtures in `tests/conftest.py` rather than demanding raw multi-gigabyte datasets during CI.
