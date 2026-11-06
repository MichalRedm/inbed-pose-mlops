.PHONY: help install install-dev lint format typecheck test fast-dev train tune mlflow-ui clean

PYTHON ?= python

help:
	@echo "Available make targets:"
	@echo "  install      - Install runtime dependencies"
	@echo "  install-dev  - Install runtime and development dependencies"
	@echo "  lint         - Check code quality with ruff"
	@echo "  format       - Auto-format code with ruff"
	@echo "  typecheck    - Run static type checking with mypy"
	@echo "  test         - Run test suite with pytest"
	@echo "  fast-dev     - Run a fast dev sanity check (1 batch train & val)"
	@echo "  train        - Run baseline model training"
	@echo "  tune         - Run Optuna HPO study"
	@echo "  mlflow-ui    - Launch local MLflow dashboard"
	@echo "  clean        - Remove build, test, and cache artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy src scripts --config-file mypy.ini

test:
	$(PYTHON) -m pytest tests/ -v

fast-dev:
	$(PYTHON) scripts/train.py --config configs/baseline.yaml --fast_dev_run

train:
	$(PYTHON) scripts/train.py --config configs/baseline.yaml

tune:
	$(PYTHON) scripts/tune.py --config configs/hpo.yaml

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
