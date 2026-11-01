.DEFAULT_GOAL := help
.PHONY: help setup data sample dbt features train mlflow evaluate api all lint test walk-forward retrain

ifneq (,$(wildcard .env))
include .env
export
endif

# Import the package from src/ directly instead of through the editable install.
# In an iCloud-synced folder macOS marks .venv as hidden, and Python then skips
# the .pth file the editable install relies on.
export PYTHONPATH := $(CURDIR)/src

# No usage data sent from MLflow (dbt's is switched off in dbt_project.yml).
export MLFLOW_DISABLE_TELEMETRY := true

help:
	@echo "make setup     install dependencies and git hooks"
	@echo "make data      download the Olist CSVs into data/raw and verify checksums"
	@echo "make sample    rebuild data/sample (the ~2k-order sample CI runs on)"
	@echo "make dbt       build and test the staging layer of the DuckDB warehouse"
	@echo "make features  build and test the feature tables, with the leakage check"
	@echo "make train     fit the baselines and tune XGBoost on the folds and validation"
	@echo "make mlflow    open the MLflow UI on http://127.0.0.1:5001"
	@echo "make evaluate  score the test set once and write reports/evaluation.md"
	@echo "make walk-forward  v1.1: retrain month by month and compare models"
	@echo "make retrain   v1.1: refit the served model (add ARGS=\"--as-of YYYY-MM-DD\")"
	@echo "make api       run the prediction API on http://127.0.0.1:8000"
	@echo "make all       data -> dbt -> features -> train -> evaluate"
	@echo "make lint      ruff check + format check"
	@echo "make test      pytest"

setup:
	uv sync
	uv run pre-commit install

data:
	uv run python -m opspilot.data.download

sample:
	uv run python -m opspilot.data.make_sample

# dbt runs from warehouse/: the paths in profiles.yml and _sources.yml are relative to it.
# Add DBT_TARGET=ci to build on data/sample instead of data/raw, as CI does.
DBT_TARGET ?= dev
dbt:
	cd warehouse && uv run dbt build --target $(DBT_TARGET) --select staging

features:
	cd warehouse && uv run dbt build --target $(DBT_TARGET) --select intermediate marts

train:
	uv run python -m opspilot.models.baselines
	uv run python -m opspilot.models.train

# Port 5000 is taken by AirPlay on macOS.
mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# Scores the test set: run once, after reports/test_protocol.md is committed.
evaluate:
	uv run python -m opspilot.models.evaluate

walk-forward:
	uv run python -m opspilot.models.walk_forward

retrain:
	uv run python -m opspilot.models.retrain $(ARGS)

# The API reads artifacts/ and the warehouse, so run `make features train retrain` first.
api:
	uv run uvicorn opspilot.api.main:app --port 8000

all: data dbt features train evaluate

lint:
	uv run ruff check .
	uv run ruff format --check .

# The tests read the warehouse built on data/sample, so build it first.
test:
	$(MAKE) dbt features DBT_TARGET=ci
	uv run pytest
