.DEFAULT_GOAL := help
.PHONY: help setup data sample dbt features train evaluate api all lint test

ifneq (,$(wildcard .env))
include .env
export
endif

# Import the package from src/ directly instead of through the editable install.
# In an iCloud-synced folder macOS marks .venv as hidden, and Python then skips
# the .pth file the editable install relies on.
export PYTHONPATH := $(CURDIR)/src

help:
	@echo "make setup     install dependencies and git hooks"
	@echo "make data      download the Olist CSVs into data/raw and verify checksums"
	@echo "make sample    rebuild data/sample (the ~2k-order sample CI runs on)"
	@echo "make dbt       build and test the DuckDB warehouse on data/raw"
	@echo "make features  build the model input table                    (Week 3)"
	@echo "make train     train baselines and the final model            (Weeks 4-5)"
	@echo "make evaluate  score the test set and write reports/          (Week 7)"
	@echo "make api       run the prediction API                         (Week 8)"
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
# `make dbt DBT_TARGET=ci` builds on data/sample instead of data/raw, as CI does.
DBT_TARGET ?= dev
dbt:
	cd warehouse && uv run dbt build --target $(DBT_TARGET)

# Later weeks replace these stubs. They fail loudly so `make all` cannot
# look green before the pipeline exists.
features:
	@echo "make features: not implemented yet (Week 3)"; exit 1

train:
	@echo "make train: not implemented yet (Weeks 4-5)"; exit 1

evaluate:
	@echo "make evaluate: not implemented yet (Week 7)"; exit 1

api:
	@echo "make api: not implemented yet (Week 8)"; exit 1

all: data dbt features train evaluate

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest
