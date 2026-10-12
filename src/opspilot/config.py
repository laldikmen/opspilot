"""Paths, split dates and random seed: the one place these are defined."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
RAW_CHECKSUMS = DATA_DIR / "raw.sha256"

WAREHOUSE_DIR = REPO_ROOT / "warehouse"
DUCKDB_PATH = Path(os.environ.get("OPSPILOT_DUCKDB_PATH", WAREHOUSE_DIR / "opspilot.duckdb"))

REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Pinned to a dataset version so a re-download fetches the same files.
KAGGLE_DATASET = "olistbr/brazilian-ecommerce/versions/2"

RANDOM_SEED = 42
