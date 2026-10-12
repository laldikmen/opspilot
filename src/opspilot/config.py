"""Paths, split dates and random seed: the one place these are defined."""

import os
from datetime import date, timedelta
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

# --- Modelling window -------------------------------------------------------
# Measured in notebooks/01_eda.ipynb; reasons in docs/decisions/0003 and 0004.
# Change these only together with the ADRs.

# Last day with a normal volume of recorded deliveries. Another 86 trickle in
# up to 2018-10-17, but the extract effectively ends here.
OBSERVATION_END = date(2018, 8, 30)
# 99th percentile of purchase-to-delivery time, in days.
CENSORING_HORIZON_DAYS = 46
# Orders purchased after this date may not have had time to be delivered.
CENSORING_CUTOFF = OBSERVATION_END - timedelta(days=CENSORING_HORIZON_DAYS)

# Chronological split by purchase date; both ends inclusive.
SPLITS = {
    "train": (date(2017, 1, 1), date(2017, 12, 31)),
    "validation": (date(2018, 1, 1), date(2018, 3, 31)),
    "test": (date(2018, 4, 1), CENSORING_CUTOFF),
}
