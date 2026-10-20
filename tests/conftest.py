"""Shared pytest fixtures."""

import os
from pathlib import Path

import duckdb
import pytest
import yaml

from opspilot.config import WAREHOUSE_DIR
from opspilot.features.load import load_features


@pytest.fixture(scope="session")
def warehouse_path() -> Path:
    """The warehouse the tests read: the one dbt builds on data/sample, unless overridden.

    Set OPSPILOT_TEST_DB=warehouse/opspilot.duckdb to run the same tests on the full data.
    """
    path = Path(os.environ.get("OPSPILOT_TEST_DB", WAREHOUSE_DIR / "opspilot_ci.duckdb"))
    if not path.exists():
        pytest.fail(f"{path} not found. Build it with `make dbt features DBT_TARGET=ci`.")
    return path


@pytest.fixture(scope="session")
def warehouse(warehouse_path):
    con = duckdb.connect(str(warehouse_path), read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def dbt_vars() -> dict:
    return yaml.safe_load((WAREHOUSE_DIR / "dbt_project.yml").read_text())["vars"]


@pytest.fixture(scope="session")
def features(warehouse_path):
    """The model input table plus outcome_known_at, as load_features returns it."""
    return load_features(warehouse_path)
