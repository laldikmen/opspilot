"""Read the dbt mart ml_late_delivery_features into pandas."""

from pathlib import Path

import duckdb
import pandas as pd

from opspilot.config import DUCKDB_PATH

MART = "ml_late_delivery_features"

ID_COLUMNS = ["order_id", "purchase_date", "t0"]
TARGET = "is_late"

NUMERIC_FEATURES = [
    "promised_days",
    "promise_slack_days",
    "approval_delay_hours",
    "purchase_dow",
    "purchase_hour",
    "total_price",
    "total_freight",
    "freight_ratio",
    "n_items",
    "n_sellers",
    "n_products",
    "total_weight_g",
    "total_volume_cm3",
    "same_state",
    "distance_km",
    "route_mean_delivery_days",
    "route_late_rate_smoothed",
    "installments",
    "n_payment_methods",
    "seller_past_delivered",
    "seller_late_rate_smoothed",
    "seller_days_active",
    "seller_mean_handover_days",
]
CATEGORICAL_FEATURES = ["main_category", "customer_state", "seller_state", "payment_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_features(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """One row per labeled, uncensored order, oldest first."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.sql(f"select * from {MART} order by t0, order_id").df()
    finally:
        con.close()
