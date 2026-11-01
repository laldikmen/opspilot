"""Read the dbt mart ml_late_delivery_features into pandas."""

from pathlib import Path

import duckdb
import pandas as pd

from opspilot.config import DUCKDB_PATH

MART = "ml_late_delivery_features"

ID_COLUMNS = ["order_id", "purchase_date", "t0"]
TARGET = "is_late"
# When the label became known. Decides which rows a model may learn from; never a feature.
OUTCOME_KNOWN_AT = "outcome_known_at"

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
    "seller_late_rate_smoothed",
    "seller_mean_handover_days",
]
CATEGORICAL_FEATURES = ["main_category", "customer_state", "seller_state", "payment_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Known only once the seller has handed the order to the carrier (v1.1).
HANDOVER_MART = "ml_handover_features"
HANDOVER_FEATURES = [
    "handover_delay_days",
    "handover_vs_usual_days",
    "days_left",
    "carrier_slack_days",
]


def load_features(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """One row per labeled, uncensored order, oldest first, with when its label became known."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.sql(f"""
            select features.*, orders.{OUTCOME_KNOWN_AT}
            from {MART} as features
            inner join fct_orders as orders using (order_id)
            order by features.t0, features.order_id
        """).df()
    finally:
        con.close()


def load_handover_features(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """The orders still open at carrier handover, with the features known by then."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.sql(f"select * from {HANDOVER_MART}").df()
    finally:
        con.close()
