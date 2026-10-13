"""Forbidden columns are absent; seller history uses only outcomes known before t0."""

import pandas as pd
import pytest

from opspilot.config import RANDOM_SEED
from opspilot.features.load import FEATURES, ID_COLUMNS, MART, TARGET

# Recorded after t0, or built from the label (docs/decisions/0005-leakage-rules.md).
FORBIDDEN_COLUMNS = {
    "carrier_handover_at",
    "delivered_at",
    "delivery_days",
    "order_status",
    "review_score",
    "review_comment",
    "seller_late_rate",
    "seller_past_late",
    "seller_last_delivery_used_at",
}

N_ORDERS = 50


def test_feature_table_has_exactly_the_agreed_columns(warehouse):
    # A new column fails here first, which forces the question "was it known at t0?"
    columns = [row[0] for row in warehouse.sql(f"describe {MART}").fetchall()]
    assert sorted(columns) == sorted(ID_COLUMNS + [TARGET] + FEATURES)
    assert len(FEATURES) == 23


def test_no_forbidden_column_is_a_feature(warehouse):
    columns = {row[0] for row in warehouse.sql(f"describe {MART}").fetchall()}
    assert not columns & FORBIDDEN_COLUMNS
    assert not [c for c in columns if "review" in c]


def test_seller_history_matches_brute_force(warehouse, dbt_vars):
    """Recompute the history of 50 orders in plain pandas and compare with the dbt model."""
    orders = warehouse.sql("""
        select order_id, order_status, purchased_at, approved_at, delivered_at,
               estimated_delivery_date
        from stg_orders
    """).df()
    order_sellers = warehouse.sql("select distinct order_id, seller_id from stg_order_items").df()
    modeled = warehouse.sql(f"""
        select order_id, seller_id, seller_past_delivered, seller_past_late,
               seller_late_rate_smoothed
        from int_seller_history_asof
        where order_id in (select order_id from {MART})
        order by order_id
    """).df()

    # The rules, written out independently of the SQL.
    orders["t0"] = orders.approved_at.fillna(orders.purchased_at)
    orders["is_late"] = orders.delivered_at.dt.normalize() > pd.to_datetime(
        orders.estimated_delivery_date
    )
    has_outcome = (orders.order_status == "delivered") & orders.delivered_at.notna()
    outcomes = orders[has_outcome].merge(order_sellers, on="order_id")
    t0 = orders.set_index("order_id").t0

    prior = dbt_vars["seller_prior_late_rate"]
    strength = dbt_vars["seller_prior_strength"]

    known_counts, purchased_counts = [], []
    for row in modeled.sample(N_ORDERS, random_state=RANDOM_SEED).itertuples():
        others = outcomes[
            (outcomes.seller_id == row.seller_id) & (outcomes.order_id != row.order_id)
        ]
        known = others[others.delivered_at < t0[row.order_id]]
        n, late = len(known), int(known.is_late.sum())

        assert row.seller_past_delivered == n, row.order_id
        assert row.seller_past_late == late, row.order_id
        assert row.seller_late_rate_smoothed == pytest.approx(
            (late + strength * prior) / (n + strength)
        )
        known_counts.append(n)
        purchased_counts.append(int((others.purchased_at < t0[row.order_id]).sum()))

    # The comparison only means something if some of the 50 orders have a history,
    # and if the wrong rule (purchased before t0) would have produced other numbers.
    assert sum(known_counts) > 0
    assert known_counts != purchased_counts
