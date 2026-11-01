"""The handover table only holds what was known then, and only open orders."""

import pandas as pd
import pytest

from opspilot.features.load import HANDOVER_FEATURES, HANDOVER_MART, load_handover_features
from opspilot.models.handover import run
from opspilot.models.train import DEFAULT_PARAMS


@pytest.fixture(scope="module")
def handover(warehouse_path):
    return load_handover_features(warehouse_path)


def test_table_has_exactly_the_agreed_columns(warehouse):
    columns = {row[0] for row in warehouse.sql(f"describe {HANDOVER_MART}").fetchall()}
    assert columns == {"order_id", "scored_at", *HANDOVER_FEATURES}


def test_only_open_orders_are_scored(warehouse, handover):
    orders = warehouse.sql("""
        select order_id, delivered_at, estimated_delivery_date, t0 from fct_orders
    """).df()
    rows = handover.merge(orders, on="order_id")
    assert (rows.delivered_at > rows.scored_at).all()
    assert (rows.scored_at.dt.normalize() <= pd.to_datetime(rows.estimated_delivery_date)).all()
    assert (rows.scored_at >= rows.t0).all()


def test_features_match_a_recomputation(warehouse, handover):
    stg = warehouse.sql("""
        select order_id, carrier_handover_at, estimated_delivery_date from stg_orders
    """).df()
    mart = warehouse.sql("""
        select order_id, t0, seller_mean_handover_days, route_mean_delivery_days
        from ml_late_delivery_features
    """).df()
    rows = handover.merge(stg, on="order_id").merge(mart, on="order_id")
    start = rows[["carrier_handover_at", "t0"]].max(axis=1)
    delay = (start - rows.t0).dt.total_seconds() / 86400
    left = (pd.to_datetime(rows.estimated_delivery_date) - start).dt.total_seconds() / 86400
    assert delay.to_numpy() == pytest.approx(rows.handover_delay_days.to_numpy())
    assert left.to_numpy() == pytest.approx(rows.days_left.to_numpy())
    usual = rows.route_mean_delivery_days - rows.seller_mean_handover_days
    both = rows.carrier_slack_days.notna()
    assert (left - usual)[both].to_numpy() == pytest.approx(
        rows.carrier_slack_days[both].to_numpy()
    )


def test_handover_features_stay_out_of_the_approval_table(warehouse):
    columns = {row[0] for row in warehouse.sql("describe ml_late_delivery_features").fetchall()}
    assert not columns & set(HANDOVER_FEATURES)


def test_both_moments_score_the_same_orders(features, handover):
    scores = run(features, handover, DEFAULT_PARAMS)
    assert scores[["at_approval", "at_handover"]].notna().all().all()
