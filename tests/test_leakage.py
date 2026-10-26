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
    "seller_last_handover_used_at",
    "route_last_delivery_used_at",
    "route_past_late",
    "outcome_known_at",
    # A calendar proxy rather than a seller trait (docs/decisions/0004).
    "seller_days_active",
}

N_ORDERS = 50


def test_feature_table_has_exactly_the_agreed_columns(warehouse):
    # A new column fails here first, which forces the question "was it known at t0?"
    columns = [row[0] for row in warehouse.sql(f"describe {MART}").fetchall()]
    assert sorted(columns) == sorted(ID_COLUMNS + [TARGET] + FEATURES)
    assert len(FEATURES) == 26


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


@pytest.fixture(scope="module")
def orders(warehouse):
    """Every order with t0, label, route and primary seller, worked out in pandas."""
    orders = warehouse.sql("""
        select stg.order_id, stg.order_status, stg.purchased_at, stg.approved_at,
               stg.carrier_handover_at, stg.delivered_at, stg.estimated_delivery_date,
               items.primary_seller_id as seller_id,
               sellers.seller_state || '>' || customers.customer_state as route
        from stg_orders as stg
        inner join int_order_items_agg as items using (order_id)
        inner join stg_sellers as sellers on sellers.seller_id = items.primary_seller_id
        inner join stg_customers as customers using (customer_id)
    """).df()
    orders["t0"] = orders.approved_at.fillna(orders.purchased_at)
    orders["labeled"] = (orders.order_status == "delivered") & orders.delivered_at.notna()
    orders["is_late"] = (
        orders.delivered_at.dt.normalize() > pd.to_datetime(orders.estimated_delivery_date)
    ).astype(int)
    orders["delivery_days"] = (orders.delivered_at - orders.t0).dt.total_seconds() / 86400
    orders["handover_days"] = (
        (orders.carrier_handover_at - orders.approved_at).dt.total_seconds() / 86400
    ).clip(lower=0)
    return orders.set_index("order_id", drop=False)


def checked_orders(warehouse, model: str) -> pd.DataFrame:
    return (
        warehouse.sql(f"""
            select * from {model}
            where order_id in (select order_id from {MART})
            order by order_id
        """)
        .df()
        .sample(N_ORDERS, random_state=RANDOM_SEED)
    )


def test_route_history_matches_brute_force(warehouse, dbt_vars, orders):
    strength = dbt_vars["history_prior_strength"]
    counts = []
    for row in checked_orders(warehouse, "int_route_history_asof").itertuples():
        me = orders.loc[row.order_id]
        known = orders[
            orders.labeled & (orders.delivered_at < me.t0) & (orders.order_id != row.order_id)
        ]
        route = known[known.route == me.route]
        n = len(route)

        assert row.route_past_delivered == n, row.order_id
        # Both sides are NaN for the first orders, when nothing has been delivered yet.
        assert row.route_mean_delivery_days == pytest.approx(
            (route.delivery_days.sum() + strength * known.delivery_days.mean()) / (n + strength),
            nan_ok=True,
        )
        assert row.route_late_rate_smoothed == pytest.approx(
            (route.is_late.sum() + strength * known.is_late.mean()) / (n + strength),
            nan_ok=True,
        )
        counts.append(n)
    assert sum(counts) > 0


def test_seller_handover_matches_brute_force(warehouse, dbt_vars, orders):
    strength = dbt_vars["history_prior_strength"]
    counts = []
    for row in checked_orders(warehouse, "int_seller_handover_asof").itertuples():
        me = orders.loc[row.order_id]
        known = orders[
            orders.handover_days.notna()
            & (orders.carrier_handover_at < me.t0)
            & (orders.order_id != row.order_id)
        ]
        seller = known[known.seller_id == me.seller_id]
        n = len(seller)

        assert row.seller_past_handovers == n, row.order_id
        assert row.seller_mean_handover_days == pytest.approx(
            (seller.handover_days.sum() + strength * known.handover_days.mean()) / (n + strength),
            nan_ok=True,
        )
        counts.append(n)
    assert sum(counts) > 0


def test_outcome_known_at_is_delivery_or_the_day_after_a_broken_promise(warehouse):
    orders = warehouse.sql("""
        select is_late, delivered_at, estimated_delivery_date, outcome_known_at
        from fct_orders where is_labeled
    """).df()
    on_time, late = orders[orders.is_late == 0], orders[orders.is_late == 1]
    assert (on_time.outcome_known_at == on_time.delivered_at).all()
    day_after = pd.to_datetime(late.estimated_delivery_date) + pd.Timedelta(days=1)
    assert (late.outcome_known_at == day_after).all()
    # A late order is known no later than it is delivered.
    assert (orders.outcome_known_at <= orders.delivered_at).all()
