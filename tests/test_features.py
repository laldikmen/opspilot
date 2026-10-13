"""Spot checks on feature values in the model input table."""

from math import asin, cos, radians, sin, sqrt

import pandas as pd
import pytest

from opspilot.features.load import MART


def haversine_km(lat1, lng1, lat2, lng2):
    a = (
        sin(radians(lat2 - lat1) / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(radians(lng2 - lng1) / 2) ** 2
    )
    return 2 * 6371.0 * asin(sqrt(a))


def test_distance_matches_haversine_and_is_null_without_a_centroid(warehouse):
    rows = warehouse.sql(f"""
        select features.distance_km,
               customer_point.latitude as customer_lat, customer_point.longitude as customer_lng,
               seller_point.latitude as seller_lat, seller_point.longitude as seller_lng
        from {MART} as features
        inner join fct_orders using (order_id)
        inner join int_order_items_agg using (order_id)
        inner join stg_customers using (customer_id)
        inner join stg_sellers on stg_sellers.seller_id = int_order_items_agg.primary_seller_id
        left join int_zip_centroids as customer_point
            on customer_point.zip_code_prefix = stg_customers.customer_zip_code_prefix
        left join int_zip_centroids as seller_point
            on seller_point.zip_code_prefix = stg_sellers.seller_zip_code_prefix
        order by features.order_id
    """).df()
    points = ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]
    has_points = rows[points].notna().all(axis=1)

    assert rows.distance_km.isna().equals(~has_points)
    # Brazil is about 4,400 km across; anything longer is a bad point or a bad formula.
    assert rows.distance_km.max() < 4500

    for row in rows[has_points].head(25).itertuples():
        expected = haversine_km(row.customer_lat, row.customer_lng, row.seller_lat, row.seller_lng)
        assert row.distance_km == pytest.approx(expected, abs=1e-6)


def test_features_have_sensible_ranges(warehouse):
    features = warehouse.sql(f"select * from {MART}").df()
    assert features.purchase_dow.between(1, 7).all()
    assert features.purchase_hour.between(0, 23).all()
    assert features.seller_late_rate_smoothed.between(0, 1).all()
    assert (features.seller_days_active >= 0).all()
    assert (features.t0 >= pd.to_datetime(features.purchase_date)).all()
