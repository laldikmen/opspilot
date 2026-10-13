"""The committed sample in data/sample is what CI builds the warehouse on."""

import duckdb
import pytest

from opspilot.config import SAMPLE_DIR
from opspilot.data.make_sample import ORDERS, sample_queries


def read(name: str) -> str:
    return f"read_csv('{SAMPLE_DIR / name}.csv', header = true, all_varchar = true)"


@pytest.mark.parametrize("name", sample_queries())
def test_every_sample_file_is_committed(name):
    assert (SAMPLE_DIR / f"{name}.csv").exists()


def test_sample_is_about_2000_orders_spread_across_all_months():
    n_orders, n_months = duckdb.sql(f"""
        select count(*), count(distinct left(order_purchase_timestamp, 7))
        from {read(ORDERS)}
    """).fetchone()
    assert 1800 <= n_orders <= 2200
    assert n_months == 25  # 2016-09 to 2018-10; there are no orders in 2016-11


def test_sample_keeps_leading_zeros_in_zip_prefixes():
    lengths = duckdb.sql(f"""
        select distinct length(customer_zip_code_prefix) from {read("olist_customers_dataset")}
    """).fetchall()
    assert lengths == [(5,)]
