"""Build data/sample: about 2,000 orders spread across all months, with every related row.

Run with `make sample` after `make data`. The sample is committed and is what CI
and the dbt `ci` target run on, so it must come out identical every time: orders
are picked by a hash of their id, never by a random number generator.

The files keep the raw names and columns, and every value is copied as text, so
the same dbt sources read either folder.
"""

import duckdb

from opspilot.config import RANDOM_SEED, RAW_DIR, SAMPLE_DIR

# 21 full months at 95 orders each, plus every order of the four near-empty months.
ORDERS_PER_MONTH = 95
# The raw file has a median of 29 points per zip prefix; a few are enough to test with.
GEO_POINTS_PER_ZIP = 3

ORDERS = "olist_orders_dataset"
ORDER_ITEMS = "olist_order_items_dataset"
CUSTOMERS = "olist_customers_dataset"
SELLERS = "olist_sellers_dataset"
PRODUCTS = "olist_products_dataset"
PAYMENTS = "olist_order_payments_dataset"
GEOLOCATION = "olist_geolocation_dataset"
CATEGORY_TRANSLATION = "product_category_name_translation"


def raw(name: str) -> str:
    return f"read_csv('{RAW_DIR / name}.csv', header = true, all_varchar = true)"


def sample_queries() -> dict[str, str]:
    """One query per output file. Each selects from the raw CSVs and the tables before it."""
    return {
        ORDERS: f"""
            select * exclude (month_rank)
            from (
                select
                    *,
                    row_number() over (
                        partition by left(order_purchase_timestamp, 7)  -- 'YYYY-MM'
                        order by md5(order_id || '{RANDOM_SEED}')
                    ) as month_rank
                from {raw(ORDERS)}
            )
            where month_rank <= {ORDERS_PER_MONTH}
            order by order_purchase_timestamp, order_id
        """,
        ORDER_ITEMS: f"""
            select * from {raw(ORDER_ITEMS)}
            where order_id in (select order_id from {ORDERS})
            order by order_id, cast(order_item_id as integer)
        """,
        PAYMENTS: f"""
            select * from {raw(PAYMENTS)}
            where order_id in (select order_id from {ORDERS})
            order by order_id, cast(payment_sequential as integer)
        """,
        CUSTOMERS: f"""
            select * from {raw(CUSTOMERS)}
            where customer_id in (select customer_id from {ORDERS})
            order by customer_id
        """,
        SELLERS: f"""
            select * from {raw(SELLERS)}
            where seller_id in (select seller_id from {ORDER_ITEMS})
            order by seller_id
        """,
        PRODUCTS: f"""
            select * from {raw(PRODUCTS)}
            where product_id in (select product_id from {ORDER_ITEMS})
            order by product_id
        """,
        GEOLOCATION: f"""
            select * exclude (zip_rank)
            from (
                select
                    *,
                    row_number() over (
                        partition by geolocation_zip_code_prefix
                        order by geolocation_lat, geolocation_lng, geolocation_city
                    ) as zip_rank
                from {raw(GEOLOCATION)}
                where geolocation_zip_code_prefix in (
                    select customer_zip_code_prefix from {CUSTOMERS}
                    union
                    select seller_zip_code_prefix from {SELLERS}
                )
            )
            where zip_rank <= {GEO_POINTS_PER_ZIP}
            order by geolocation_zip_code_prefix, zip_rank
        """,
        CATEGORY_TRANSLATION: f"select * from {raw(CATEGORY_TRANSLATION)}",
    }


def build_sample() -> dict[str, int]:
    """Write the sample CSVs and return the row count of each."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    rows = {}
    for name, query in sample_queries().items():
        con.execute(f"create temp table {name} as {query}")
        con.execute(f"copy {name} to '{SAMPLE_DIR / name}.csv' (header, delimiter ',')")
        rows[name] = con.execute(f"select count(*) from {name}").fetchone()[0]
    return rows


def main() -> None:
    rows = build_sample()
    print(f"Wrote {len(rows)} files to {SAMPLE_DIR}")
    for name, n in rows.items():
        size_kb = (SAMPLE_DIR / f"{name}.csv").stat().st_size / 1024
        print(f"{n:>7,} rows  {size_kb:>7,.0f} KB  {name}.csv")


if __name__ == "__main__":
    main()
