-- Fails if any seller-history row uses an order delivered at or after t0.

select
    order_id,
    t0,
    seller_last_delivery_used_at
from {{ ref('int_seller_history_asof') }}
where seller_last_delivery_used_at >= t0
