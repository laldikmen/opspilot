-- Fails if any seller-history row uses an order delivered at or after t0, or a
-- carrier handover at or after t0.

select
    order_id,
    t0,
    seller_last_delivery_used_at as used_at
from {{ ref('int_seller_history_asof') }}
where seller_last_delivery_used_at >= t0

union all

select
    order_id,
    t0,
    seller_last_handover_used_at as used_at
from {{ ref('int_seller_handover_asof') }}
where seller_last_handover_used_at >= t0
