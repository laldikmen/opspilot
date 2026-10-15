-- Fails if any route-history row uses a delivery at or after the order's t0.

select
    order_id,
    t0,
    route_last_delivery_used_at
from {{ ref('int_route_history_asof') }}
where route_last_delivery_used_at >= t0
