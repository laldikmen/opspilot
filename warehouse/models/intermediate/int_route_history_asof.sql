-- One row per order: what was known at t0 about deliveries on its route, from
-- seller state to customer state, counting only orders delivered before t0.
--
-- Running totals plus an as-of join: each order picks up the totals at the last
-- delivery strictly before its t0.

{% set strength = var('history_prior_strength') %}

with orders as (

    select
        fct.order_id,
        fct.t0,
        fct.delivered_at,
        fct.is_late,
        fct.is_labeled,
        sellers.seller_state || '>' || customers.customer_state as route,
        -- Measured from t0, like promised_days, so the two can be compared.
        date_diff('second', fct.t0, fct.delivered_at) / 86400.0 as delivery_days
    from {{ ref('fct_orders') }} as fct
    inner join {{ ref('int_order_items_agg') }} as items on fct.order_id = items.order_id
    inner join {{ ref('stg_sellers') }} as sellers on items.primary_seller_id = sellers.seller_id
    inner join {{ ref('stg_customers') }} as customers on fct.customer_id = customers.customer_id

),

deliveries as (

    select * from orders where is_labeled

),

route_totals as (

    select
        route,
        delivered_at,
        sum(count(*)) over running as n,
        sum(sum(is_late)) over running as late,
        sum(sum(delivery_days)) over running as days
    from deliveries
    group by route, delivered_at
    window running as (partition by route order by delivered_at)

),

all_totals as (

    select
        delivered_at,
        sum(count(*)) over running as n,
        sum(sum(is_late)) over running as late,
        sum(sum(delivery_days)) over running as days
    from deliveries
    group by delivered_at
    window running as (order by delivered_at)

),

as_of_t0 as (

    select
        orders.order_id,
        orders.t0,
        coalesce(route_totals.n, 0) as route_n,
        coalesce(route_totals.late, 0) as route_late,
        coalesce(route_totals.days, 0) as route_days,
        route_totals.delivered_at as route_last_delivery_used_at,
        coalesce(all_totals.n, 0) as all_n,
        coalesce(all_totals.late, 0) as all_late,
        coalesce(all_totals.days, 0) as all_days,
        -- An order never counts toward its own history. A few are recorded as
        -- delivered before their own approval, so take them back out.
        cast(coalesce(orders.is_labeled and orders.delivered_at < orders.t0, false) as integer)
            as own,
        coalesce(orders.is_late, 0) as own_late,
        coalesce(orders.delivery_days, 0) as own_days
    from orders
    asof left join route_totals
        on orders.route = route_totals.route and orders.t0 > route_totals.delivered_at
    asof left join all_totals
        on orders.t0 > all_totals.delivered_at

),

excluding_self as (

    select
        order_id,
        t0,
        route_last_delivery_used_at,
        route_n - own as route_n,
        route_late - own * own_late as route_late,
        route_days - own * own_days as route_days,
        all_n - own as all_n,
        all_late - own * own_late as all_late,
        all_days - own * own_days as all_days
    from as_of_t0

)

select
    order_id,
    t0,
    route_n as route_past_delivered,
    -- Shrunk toward the same figure over every route, so thin routes are not extreme.
    (route_days + {{ strength }} * all_days / nullif(all_n, 0))
    / (route_n + {{ strength }}) as route_mean_delivery_days,
    (route_late + {{ strength }} * all_late / nullif(all_n, 0))
    / (route_n + {{ strength }}) as route_late_rate_smoothed,
    -- Kept so that assert_route_history_no_future can check the as-of rule.
    route_last_delivery_used_at
from excluding_self
