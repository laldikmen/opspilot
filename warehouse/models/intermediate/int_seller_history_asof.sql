-- One row per order: what was known about its primary seller at the order's t0.
--
-- Reads t0 and the label from fct_orders on purpose, so they are defined once.

{% set prior = var('seller_prior_late_rate') %}
{% set strength = var('seller_prior_strength') %}

with orders as (

    select * from {{ ref('fct_orders') }}

),

-- An order counts toward the history of every seller with an item in it.
seller_orders as (

    select distinct
        items.seller_id,
        orders.order_id,
        orders.purchased_at,
        orders.delivered_at,
        orders.is_late
    from {{ ref('stg_order_items') }} as items
    inner join orders on items.order_id = orders.order_id

),

targets as (

    select
        orders.order_id,
        orders.t0,
        order_items.primary_seller_id as seller_id
    from orders
    inner join {{ ref('int_order_items_agg') }} as order_items
        on orders.order_id = order_items.order_id

),

history as (

    select
        targets.order_id,
        count(past.order_id) as seller_past_delivered,
        coalesce(sum(past.is_late), 0) as seller_past_late,
        max(past.delivered_at) as seller_last_delivery_used_at
    from targets
    left join seller_orders as past
        on targets.seller_id = past.seller_id
        -- The leakage rule (ADR 0005): an outcome counts only once it was known,
        -- which means delivered before t0, not merely purchased before t0.
        and past.delivered_at < targets.t0
        and past.is_late is not null
        -- A few orders are recorded as delivered before their own approval.
        and past.order_id != targets.order_id
    group by targets.order_id

)

select
    targets.order_id,
    targets.seller_id,
    targets.t0,
    history.seller_past_delivered,
    history.seller_past_late,
    -- Shrunk toward the prior, so a seller with two orders is not rated 0% or 100%.
    (history.seller_past_late + {{ strength }} * {{ prior }})
    / (history.seller_past_delivered + {{ strength }}) as seller_late_rate_smoothed,
    -- Kept so that assert_seller_history_no_future can check the rule above.
    history.seller_last_delivery_used_at
from targets
inner join history on targets.order_id = history.order_id
