-- What is newly known when the seller hands an order to the carrier: the second
-- prediction moment in v1.1. One row per order that can still be scored then.
-- The history features in ml_late_delivery_features are as of t0, so also known here.

with orders as (

    select
        features.order_id,
        features.seller_mean_handover_days,
        features.route_mean_delivery_days,
        cast(stg.estimated_delivery_date as timestamp) as estimated_at,
        stg.delivered_at,
        -- A few handovers are recorded before approval; nothing is scored before t0.
        greatest(stg.carrier_handover_at, features.t0) as scored_at,
        features.t0
    from {{ ref('ml_late_delivery_features') }} as features
    inner join {{ ref('stg_orders') }} as stg on features.order_id = stg.order_id
    where stg.carrier_handover_at is not null

),

timed as (

    select
        *,
        date_diff('second', t0, scored_at) / 86400.0 as handover_delay_days,
        date_diff('second', scored_at, estimated_at) / 86400.0 as days_left
    from orders
    -- Handed over after the promised day: already known to be late, nothing to predict.
    -- Delivered by then: already over.
    where cast(scored_at as date) <= cast(estimated_at as date)
        and delivered_at > scored_at

)

select
    order_id,
    scored_at,
    handover_delay_days,
    handover_delay_days - seller_mean_handover_days as handover_vs_usual_days,
    days_left,
    -- Days left against what this route's carrier leg usually takes.
    days_left - (route_mean_delivery_days - seller_mean_handover_days) as carrier_slack_days
from timed
