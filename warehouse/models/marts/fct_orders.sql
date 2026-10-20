-- One row per order: label, t0, purchase date, censoring flag.
-- No split column: the split is assigned in split.py from the dates in config.py.

with orders as (

    select * from {{ ref('stg_orders') }}

),

labeled as (

    select
        *,
        -- Only delivered orders with a delivery date have an outcome (ADR 0001).
        coalesce(order_status = 'delivered' and delivered_at is not null, false) as is_labeled
    from orders

)

select
    order_id,
    customer_id,
    order_status,
    purchased_at,
    cast(purchased_at as date) as purchase_date,
    approved_at,
    -- Prediction time (ADR 0002).
    coalesce(approved_at, purchased_at) as t0,
    estimated_delivery_date,
    delivered_at,
    is_labeled,
    -- Late means delivered after the promised calendar date, so compare dates.
    case
        when is_labeled then cast(cast(delivered_at as date) > estimated_delivery_date as integer)
    end as is_late,
    -- When the outcome became known: an on-time order once it was delivered, a
    -- late one as soon as its promised day had passed without a delivery.
    -- Models only learn from outcomes known when they would be used (ADR 0004).
    case
        when is_labeled and cast(delivered_at as date) > estimated_delivery_date
            then cast(estimated_delivery_date + interval 1 day as timestamp)
        when is_labeled then delivered_at
    end as outcome_known_at,
    -- Purchased too close to the end of the data for a slow delivery to be seen (ADR 0003).
    cast(purchased_at as date) > date '{{ var("censoring_cutoff") }}' as is_censored
from labeled
