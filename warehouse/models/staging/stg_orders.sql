with source as (

    select * from {{ source('olist', 'orders') }}

)

select
    order_id,
    customer_id,
    lower(order_status) as order_status,
    cast(order_purchase_timestamp as timestamp) as purchased_at,
    cast(order_approved_at as timestamp) as approved_at,
    -- Named "date" in the source, but both carry a time of day.
    cast(order_delivered_carrier_date as timestamp) as carrier_handover_at,
    cast(order_delivered_customer_date as timestamp) as delivered_at,
    -- Always midnight in the source: a promised day, not a promised moment.
    cast(order_estimated_delivery_date as date) as estimated_delivery_date
from source
