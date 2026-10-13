with source as (

    select * from {{ source('olist', 'payments') }}

)

select
    order_id || '-' || payment_sequential as payment_key,
    order_id,
    cast(payment_sequential as integer) as payment_sequence,
    lower(payment_type) as payment_type,
    cast(payment_installments as integer) as installments,
    cast(payment_value as decimal(10, 2)) as payment_value
from source
