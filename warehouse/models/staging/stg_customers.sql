with source as (

    select * from {{ source('olist', 'customers') }}

)

select
    customer_id,
    customer_unique_id,
    -- Text, not a number: prefixes such as 01037 keep their leading zero.
    customer_zip_code_prefix,
    lower(trim(customer_city)) as customer_city,
    upper(trim(customer_state)) as customer_state
from source
