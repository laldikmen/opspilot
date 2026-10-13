with source as (

    select * from {{ source('olist', 'sellers') }}

)

select
    seller_id,
    seller_zip_code_prefix,
    lower(trim(seller_city)) as seller_city,
    upper(trim(seller_state)) as seller_state
from source
