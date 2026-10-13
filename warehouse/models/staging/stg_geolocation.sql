with source as (

    select * from {{ source('olist', 'geolocation') }}

)

-- Many rows per zip prefix, some of them exact duplicates and a few outside
-- Brazil. Kept as they are here; int_zip_centroids reduces them to one point.
select
    geolocation_zip_code_prefix as zip_code_prefix,
    cast(geolocation_lat as double) as latitude,
    cast(geolocation_lng as double) as longitude,
    lower(trim(geolocation_city)) as city,
    upper(trim(geolocation_state)) as state
from source
