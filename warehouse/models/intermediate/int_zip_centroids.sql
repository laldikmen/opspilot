-- One point per zip prefix: the mean of its geolocation points.

select
    zip_code_prefix,
    avg(latitude) as latitude,
    avg(longitude) as longitude,
    count(*) as n_points
from {{ ref('stg_geolocation') }}
-- A few dozen points are bad geocodes far outside Brazil; drop them before averaging.
where latitude between -34.0 and 5.3
    and longitude between -74.0 and -34.7
group by zip_code_prefix
