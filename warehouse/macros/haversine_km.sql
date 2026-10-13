{# Great-circle distance in kilometres between two latitude/longitude points.
   Null when any coordinate is null. The explicit check is needed because
   DuckDB's least() skips nulls, which would turn a missing point into 20,015 km. #}
{% macro haversine_km(lat1, lng1, lat2, lng2) %}
    case
        when {{ lat1 }} is not null and {{ lng1 }} is not null
            and {{ lat2 }} is not null and {{ lng2 }} is not null
            then 2 * 6371.0 * asin(sqrt(least(1.0,
                pow(sin(radians({{ lat2 }} - {{ lat1 }}) / 2), 2)
                + cos(radians({{ lat1 }})) * cos(radians({{ lat2 }}))
                * pow(sin(radians({{ lng2 }} - {{ lng1 }}) / 2), 2)
            )))
    end
{% endmacro %}
