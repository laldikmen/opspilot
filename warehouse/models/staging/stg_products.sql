with source as (

    select * from {{ source('olist', 'products') }}

),

translation as (

    select * from {{ source('olist', 'category_translation') }}

)

select
    source.product_id,
    -- English name where one exists; two categories have no translation and
    -- keep their Portuguese name. Null when the product has no category.
    lower(
        coalesce(translation.product_category_name_english, source.product_category_name)
    ) as category_name,
    lower(source.product_category_name) as category_name_portuguese,
    -- "lenght" is misspelled in the source.
    cast(source.product_name_lenght as integer) as name_length,
    cast(source.product_description_lenght as integer) as description_length,
    cast(source.product_photos_qty as integer) as photos_qty,
    cast(source.product_weight_g as integer) as weight_g,
    cast(source.product_length_cm as integer) as length_cm,
    cast(source.product_height_cm as integer) as height_cm,
    cast(source.product_width_cm as integer) as width_cm
from source
left join translation
    on source.product_category_name = translation.product_category_name
