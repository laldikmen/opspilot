-- One row per order that has items: totals, counts, primary seller, main category.

with items as (

    select
        items.order_id,
        items.seller_id,
        items.product_id,
        items.price,
        items.freight_value,
        products.category_name,
        products.weight_g,
        products.length_cm * products.height_cm * products.width_cm as volume_cm3
    from {{ ref('stg_order_items') }} as items
    left join {{ ref('stg_products') }} as products
        on items.product_id = products.product_id

),

totals as (

    select
        order_id,
        sum(price) as total_price,
        sum(freight_value) as total_freight,
        count(*) as n_items,
        count(distinct seller_id) as n_sellers,
        count(distinct product_id) as n_products,
        -- Null when any item has no weight or size, rather than a total that is
        -- silently too small. Imputed later with train medians.
        case when count(weight_g) = count(*) then sum(weight_g) end as total_weight_g,
        case when count(volume_cm3) = count(*) then sum(volume_cm3) end as total_volume_cm3
    from items
    group by order_id

),

-- The seller with the highest item value in the order; ties go to the lower id.
primary_seller as (

    select
        order_id,
        seller_id as primary_seller_id
    from items
    group by order_id, seller_id
    qualify row_number() over (partition by order_id order by sum(price) desc, seller_id) = 1

),

-- The category with the highest item value in the order. Null when that
-- product has no category.
main_category as (

    select
        order_id,
        category_name as main_category
    from items
    group by order_id, category_name
    qualify row_number() over (partition by order_id order by sum(price) desc, category_name) = 1

)

select
    totals.order_id,
    totals.total_price,
    totals.total_freight,
    totals.n_items,
    totals.n_sellers,
    totals.n_products,
    totals.total_weight_g,
    totals.total_volume_cm3,
    primary_seller.primary_seller_id,
    main_category.main_category
from totals
inner join primary_seller on totals.order_id = primary_seller.order_id
inner join main_category on totals.order_id = main_category.order_id
