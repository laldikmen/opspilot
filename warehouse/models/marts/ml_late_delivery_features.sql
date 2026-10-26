-- The model input table: one row per labeled, uncensored order.
-- Every feature must have been known at t0 (docs/decisions/0005-leakage-rules.md).
-- tests/test_leakage.py pins the exact column list.

with orders as (

    select * from {{ ref('fct_orders') }}
    where is_labeled and not is_censored

),

order_items as (

    select * from {{ ref('int_order_items_agg') }}

),

seller_history as (

    select * from {{ ref('int_seller_history_asof') }}

),

seller_handover as (

    select * from {{ ref('int_seller_handover_asof') }}

),

route_history as (

    select * from {{ ref('int_route_history_asof') }}

),

customers as (

    select
        customers.customer_id,
        customers.customer_state,
        centroids.latitude,
        centroids.longitude
    from {{ ref('stg_customers') }} as customers
    left join {{ ref('int_zip_centroids') }} as centroids
        on customers.customer_zip_code_prefix = centroids.zip_code_prefix

),

sellers as (

    select
        sellers.seller_id,
        sellers.seller_state,
        centroids.latitude,
        centroids.longitude
    from {{ ref('stg_sellers') }} as sellers
    left join {{ ref('int_zip_centroids') }} as centroids
        on sellers.seller_zip_code_prefix = centroids.zip_code_prefix

),

payment_totals as (

    select
        order_id,
        count(distinct payment_type) as n_payment_methods,
        max(installments) as installments
    from {{ ref('stg_payments') }}
    group by order_id

),

-- The payment type that carries the most money; ties go alphabetically.
main_payment as (

    select
        order_id,
        payment_type
    from {{ ref('stg_payments') }}
    group by order_id, payment_type
    qualify row_number() over (
        partition by order_id order by sum(payment_value) desc, payment_type
    ) = 1

)

select
    orders.order_id,
    orders.purchase_date,
    orders.t0,
    orders.is_late,

    -- The promise and the approval
    date_diff('second', orders.t0, cast(orders.estimated_delivery_date as timestamp))
    / 86400.0 as promised_days,
    -- Positive when the promise is more generous than this route usually needs.
    date_diff('second', orders.t0, cast(orders.estimated_delivery_date as timestamp)) / 86400.0
    - route_history.route_mean_delivery_days as promise_slack_days,
    date_diff('second', orders.purchased_at, orders.approved_at) / 3600.0 as approval_delay_hours,
    isodow(orders.purchased_at) as purchase_dow,
    hour(orders.purchased_at) as purchase_hour,

    -- The basket
    cast(order_items.total_price as double) as total_price,
    cast(order_items.total_freight as double) as total_freight,
    cast(order_items.total_freight / nullif(order_items.total_price, 0) as double)
        as freight_ratio,
    order_items.n_items,
    order_items.n_sellers,
    order_items.n_products,
    order_items.total_weight_g,
    order_items.total_volume_cm3,
    order_items.main_category,

    -- The route
    route_history.route_mean_delivery_days,
    route_history.route_late_rate_smoothed,
    customers.customer_state,
    sellers.seller_state,
    cast(customers.customer_state = sellers.seller_state as integer) as same_state,
    {{ haversine_km(
        'customers.latitude', 'customers.longitude', 'sellers.latitude', 'sellers.longitude'
    ) }} as distance_km,

    -- The payment
    main_payment.payment_type,
    payment_totals.installments,
    payment_totals.n_payment_methods,

    -- The seller, as of t0
    seller_history.seller_late_rate_smoothed,
    seller_handover.seller_mean_handover_days

from orders
left join order_items on orders.order_id = order_items.order_id
left join seller_history on orders.order_id = seller_history.order_id
left join seller_handover on orders.order_id = seller_handover.order_id
left join route_history on orders.order_id = route_history.order_id
left join customers on orders.customer_id = customers.customer_id
left join sellers on order_items.primary_seller_id = sellers.seller_id
left join payment_totals on orders.order_id = payment_totals.order_id
left join main_payment on orders.order_id = main_payment.order_id
