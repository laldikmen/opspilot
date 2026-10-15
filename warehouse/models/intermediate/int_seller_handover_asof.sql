-- One row per order: how quickly its primary seller had handed earlier orders to
-- the carrier, counting only handovers that happened before t0.
--
-- A past order's handover date is known once it happens; this order's own
-- handover is recorded after t0 and is never used.

{% set strength = var('history_prior_strength') %}

with orders as (

    select
        fct.order_id,
        fct.t0,
        items.primary_seller_id as seller_id,
        stg.carrier_handover_at,
        -- Days from approval to handover. A few handovers are recorded before
        -- approval; they count as zero rather than negative. The explicit null
        -- check matters: DuckDB's greatest() skips nulls, so a missing approval
        -- would otherwise become a zero-day handover.
        case
            when stg.approved_at is not null and stg.carrier_handover_at is not null
                then greatest(date_diff('second', stg.approved_at, stg.carrier_handover_at) / 86400.0, 0)
        end as handover_days
    from {{ ref('fct_orders') }} as fct
    inner join {{ ref('int_order_items_agg') }} as items on fct.order_id = items.order_id
    inner join {{ ref('stg_orders') }} as stg on fct.order_id = stg.order_id

),

handovers as (

    select * from orders where handover_days is not null

),

seller_totals as (

    select
        seller_id,
        carrier_handover_at,
        sum(count(*)) over running as n,
        sum(sum(handover_days)) over running as days
    from handovers
    group by seller_id, carrier_handover_at
    window running as (partition by seller_id order by carrier_handover_at)

),

all_totals as (

    select
        carrier_handover_at,
        sum(count(*)) over running as n,
        sum(sum(handover_days)) over running as days
    from handovers
    group by carrier_handover_at
    window running as (order by carrier_handover_at)

),

as_of_t0 as (

    select
        orders.order_id,
        orders.t0,
        coalesce(seller_totals.n, 0) as seller_n,
        coalesce(seller_totals.days, 0) as seller_days,
        seller_totals.carrier_handover_at as seller_last_handover_used_at,
        coalesce(all_totals.n, 0) as all_n,
        coalesce(all_totals.days, 0) as all_days,
        -- 1.4% of orders were handed over before their own approval; never let
        -- an order's own handover into its history.
        cast(coalesce(orders.carrier_handover_at < orders.t0 and orders.handover_days is not null, false) as integer)
            as own,
        coalesce(orders.handover_days, 0) as own_days
    from orders
    asof left join seller_totals
        on orders.seller_id = seller_totals.seller_id
        and orders.t0 > seller_totals.carrier_handover_at
    asof left join all_totals
        on orders.t0 > all_totals.carrier_handover_at

)

select
    order_id,
    t0,
    seller_n - own as seller_past_handovers,
    (seller_days - own * own_days + {{ strength }} * (all_days - own * own_days) / nullif(all_n - own, 0))
    / (seller_n - own + {{ strength }}) as seller_mean_handover_days,
    -- Kept so that assert_seller_history_no_future can check the as-of rule.
    seller_last_handover_used_at
from as_of_t0
