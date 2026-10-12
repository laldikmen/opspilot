# 0001 · Target definition

Status: accepted · 2026-10-12

## Context

The project predicts late deliveries, so "late" needs one exact definition that
the warehouse, the model and the README all share. Two details in the data make
the obvious definition wrong:

- `order_estimated_delivery_date` is always stored at midnight (0 exceptions in
  99,441 orders). It is a promised day, not a promised moment.
- Not every order has an outcome. 2,963 orders never reached `delivered`, and 8
  are marked `delivered` with no delivery date.

## Decision

`is_late = 1` when the **calendar date** of `order_delivered_customer_date` is
after the calendar date of `order_estimated_delivery_date`; otherwise 0.

Only orders with `order_status = 'delivered'` and a non-null delivery date are
labeled. Everything else is excluded from training and evaluation.

## Numbers (from `notebooks/01_eda.ipynb`)

| | Orders |
| --- | ---: |
| All orders | 99,441 |
| Labeled | 96,470 |
| Late | 6,534 (**6.77%**) |
| Excluded: shipped, not yet delivered | 1,107 |
| Excluded: canceled | 625 |
| Excluded: unavailable | 609 |
| Excluded: invoiced, processing, created, approved | 622 |
| Excluded: delivered with no delivery date | 8 |

## Consequences

- Comparing raw timestamps instead of dates would count an order delivered at
  15:00 on the promised day as late and put the late rate at 8.11%. The dbt
  model must cast both sides to `date`.
- The model says nothing about cancellations. An order canceled *because* it
  was running late is invisible to it, which belongs in the README limitations.
- The 6.77% is for all labeled orders. The rate inside the modelling window is
  different per split; see [0004](0004-chronological-split.md).
