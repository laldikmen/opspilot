# 0002 · Prediction time (t0)

Status: accepted · 2026-10-12

## Context

A feature is only honest if it was known at the moment the prediction would be
made. That moment has to be one timestamp per order, fixed before any feature
is written.

## Decision

`t0 = order_approved_at`, falling back to `order_purchase_timestamp` when the
approval timestamp is null.

A feature is allowed only if its value was known at t0. Approval is the first
point at which an order is real (payment cleared) and an operations team could
act on a risk score. It also makes the wait for approval a legitimate feature
(`approval_delay_hours`).

## Numbers (from `notebooks/01_eda.ipynb`, labeled orders)

| | |
| --- | --- |
| Approval timestamp missing | 14 of 96,470 (160 of all 99,441 orders) |
| Approval delay, median | 0.34 hours (20 minutes) |
| Approved within 1 hour | 64.0% |
| Approved after more than 24 hours | 17.5% |
| Approval delay, 99th percentile / max | 90 hours / 741 hours (31 days) |
| Negative delays | 0 |

## Consequences

- The fallback is used for 14 labeled orders, so it has no practical effect on
  results. For those orders `approval_delay_hours` is null.
- The approval timestamp is not perfectly reliable. 1,350 labeled orders (1.4%)
  were handed to the carrier before t0, 61 were delivered before t0, and 6 have
  an estimated date before t0. They are kept: they are few, and removing them
  would mean filtering on columns recorded after t0. Week 7 error analysis
  should check whether the model does unusually well on them.
- Because t0 can be days after purchase, `promised_days` (estimated date minus
  t0) can be short or even negative. That is real information at t0, not a leak.
