# 0005 · Leakage rules

Status: accepted · 2026-10-12 · enforced by tests since 2026-10-13 · extended 2026-10-15

## Context

On this dataset it is easy to build a model that looks excellent because a
feature quietly contains the answer. Three rules keep every feature honest.
They follow from [0002](0002-prediction-time.md): a feature is allowed only if
it was known at t0.

## Decision

1. **No column recorded after t0.** Forbidden as features: the carrier handover
   date, the delivery date and anything derived from it (actual delivery days),
   review score and review text, and the final order status.
2. **History features use only events that had happened before t0.** A seller's
   past late rate for an order counts only that seller's orders **delivered
   before t0**, not orders merely purchased before t0. An order purchased last
   week and still in transit has no outcome yet. A seller's late rate over the
   full dataset is forbidden. The same goes for every history feature: route
   delivery times count deliveries before t0, and a seller's handover speed
   counts carrier handovers before t0.
3. **Encoders, scalers and imputers are fit on train only**, then applied
   unchanged to validation and test.

## How it is enforced

- dbt tests `assert_seller_history_no_future` and `assert_route_history_no_future`:
  each history model keeps the latest event it counted for each order
  (delivery or handover), and the tests fail if that is ever at or after t0.
- `tests/test_leakage.py` recomputes the seller history, route history and
  seller handover speed of 50 random orders in plain pandas and compares them
  with the dbt models. It also pins the exact
  column list of the model input table, so a new column cannot slip in without
  someone asking whether it was known at t0.

Both checks were tried against the mistake they exist for: with the history
filtered by purchase date instead of delivery date, the dbt test fails on 432
rows of the sample and the pandas comparison fails.

## Details fixed during implementation

- A seller's history counts every order the seller has an item in, not only
  the orders where they are the primary seller.
- An order never counts toward its own history. This matters for the 61 orders
  recorded as delivered before their own approval ([0002](0002-prediction-time.md)).
- The smoothed rate is `(late + 20 × 0.0565) / (delivered + 20)`: the prior is
  the train late rate, weighted as 20 orders. Both numbers are dbt vars, so the
  warehouse does not need to know the split.

## Changes

- 2026-10-15: four history features were added after the first 23 left XGBoost
  level with logistic regression (`reports/model_selection.md`):
  `route_mean_delivery_days`, `route_late_rate_smoothed`, `promise_slack_days`
  and `seller_mean_handover_days`. Each is shrunk toward the same quantity over
  all orders known at t0, weighted as 20 orders (`history_prior_strength`). An
  order never counts toward its own history: 1.4% of orders were handed to the
  carrier before their own approval.
- The pandas comparison caught a real bug while these were built. DuckDB's
  `greatest()` skips nulls, so orders without an approval time became zero-day
  handovers and pulled the averages down. Missing values are now checked
  explicitly. The same trap had earlier turned missing distances into 20,015 km.

## Consequences

- Filtering history by purchase date instead of delivery date is the most
  likely mistake. If the model beats the baselines by a suspiciously large
  margin, check this before anything else.
- New and small sellers have little or no history at t0. Their rate is shrunk
  toward the global train rate, which the model card has to mention.
