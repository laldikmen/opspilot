# 0005 · Leakage rules

Status: accepted · 2026-10-12 · enforced by tests since 2026-10-13

## Context

On this dataset it is easy to build a model that looks excellent because a
feature quietly contains the answer. Three rules keep every feature honest.
They follow from [0002](0002-prediction-time.md): a feature is allowed only if
it was known at t0.

## Decision

1. **No column recorded after t0.** Forbidden as features: the carrier handover
   date, the delivery date and anything derived from it (actual delivery days),
   review score and review text, and the final order status.
2. **History features use only outcomes known before t0.** A seller's past late
   rate for an order counts only that seller's orders **delivered before t0**,
   not orders merely purchased before t0. An order purchased last week and
   still in transit has no outcome yet. A seller's late rate over the full
   dataset is forbidden.
3. **Encoders, scalers and imputers are fit on train only**, then applied
   unchanged to validation and test.

## How it is enforced

- dbt test `assert_seller_history_no_future`: `int_seller_history_asof` keeps
  the latest delivery it counted for each order, and the test fails if that is
  ever at or after t0.
- `tests/test_leakage.py` recomputes the seller history of 50 random orders in
  plain pandas and compares it with the dbt model. It also pins the exact
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

## Consequences

- Filtering history by purchase date instead of delivery date is the most
  likely mistake. If the model beats the baselines by a suspiciously large
  margin, check this before anything else.
- New and small sellers have little or no history at t0. Their rate is shrunk
  toward the global train rate, which the model card has to mention.
