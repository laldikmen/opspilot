# 0005 · Leakage rules

Status: accepted · 2026-10-12 · enforced by tests from Week 3

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

- dbt test `assert_seller_history_no_future`: fails if any row of the seller
  history uses an order delivered at or after t0.
- `tests/test_leakage.py`: recomputes seller history for 50 random orders by
  brute force in pandas and compares, and asserts that no forbidden column is
  in the model input table.

## Consequences

- Filtering history by purchase date instead of delivery date is the most
  likely mistake. If the model beats the baselines by a suspiciously large
  margin, check this before anything else.
- New and small sellers have little or no history at t0. Their rate is shrunk
  toward the global train rate, which the model card has to mention.
