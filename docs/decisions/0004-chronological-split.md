# 0004 · Chronological split

Status: accepted · 2026-10-12 · revised 2026-10-13 (see Changes)

## Context

The model will be used on orders placed after the ones it was trained on. A
random split would let it train on orders from the same weeks it is tested on,
and would let history features see the future. Results from a random split are
flattering and wrong.

## Decision

Split by purchase date, both ends inclusive:

| Split | Purchase dates | Labeled orders | Share | Late orders | Late rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | 2017-01-01 to 2017-12-31 | 43,426 | 50.5% | 2,453 | 5.65% |
| Validation | 2018-01-01 to 2018-03-31 | 20,627 | 24.0% | 2,657 | 12.88% |
| Test | 2018-04-01 to 2018-07-15 | 21,949 | 25.5% | 889 | 4.05% |

- The split is assigned in `src/opspilot/models/split.py` from the dates in
  `config.py`. It is not stored in the warehouse, so changing it later is a
  config edit, not a rebuild.
- Hyperparameters are tuned on three expanding-window folds **inside train**.
- Models are selected on the average across those folds plus validation, never
  on validation alone, because validation is a high-late-rate period.
- The test set is scored once, in Week 7, and the model is not changed after.

The start and end of the window come from [0003](0003-censoring-cutoff.md).

## Consequences

**The three splits do not share a late rate.** Late deliveries spike in
November 2017 (12.4%, Black Friday) and again in February and March 2018
(14.1% and 19.0%). The second spike makes up most of validation, so validation
runs at three times the test rate. This is how the data behaves over time, not
a leak, but it has to be handled:

- A calibrator fitted on validation learns a base rate of 12.9% and will be
  applied to a period at 4.1%. Calibrated probabilities are likely to run high
  on test. How to calibrate under this shift is its own decision:
  [0006](0006-calibration-under-shift.md).
- PR-AUC depends on the base rate, so validation and test PR-AUC are not
  comparable with each other. Models are compared within one period only, and
  every PR-AUC is reported with its late rate and its lift (PR-AUC divided by
  the late rate). Across periods, ROC-AUC and top-10% recall are the numbers
  to lean on.
- `purchase_month` is dropped as a feature. One year of training data cannot
  teach seasonality; the model would only memorize the months of 2017.
- Train contains one spike (November 2017), so the model has seen a bad month,
  but only one.

## Alternative considered

Train through February 2018, validate on March to May, test on June to the
cutoff. Rejected: validation would still contain the March spike (10.1% late),
and the test set would shrink to 8,402 orders with only 140 late ones, too few
for a stable PR-AUC interval.

## Changes

- 2026-10-13: the split dates were confirmed unchanged after the EDA. Because
  of the validation spike, model selection moved from "validation" to "the
  average across the folds plus validation", the split moved out of dbt into
  `split.py`, `purchase_month` was dropped, and calibration became ADR 0006.
