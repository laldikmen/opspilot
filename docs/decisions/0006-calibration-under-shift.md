# 0006 · Calibration under shift

Status: accepted · 2026-10-15 · numbers updated 2026-10-21 and 2026-10-26 · outcome added (see What happened on test)

## Context

The API returns probabilities, so they have to mean what they say. The usual
recipe is to fit a calibrator on validation, but here validation is the
anomalous quarter: its late rate is 12.88%, against 5.65% in train and 4.05%
in test ([0004](0004-chronological-split.md)). A calibrator learns the base rate
of whatever it was fitted on.

## Options compared

1. **Last period**: a calibrator fitted on the most recent scored period. For
   the final model that is validation.
2. **Pooled**: a calibrator fitted on the out-of-sample predictions of every
   earlier period together. For the final model that is the three folds plus
   validation, July 2017 to March 2018.
3. **None.**

Isotonic and sigmoid were both tried for options 1 and 2. A calibrator only
learns from orders whose outcome was known on the day it starts being used. On
the first test day that leaves 17,427 of the 20,627 validation orders (10.3%
late), and 46,921 orders for the pooled set (7.9% late).

## How they were compared

The choice cannot be scored on the period a calibrator was fitted on, and the
test set is off limits. So each strategy was replayed one step back in time.
For fold 2, fold 3 and validation in turn, it was fitted only on the earlier
periods (and only on outcomes known when that period starts), applied to that
period, and scored with the Brier score and ECE (`calibrate.backtest`). Final
XGBoost, Brier score per period:

| Strategy | Fold 2 | Fold 3 | Validation | Mean Brier | Mean ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Pooled, sigmoid** | 0.0402 | 0.0959 | 0.1021 | **0.0794** | **0.037** |
| Last, sigmoid | 0.0402 | 0.0960 | 0.1037 | 0.0800 | 0.043 |
| Last, isotonic | 0.0403 | 0.0961 | 0.1044 | 0.0803 | 0.038 |
| Pooled, isotonic | 0.0403 | 0.0960 | 0.1048 | 0.0804 | 0.041 |
| None | 0.0400 | 0.0963 | 0.1086 | 0.0816 | 0.053 |

`make train` regenerates this table in `reports/selection_tables.md`.

## Decision

**Sigmoid calibration fitted on the pooled folds and validation.** It has the
lowest mean Brier score and the lowest mean ECE. Pooling several periods with
late rates from 2.9% to 12.9% makes it less hostage to any one period than
calibrating on validation alone. On the last-period option, isotonic and sigmoid
swap places between runs, which is another sign that one period is too little
to calibrate on.

## What happened on test

Calibrated probabilities averaged 17.2% against a 4.05% late rate. The shift
explains part of that, but the decision above also had a design flaw that the
backtest could not show: **the calibrator was fitted on one kind of model and
applied to another.**

| Step | Mean prediction | Actual late rate |
| --- | ---: | ---: |
| Earlier-period models, on the calibrator's rows | 3.8% | 7.9% |
| The fitted sigmoid | about doubles a raw score (5% becomes 10.7%, 9% becomes 17.5%) | |
| Final model, uncalibrated, on test | 9.2% | 4.05% |
| Final model, calibrated, on test | 17.2% | 4.05% |

The earlier-period models read low, so the calibrator learned to push scores up.
The final model read higher to begin with. Under the known-outcome rule
([0004](0004-chronological-split.md)) it trained on 2,165 December 2017 orders
the validation-period model never saw, because their outcomes were unknown on 1
January. Those orders ran 16.6% late, and on the same validation orders they
lift the mean prediction from 5.9% to 8.1%. Doubling that higher reading produced
the 17%.

The uncalibrated model was still about twice too high, so the shift is real, but
the calibrator made it worse. The frozen model was not changed. v1.1 tests a fix:
correcting only the overall level, from the same kind of model it will be applied
to.

## Consequences

- The margins are small, and every step in the backtest went from a lower late
  rate to a higher one. Test went the other way: 4.05% after a 12.9% quarter.
  Calibrated probabilities were expected to run high on test, though not by this
  much; see above.
- Calibration is monotone, so ranking metrics (PR-AUC, ROC-AUC, top-10% recall)
  do not depend on this choice. Only the Brier score and ECE do.
- In Week 7 all three options are still scored on test and reported side by
  side in `reports/evaluation.md`. `make train` saves all of them to
  `artifacts/calibrators.joblib` for that.
- The lasting fix is monitoring, not a better static calibrator: track the
  predicted against the observed late rate as outcomes arrive and recalibrate
  when they drift. That is in the v2 plan.

## Changes

- 2026-10-21: calibrators now skip outcomes that were still unknown on the day
  they start being used ([0004](0004-chronological-split.md)), and the final
  model's parameters changed with the re-run search. The numbers above are from
  that re-run. The decision did not change.
- 2026-10-26: re-run after two calendar-proxy features were dropped
  ([0004](0004-chronological-split.md)). Pooled sigmoid is still best. For the
  "last period" variant reported in Week 7, sigmoid now edges isotonic, so that
  variant uses sigmoid.
