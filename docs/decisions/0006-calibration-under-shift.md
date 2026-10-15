# 0006 · Calibration under shift

Status: accepted · 2026-10-15

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
   validation (July 2017 to March 2018, 9.1% late).
3. **None.**

Isotonic and sigmoid were both tried for options 1 and 2.

## How they were compared

The choice cannot be scored on the period a calibrator was fitted on, and the
test set is off limits. So each strategy was replayed one step back in time.
For fold 2, fold 3 and validation in turn, it was fitted only on the periods
before, applied to that period, and scored with the Brier score and ECE
(`calibrate.backtest`). Final XGBoost, Brier score per period:

| Strategy | Fold 2 | Fold 3 | Validation | Mean Brier | Mean ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Pooled, sigmoid** | 0.0403 | 0.0945 | 0.1038 | **0.0795** | **0.031** |
| Last, sigmoid | 0.0403 | 0.0938 | 0.1076 | 0.0806 | 0.048 |
| Pooled, isotonic | 0.0422 | 0.0960 | 0.1044 | 0.0809 | 0.035 |
| None | 0.0402 | 0.0949 | 0.1088 | 0.0813 | 0.046 |
| Last, isotonic | 0.0422 | 0.0947 | 0.1085 | 0.0818 | 0.049 |

## Decision

**Sigmoid calibration fitted on the pooled folds and validation.** It has the
lowest mean Brier score and the lowest mean ECE. Pooling several periods with
late rates from 2.9% to 12.9% makes it less hostage to any one period than
calibrating on validation alone. Sigmoid beats isotonic, which fits steps to
noise.

## Consequences

- The margins are small, and every step in the backtest went from a lower late
  rate to a higher one. Test goes the other way: 4.05% after a 12.9% quarter,
  against a calibration set at 9.1%. Calibrated probabilities are therefore
  expected to run high on test. That is the prevalence shift, not a bug, and it
  is what the README section "Calibration under prevalence shift" will discuss.
- Calibration is monotone, so ranking metrics (PR-AUC, ROC-AUC, top-10% recall)
  do not depend on this choice. Only the Brier score and ECE do.
- In Week 7 all three options are still scored on test and reported side by
  side in `reports/evaluation.md`. `make train` saves all of them to
  `artifacts/calibrators.joblib` for that.
- The lasting fix is monitoring, not a better static calibrator: track the
  predicted against the observed late rate as outcomes arrive and recalibrate
  when they drift. That is in the v2 plan.
