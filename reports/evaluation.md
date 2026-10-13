# Evaluation

Filled in once, in Week 7, when the test set is scored. Until then every cell
stays empty: models are selected on the expanding-window folds plus validation.

## Test-set results

Test late rate (the PR-AUC of a random model): _TBD_

| Model | PR-AUC (95% CI) | Lift | ROC-AUC | Precision @ top 10% | Recall @ top 10% | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Prior (train late rate) | | | | | | | |
| Rule: seller smoothed late rate | | | | | | | |
| Logistic regression | | | | | | | |
| XGBoost, uncalibrated | | | | | | | |
| XGBoost, calibrated on validation | | | | | | | |
| XGBoost, calibrated on pooled folds | | | | | | | |

Lift is PR-AUC divided by the late rate. PR-AUC is not comparable across
periods with different late rates; lift, ROC-AUC and top-10% recall are.
Intervals: 1,000 bootstrap resamples of the test set.

## Slices

By customer state, top 10 product categories and purchase month. Slices with
fewer than 200 orders are flagged as unreliable.

## Calibration

Brier score, expected calibration error (10 bins) and a calibration plot with
validation and test on the same axes. Validation's late rate is about three
times test's, so all three calibration options are reported
([ADR 0006](../docs/decisions/0006-calibration-under-shift.md)).
