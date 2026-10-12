# Evaluation

Filled in once, in Week 7, when the test set is scored. Until then every cell
stays empty: model selection uses validation data and the expanding-window
folds only.

## Test-set results

Test late rate (the PR-AUC of a random model): _TBD_

| Model | PR-AUC (95% CI) | ROC-AUC | Precision @ top 10% | Recall @ top 10% | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Prior (train late rate) | | | | | | |
| Rule: seller smoothed late rate | | | | | | |
| Logistic regression | | | | | | |
| XGBoost, uncalibrated | | | | | | |
| XGBoost, calibrated | | | | | | |

Intervals: 1,000 bootstrap resamples of the test set.

## Slices

By customer state, top 10 product categories and purchase month. Slices with
fewer than 200 orders are flagged as unreliable.

## Calibration

Brier score, expected calibration error (10 bins) and the calibration plot.
