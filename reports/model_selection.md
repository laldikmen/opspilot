# Model selection

How the final candidate was chosen, using only the three expanding-window folds
and validation ([ADR 0004](../docs/decisions/0004-chronological-split.md)). The
test set had not been scored when this was written. Every number here is in the
`late-delivery` MLflow experiment (`make mlflow`).

## Results

Lift is PR-AUC divided by the period's late rate; a random model scores 1. The
selection number is the mean lift across the four periods.

| Model | Fold 1 | Fold 2 | Fold 3 | Validation | Mean lift | Mean ROC-AUC | Mean top-10% recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Late rate of the period | 2.9% | 4.3% | 10.3% | 12.9% | | | |
| Prior (train late rate) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 | 10.0% |
| Rule: seller smoothed late rate | 1.52 | 1.22 | 1.11 | 1.16 | 1.25 | 0.544 | 12.8% |
| Logistic regression | 1.71 | 2.24 | 1.76 | 2.19 | 1.97 | 0.680 | 24.6% |
| XGBoost, default parameters | | | | | 2.10 | | |
| **XGBoost, tuned (final)** | **2.08** | **2.61** | **1.86** | **2.12** | **2.17** | **0.687** | **25.4%** |

## Why XGBoost is the final candidate

**With the first 23 features it was not.** Tuned XGBoost and logistic regression
tied on mean lift (1.907 against 1.901), and logistic regression was clearly
better on validation. On that evidence the honest choice was the simpler model.

**What changed it was four new features, not more tuning.** They describe the
promise against what normally happens, using only what was known at t0
([ADR 0005](../docs/decisions/0005-leakage-rules.md)):

- `route_mean_delivery_days` and `route_late_rate_smoothed`: how long deliveries
  on the same seller-state to customer-state route took, and how often they were
  late, over orders delivered before t0.
- `promise_slack_days`: the promised days minus that route's usual delivery time.
- `seller_mean_handover_days`: how long the seller usually took to hand an order
  to the carrier, over handovers before t0.

They help both models. Logistic regression goes from 1.90 to 1.97, and XGBoost
from 1.91 to 2.17, because it can combine them with the state and basket
features.

**The lead is real but modest.** In a paired bootstrap of the lift difference
(XGBoost minus logistic regression, 1,000 resamples per period):

| Period | Difference | 95% interval |
| --- | ---: | --- |
| Fold 1 | +0.37 | −0.24 to +1.04 |
| Fold 2 | +0.38 | −0.01 to +0.83 |
| Fold 3 | +0.11 | +0.02 to +0.20 |
| Validation | −0.07 | −0.15 to +0.01 |

XGBoost is ahead on the folds and level on validation, and its ROC-AUC is equal
or higher in every period. The win is not search luck: XGBoost with default
parameters already reaches 2.10, and the five best of 40 trials all fall between
2.14 and 2.17.

**Both models beat the seller-history rule by a wide margin**: 2.17 against
1.25 mean lift, and a quarter of late orders caught in the riskiest 10% against
one in eight.

## Tried and rejected: recent late rate

`recent_late_rate_14d` (late rate among all orders delivered in the 14 days
before t0) was the optional feature in the plan. Trained with and without it,
on the first 23 features:

| Model | Without | With |
| --- | ---: | ---: |
| Logistic regression, mean lift | 1.90 | 1.90 |
| Logistic regression, validation lift | 2.14 | 2.00 |
| XGBoost, mean lift | 1.91 | 1.82 |
| XGBoost, validation lift | 1.87 | 1.62 |

It hurts, most of all on validation. The reason is lag: an order's outcome is
only known when it is delivered, and late orders are the slow ones, so a window
of recent deliveries trails a spike by weeks. In February 2018 the recent rate
read 4.6% while that month's orders ran 14.1% late. Inside validation the
feature is anti-correlated with the label (ROC-AUC 0.44 on its own). It is not
in the feature table.

## The final candidate

XGBoost with `max_depth` 5, `learning_rate` 0.0214, `min_child_weight` 2.68,
`subsample` 0.72, `colsample_bytree` 0.44 and `reg_lambda` 1.39, with the number
of trees set by early stopping. It is fitted on train (2017) and calibrated with
a sigmoid fitted on the pooled folds and validation
([ADR 0006](../docs/decisions/0006-calibration-under-shift.md)). `make train`
saves it, with its preprocessing, calibrator, feature list and version, to
`artifacts/model_bundle.joblib`, and logs it as the `xgboost_final` MLflow run.
