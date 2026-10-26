# Test protocol

Written and committed **before** the test set was scored, so the git history
shows the model was fixed first. `make evaluate` recomputes the prediction
fingerprint below and states in `reports/evaluation.md` whether it matches.

## What is frozen

- Bundle `artifacts/model_bundle.joblib`, version `2026.10.26+01a157f`, built by
  `make train` at commit `01a157f`.
- Bundle file SHA-256: `cd491aa873591dc601a280726bb2891395eed2ed064dfbb605286d950cd94a4f`
- Calibrators file SHA-256: `e74434598fb85f277cf2ddee52815ad9bd33899672335844a7aec8c2eae9c9e5`
- Prediction fingerprint: `6da76c34b8f3ba6dbee598aa81f78b72bcfd4d00f9366a87cb4035c851f911b8`
  (SHA-256 of the bundle's raw predictions on validation, rounded to 8
  decimals; it survives a deterministic rebuild on another day.)
- Final model: XGBoost with `colsample_bytree` 0.767, `learning_rate` 0.0102, `max_depth` 2, `min_child_weight` 3.1246, `reg_lambda` 2.5579, `subsample` 0.6558; 25 features; fitted on
  train orders whose outcome was known by 2018-04-01; calibrated with a
  sigmoid fitted on the pooled folds and validation (ADR 0006).

## What is scored, once

The test set: purchases from 2018-04-01 to 2018-07-15, 21,949 labeled orders.

| # | Model | Notes |
| --- | --- | --- |
| 1 | Prior (train late rate) | |
| 2 | Rule: seller smoothed late rate | nothing fitted |
| 3 | Rule: route late rate | nothing fitted |
| 4 | Logistic regression | refitted on the same train rows; deterministic |
| 5 | XGBoost, uncalibrated | the frozen bundle's raw output |
| 6 | XGBoost, calibrated on validation | `last_sigmoid`: lower backtest Brier than the other method for this source (0.0800) |
| 7 | XGBoost, calibrated on pooled folds | `pooled_sigmoid` (0.0794); **the final model** |

- Metrics for every model: PR-AUC with a 95% bootstrap interval (1,000
  resamples), lift, ROC-AUC, precision and recall in the riskiest 10% (recall
  with an interval), Brier score, ECE (10 bins), mean prediction.
- The final model against logistic regression and the route rule: paired
  bootstrap of the PR-AUC and top-10% recall differences.
- Capacity: share of late orders caught when 5%, 10%, 20% or 30% are checked.
- Slices of the final model: customer state, the 10 largest product
  categories, purchase month. Slices under 200 orders are flagged.
- Figures: precision-recall curves, capacity curve, calibration on validation
  and test, late rate by risk decile, SHAP importance.
- Explanations for three orders picked by rule: the riskiest late order, the
  riskiest on-time order, and the late order rated safest.
- Supplementary: each test month scored by the frozen model and by the same
  parameters retrained on everything known on that month's first day.

## Rules

- Nothing about the model, features or calibration changes after the test
  results are seen.
- If a bug in the evaluation code turns up afterwards, it is fixed and the
  report regenerated, the fix is recorded here, and the model is not touched.
- Results are reported as they come out, including any baseline the final
  model fails to beat.
