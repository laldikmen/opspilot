# 0004 · Chronological split

Status: accepted · 2026-10-12

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

- Hyperparameters are tuned on three expanding-window folds **inside train**.
- Validation is used for model selection and calibration.
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
  on test. Week 5 must check this before the calibration method is fixed (for
  example by also checking calibration on the last expanding-window fold).
- PR-AUC depends on the base rate, so validation and test PR-AUC are not
  comparable with each other. Every PR-AUC is reported next to its late rate.
- Train contains one spike (November 2017), so the model has seen a bad month,
  but only one.

## Alternative considered

Train through February 2018, validate on March to May, test on June to the
cutoff. Rejected: validation would still contain the March spike (10.1% late),
and the test set would shrink to 8,402 orders with only 140 late ones, too few
for a stable PR-AUC interval.
