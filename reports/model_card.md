# Model card: late-delivery risk (OpsPilot v1)

## Intended use

- **Who:** an e-commerce operations team that can review a limited share of
  orders, for example the riskiest 10%, and act early: warn the customer, nudge
  the seller, choose another carrier.
- **What it does:** when an order is approved, it estimates the probability that
  the order arrives after its promised delivery date, with the three features
  that moved that estimate most.
- **How to use the output:** as a ranking. On new periods the probabilities can
  be badly off in level (see Calibration below), so treat them as relative risk
  unless they have been recalibrated on recent outcomes.
- **Not for:** penalising or delisting sellers, pricing for customers, promising
  delivery dates, or any marketplace other than the one it was trained on.

## Model

- XGBoost, 656 trees of depth 2 (`learning_rate` 0.0102, `min_child_weight`
  3.12, `subsample` 0.66, `colsample_bytree` 0.77, `reg_lambda` 2.56), with
  median imputation and one-hot encoding fitted on train only.
- 25 features, every one known when the order is approved: the promise, the
  basket, the route, payment, and the seller's and route's history counted only
  from outcomes known by then ([ADR 0005](../docs/decisions/0005-leakage-rules.md)).
- Calibrated with a sigmoid fitted on out-of-time predictions pooled over July
  2017 to March 2018 ([ADR 0006](../docs/decisions/0006-calibration-under-shift.md)).
- Bundle `2026.10.26+01a157f`; chosen on validation data only
  ([model selection](model_selection.md)).

## Data

- [Brazilian E-Commerce Public Dataset by Olist](../data/README.md), CC BY-NC-SA
  4.0: about 100k orders from 2016 to 2018.
- **Label:** late when delivered after the promised calendar date
  ([ADR 0001](../docs/decisions/0001-target-definition.md)). Only delivered orders
  are labeled; 2,971 cancelled or undelivered orders are excluded.
- **Window:** orders purchased after 2018-07-15 are dropped because their slow
  deliveries are missing from the data
  ([ADR 0003](../docs/decisions/0003-censoring-cutoff.md)).
- **Split by purchase date** ([ADR 0004](../docs/decisions/0004-chronological-split.md)):
  train 2017 (43,426 orders, 5.65% late), validation January to March 2018
  (20,627, 12.88%), test April 2018 to mid-July (21,949, 4.05%).

## Evaluation on the test set

Scored once, on 2026-10-26, after the model and the evaluation plan were
committed in [test_protocol.md](test_protocol.md). Full results, slices and
figures: [evaluation.md](evaluation.md).

| Model | PR-AUC (95% CI) | ROC-AUC | Late orders caught in riskiest 10% | Brier | Mean prediction |
| --- | --- | ---: | ---: | ---: | ---: |
| Prior (train late rate) | 0.041 | 0.500 | 10.0% | 0.039 | 5.6% |
| Rule: route late rate | 0.067 (0.061 to 0.075) | 0.642 | 15.2% | 0.040 | 6.9% |
| Logistic regression | 0.116 (0.105 to 0.129) | 0.767 | 35.3% | 0.039 | 7.3% |
| **XGBoost, calibrated (this model)** | **0.094 (0.086 to 0.105)** | **0.722** | **28.6%** | **0.058** | **17.2%** |

The test late rate is 4.05%, so a random ranking scores a PR-AUC of 0.041.

**Ranking.** The model ranks well above chance and above the best simple rule.
If a team checks the riskiest 10% of orders, it finds 28.6% of all late orders,
against 15.2% for ranking by the route's late rate and 10% at random.

**Logistic regression ranked better on test.** Its PR-AUC was 0.022 higher (95%
interval 0.013 to 0.030) and it caught 35.3% of late orders in the riskiest 10%.
On validation XGBoost had been ahead by a similar margin. The model was not
switched after seeing this: choosing between models on the test set would leave
no honest estimate for the one chosen. The likeliest reason is that the deeper
interactions XGBoost learned were tied to 2017 and early 2018. Retrained each
month on everything known so far, XGBoost catches 31% to 37% of late orders in
the riskiest 10% instead of 22% to 34%. Logistic regression was not retrained
in that check, so this shows retraining helps XGBoost, not that it catches up.
XGBoost stays the served model because that is what was committed to before
scoring. Validation favoured it and test did not, and no later data exists to
settle which period the future would resemble.

**Calibration.** The probabilities run about four times too high on test: a mean prediction of 17.2% against a 4.05% late rate, an ECE of 0.131,
and a worse Brier score than predicting the train late rate for everyone. This
is the prevalence shift that ADR 0006 expected: the calibrator learned from
periods running at 2.9% to 12.9% late, and test ran at 4.05%. The model still
orders risk correctly within test (see the calibration figure in
[evaluation.md](evaluation.md)), but the level drifted.

## Limitations

- **Prevalence shift.** The late rate swings from 1% to 19% between months.
  Probabilities need recalibrating on recent outcomes, and the predicted against
  observed late rate needs monitoring (planned for v2).
- **What happens after approval is invisible.** Many late orders were caused by a
  seller taking weeks to hand the order over, or by a carrier leg of a month or
  more ([error analysis](../notebooks/02_error_analysis.ipynb)). Re-scoring at
  carrier handover would catch the first kind.
- **Recovered routes look risky.** The model's most confident false alarms are
  São Paulo to Rio de Janeiro orders: the route was slow during the early-2018
  spike and recovered afterwards.
- **Old, single-marketplace data.** 2016 to 2018, one Brazilian marketplace.
- **Label choices.** Lateness is counted in calendar days; cancelled orders are
  excluded, so an order cancelled because it was running late is not counted.
  Some censoring remains near the cutoff ([ADR 0003](../docs/decisions/0003-censoring-cutoff.md)).

## Fairness

- **Sellers.** A new or small seller has little history, so its late rate is
  shrunk toward the train average: it is neither punished nor rewarded for a
  handful of orders. The seller's history still raises or lowers its orders'
  risk, so the score must not be used to penalise sellers, only to decide where
  help is needed first.
- **Regions.** Risk is driven heavily by the destination. The riskiest 10%
  contains 80% of the late orders bound for Rio de Janeiro but only 9% of those
  bound for São Paulo, and orders to the Northeast are flagged three times as
  often as average (29% of them, against 10% overall). That reflects real delivery
  performance, but a team acting on the flags would give those customers more
  attention, and the model would keep reinforcing where attention goes. Slices
  under 200 orders are flagged as unreliable in [evaluation.md](evaluation.md).

## Reproduce

`make all` rebuilds the data, warehouse, features, model and this evaluation
from the raw CSVs; `make test` runs the checks on the committed sample.
