# 0007 · The v1.1 recipe

Status: accepted · 2026-11-01

## Context

The v1 test showed three problems: logistic regression ranked better than the
frozen XGBoost, probabilities ran about four times too high, and many late orders
go wrong after approval. v1 itself does not change. v1.1 is a separate version.

**None of the numbers here are a clean test.** Features and parameters were chosen
on data up to March 2018, and April to July 2018 is the v1 test set, already seen.
The data ends there, so no untouched period remains. v1.1 is compared month by
month over July 2017 to July 2018, each model retrained on outcomes known on the
month's first day. See [walk_forward.md](../../reports/v1.1/walk_forward.md) and
[handover.md](../../reports/v1.1/handover.md).

## Decision

1. **Average logistic regression and XGBoost**, in log-odds. Retrained monthly,
   neither single model clearly wins: logistic regression has the better lift
   (2.37 against 2.26), XGBoost the better top-10% recall (27.8% against 26.1%).
   The average beats both on lift (2.51) in 11 of 13 months, has the best ROC-AUC
   (0.704), and ties XGBoost on recall (27.6%).
2. **Retrain monthly** with `make retrain`. The months from April 2018 improve from
   22% to 34% top-10% recall for the frozen model to 31% to 37% retrained.
3. **No calibrator.** The flaw in [0006](0006-calibration-under-shift.md) suggested a
   fix: shift the level using the same model's own earlier predictions. It does not
   work. It helps while late rates rise (mean monthly gap 4.4 points against 5.5
   before April 2018) and hurts when they fall (11.0 against 5.9 after), landing on
   the same 17% average as v1 on April to July, because it always follows the recent
   past. Choosing on the months before April would have picked it. Retraining without
   a calibrator gives about 10% on those months, still about twice the true 4.05%,
   and it under-predicts spikes badly. Past outcomes alone cannot give next month's
   late rate.
4. **Risk bands by rank.** On its first day each model scores the previous 90 days of
   orders; high is their top 10%, medium the next 20%. This needs no outcomes. The
   bands stay ordered in every month, though the share of high orders still moves
   between 1.5% and 20%.
5. **Add a second prediction at carrier handover**, on orders still open by then.
   Four new features: the handover delay, the delay against the seller's usual, the
   days left, and those days against the route's usual carrier time. On the same
   orders, top-10% recall rises from 28.2% to 36.5%, in all 13 months. With these four
   features shuffled within each month the gain disappears (27.4%).

Recent weighting (a 90-day half-life) was tried and rejected: it did not help either
model.

## Consequences

- Reporting: quote the percentage as relative risk, and lean on the band.
- The handover gain is partly smaller than the lift suggests. Lift doubles mostly
  because of a few near-certain orders with little time left, so recall is the fairer
  headline.
- Serving the averaged model explains each order with the mean of TreeSHAP and linear
  SHAP values, which still add up to its log-odds.
- The served model is refitted by hand. Automating it is v2.
