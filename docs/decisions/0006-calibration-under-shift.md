# 0006 · Calibration under shift

Status: open · to be decided in Week 5

## Context

The API returns probabilities, so they have to mean what they say. The usual
recipe is to fit a calibrator on validation, but here validation is the
anomalous quarter: its late rate is 12.88%, against 5.65% in train and 4.05%
in test ([0004](0004-chronological-split.md)). A calibrator fitted there learns
a base rate about three times the one it will be used on.

## Options to compare

1. Calibrator fitted on validation.
2. Calibrator fitted on out-of-fold predictions pooled across the 2017
   expanding-window folds and validation.
3. No calibration.

Isotonic and sigmoid are both tried for options 1 and 2.

## Decision

Not made yet. It is made in Week 5 without looking at the test set, and the
choice and its reasoning are recorded here. All three options are still scored
on test in Week 7 and reported side by side in `reports/evaluation.md`.
