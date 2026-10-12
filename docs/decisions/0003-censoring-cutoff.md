# 0003 · Censoring cutoff

Status: accepted · 2026-10-12

## Context

Orders purchased shortly before the data was extracted may not have been
delivered yet. If they are kept, recent orders look artificially on time,
because only the fast ones have an outcome. The rule from the plan is: drop
orders purchased after *the last observed delivery date minus the 99th
percentile of delivery time*.

The EDA showed that "the last observed delivery date" cannot be taken as the
maximum of the column. The maximum is 2018-10-17, but deliveries are recorded
at 300 to 450 per weekday until **2018-08-30**, then 27 the next day and never
more than 16 on any day after. Only 86 deliveries are recorded after
2018-08-30. The extract was effectively taken at the end of August 2018.

## Decision

- **Observation end: 2018-08-30**, the last day with at least 100 recorded
  deliveries.
- **Horizon: 46 days**, the 99th percentile of purchase-to-delivery time over
  all labeled orders (46.05 days; the median is 10.2).
- **Cutoff: keep orders purchased on or before 2018-07-15.**
- Also drop orders purchased before 2017-01-01. 2016 holds 329 orders, with
  none in November, and continuous data starts on 2017-01-05.

The values live in `src/opspilot/config.py`; the notebook asserts that they
match what it measures.

## Evidence that the dropped weeks are censored

By purchase week, after the cutoff: the 99th-percentile delivery time falls
from about 31 days to 3.6, weekly order counts collapse from about 1,900 to
116, and the late rate drops to 0%. Slow orders from those weeks are missing
from the data, not on time.

## Consequences

- 10,201 labeled orders after the cutoff and 267 before 2017 are dropped,
  leaving **86,002** labeled orders for modelling.
- Using the 2018-10-17 maximum would have put the cutoff at 2018-09-01 and kept
  every censored week. This is the mistake the rule exists to prevent.
- Some censoring remains. Late orders are the slow tail: 14.3% of them take
  longer than 46 days. For orders purchased in the last weeks before the
  cutoff, the very latest deliveries can still be missing, so the test-set late
  rate is slightly understated. A horizon long enough to remove this (the 99th
  percentile among late orders is 100 days) would leave a test set of under two
  months. Recorded as a limitation instead.
