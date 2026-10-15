"""No date overlap between splits; test after validation after train."""

from datetime import date, timedelta

import pandas as pd

from opspilot.config import CENSORING_CUTOFF, OBSERVATION_END, SPLITS


def test_splits_are_ordered_and_contiguous():
    assert list(SPLITS) == ["train", "validation", "test"]
    bounds = list(SPLITS.values())
    for start, end in bounds:
        assert start <= end
    for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:], strict=False):
        assert next_start == prev_end + timedelta(days=1)


def test_no_split_reaches_past_the_censoring_cutoff():
    assert CENSORING_CUTOFF == date(2018, 7, 15)
    assert CENSORING_CUTOFF < OBSERVATION_END
    assert max(end for _, end in SPLITS.values()) == CENSORING_CUTOFF


def test_dbt_censoring_cutoff_matches_config(dbt_vars):
    assert date.fromisoformat(dbt_vars["censoring_cutoff"]) == CENSORING_CUTOFF


def test_mart_rows_get_one_split_in_date_order(warehouse):
    from opspilot.models.split import assign_split

    dates = warehouse.sql("select purchase_date from ml_late_delivery_features").df().purchase_date
    split = assign_split(dates)
    ranges = {name: (dates[split == name].min(), dates[split == name].max()) for name in SPLITS}
    assert ranges["train"][1] < ranges["validation"][0]
    assert ranges["validation"][1] < ranges["test"][0]
    # Only the sparse months before 2017 are left without a split.
    assert (dates[split.isna()] < pd.Timestamp(SPLITS["train"][0])).all()


def test_folds_expand_inside_train_and_never_reach_test():
    from opspilot.models.split import expanding_window_folds, selection_periods

    folds = expanding_window_folds()
    train_start, train_end = SPLITS["train"]
    assert len(folds) == 3
    for fold in folds:
        assert fold.fit_start == train_start
        assert fold.fit_end + timedelta(days=1) == fold.score_start
        assert fold.score_end <= train_end
    assert [f.fit_end for f in folds] == sorted(f.fit_end for f in folds)
    assert folds[-1].score_end == train_end

    test_start = SPLITS["test"][0]
    assert all(period.score_end < test_start for period in selection_periods())
