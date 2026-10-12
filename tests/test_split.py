"""No date overlap between splits; test after validation after train."""

from datetime import date, timedelta

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
