"""Assign train/validation/test from the dates in config.py, plus expanding-window folds.

The split lives here rather than in the warehouse, so moving a boundary is a
config edit (docs/decisions/0004-chronological-split.md).
"""

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from opspilot.config import FOLD_STARTS, SPLITS


def assign_split(purchase_date: pd.Series) -> pd.Series:
    """'train', 'validation' or 'test' for each purchase date; missing outside every range."""
    dates = pd.to_datetime(purchase_date)
    split = pd.Series(pd.NA, index=purchase_date.index, dtype="string")
    for name, (start, end) in SPLITS.items():
        split[dates.between(pd.Timestamp(start), pd.Timestamp(end))] = name
    return split


@dataclass(frozen=True)
class Period:
    """Fit on orders purchased in [fit_start, fit_end], score those in [score_start, score_end]."""

    name: str
    fit_start: date
    fit_end: date
    score_start: date
    score_end: date

    def masks(self, purchase_date: pd.Series) -> tuple[pd.Series, pd.Series]:
        dates = pd.to_datetime(purchase_date)
        fit = dates.between(pd.Timestamp(self.fit_start), pd.Timestamp(self.fit_end))
        score = dates.between(pd.Timestamp(self.score_start), pd.Timestamp(self.score_end))
        return fit, score


def expanding_window_folds() -> list[Period]:
    """Three folds inside train, each fitted on everything before the months it scores."""
    train_start, train_end = SPLITS["train"]
    starts = [*FOLD_STARTS, train_end + timedelta(days=1)]
    return [
        Period(
            f"fold_{i}",
            train_start,
            start - timedelta(days=1),
            start,
            next_start - timedelta(days=1),
        )
        for i, (start, next_start) in enumerate(zip(starts, starts[1:], strict=False), start=1)
    ]


def selection_periods() -> list[Period]:
    """The periods models are compared on: the three folds plus validation. Never test."""
    train_start, train_end = SPLITS["train"]
    validation_start, validation_end = SPLITS["validation"]
    validation = Period("validation", train_start, train_end, validation_start, validation_end)
    return [*expanding_window_folds(), validation]
