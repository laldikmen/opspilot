"""Calibrators map scores to observed rates; the backtest only fits on earlier periods."""

import numpy as np
import pandas as pd
import pytest

from opspilot.models.calibrate import (
    STRATEGIES,
    backtest,
    calibration_rows,
    fit_calibrator,
    summarize_backtest,
)


def period(late_rate: float, n: int = 1000, seed: int = 0, signal: float = 0.02) -> pd.DataFrame:
    """Scores around 0.05 whatever the true late rate; `signal` sets how well they rank."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < late_rate).astype(int)
    p = np.clip(0.05 + signal * y + rng.normal(0, 0.01, n), 0.001, 0.999)
    return pd.DataFrame({"y": y, "p": p})


PREDICTIONS = {
    "fold_1": period(0.03, seed=1),
    "fold_2": period(0.05, seed=2),
    "fold_3": period(0.10, seed=3),
    "validation": period(0.13, seed=4),
}


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_learns_the_observed_rate(method):
    rows = period(0.20, n=4000)
    calibrated = fit_calibrator(method, rows.p, rows.y).predict(rows.p)
    assert calibrated.mean() == pytest.approx(rows.y.mean(), abs=0.01)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_keeps_the_ranking(method):
    rows = period(0.20, n=4000)
    calibrated = fit_calibrator(method, rows.p, rows.y).predict(np.sort(rows.p))
    assert np.all(np.diff(calibrated) >= -1e-12)


def test_calibration_rows_use_only_periods_before_the_target():
    assert calibration_rows(PREDICTIONS, "last_isotonic", upto=2) is PREDICTIONS["fold_2"]
    pooled = calibration_rows(PREDICTIONS, "pooled_sigmoid", upto=2)
    assert len(pooled) == len(PREDICTIONS["fold_1"]) + len(PREDICTIONS["fold_2"])


def test_backtest_scores_every_strategy_on_every_later_period():
    results = backtest(PREDICTIONS)
    assert set(results.strategy) == set(STRATEGIES)
    assert list(results.period.unique()) == ["fold_2", "fold_3", "validation"]


def test_with_uninformative_scores_last_carries_over_the_previous_late_rate():
    # Scores that say nothing about the label: a calibrator can only learn the base
    # rate of what it was fitted on, so validation inherits fold_3's rate.
    flat = {
        name: period(rate, seed=i, signal=0.0)
        for i, (name, rate) in enumerate(
            [("fold_1", 0.03), ("fold_2", 0.05), ("fold_3", 0.10), ("validation", 0.13)]
        )
    }
    results = backtest(flat)
    last = results[(results.strategy == "last_sigmoid") & (results.period == "validation")]
    assert last.mean_prediction.iloc[0] == pytest.approx(flat["fold_3"].y.mean(), abs=0.01)


def test_summary_ranks_strategies_by_mean_brier():
    summary = summarize_backtest(backtest(PREDICTIONS))
    assert list(summary.mean_brier) == sorted(summary.mean_brier)
