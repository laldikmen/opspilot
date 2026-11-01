"""Month-by-month fits only use what was known, and the level shift does what it says."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from opspilot.models.split import recency_weights
from opspilot.models.train import DEFAULT_PARAMS
from opspilot.models.walk_forward import (
    MODELS,
    level_rows,
    level_shift,
    monthly_ranking,
    months,
    paired_months,
    run,
    with_corrections,
)


def test_level_shift_matches_the_observed_late_rate():
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(2, 20, 2000), 1e-3, 1)
    y = (rng.random(2000) < 0.02).astype(int)
    shift = level_shift(p, y)
    logits = np.log(p / (1 - p))
    assert expit(logits + shift).mean() == pytest.approx(y.mean(), abs=1e-6)
    assert shift < 0


def test_level_shift_keeps_the_ranking():
    p = np.array([0.1, 0.2, 0.3])
    shifted = expit(np.log(p / (1 - p)) - 1.0)
    assert list(np.argsort(shifted)) == list(np.argsort(p))


def test_level_rows_use_only_the_window_and_known_outcomes():
    start = pd.Timestamp("2018-04-01")
    history = pd.DataFrame(
        {
            "purchase_date": pd.to_datetime(
                ["2017-12-01", "2018-01-15", "2018-02-15", "2018-02-15", "2018-03-15"]
            ),
            "outcome_known_at": pd.to_datetime(
                ["2018-01-01", "2018-02-01", "2018-03-01", "2018-05-01", "2018-03-20"]
            ),
        }
    )
    assert list(level_rows(history, start).index) == [1, 2]


def test_recency_weights_halve_every_half_life():
    fit = pd.DataFrame({"t0": pd.to_datetime(["2018-01-01", "2018-04-01"])})
    weights = recency_weights(fit, 90)
    assert weights[1] == 1
    assert weights[0] == pytest.approx(0.5)
    assert recency_weights(fit, None) is None


@pytest.fixture(scope="module")
def predictions(features):
    return with_corrections(run(features, DEFAULT_PARAMS))


def test_every_model_scores_every_month_once(predictions):
    assert set(predictions.model) == set(MODELS)
    assert list(sorted(predictions.month.unique())) == [str(m) for m in months()]
    assert not predictions.duplicated(["order_id", "model"]).any()


def test_corrections_only_appear_once_history_exists(predictions):
    first = predictions[predictions.month == str(months()[0])]
    assert first.p_level.isna().all()


def test_monthly_tables_build(predictions):
    monthly = monthly_ranking(predictions)
    row = paired_months(monthly, "xgboost", "logistic_regression", "lift")
    assert row["95% low"] <= row["Mean difference"] <= row["95% high"]
