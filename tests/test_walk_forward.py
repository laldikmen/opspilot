"""Month-by-month fits only use what was known, and the level shift does what it says."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from opspilot.features.load import FEATURES
from opspilot.models.bundle import predict_proba
from opspilot.models.explain import contributions, top_reasons
from opspilot.models.split import recency_weights
from opspilot.models.train import DEFAULT_PARAMS
from opspilot.models.walk_forward import (
    MODELS,
    band_check,
    band_cutoffs,
    fit_recipe,
    fit_rows,
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


def test_band_cutoffs_mark_the_top_10_and_next_20_percent():
    scores = np.arange(1000) / 1000
    cutoffs = band_cutoffs(scores)
    assert (scores >= cutoffs["high"]).mean() == pytest.approx(0.1, abs=0.002)
    assert (scores >= cutoffs["medium"]).mean() == pytest.approx(0.3, abs=0.002)


def test_band_check_covers_the_months_with_recent_orders(features):
    table = band_check(features, DEFAULT_PARAMS)
    assert len(table) > 0
    assert table["share high"].between(0, 1).all()


@pytest.fixture(scope="module")
def recipe_bundle(features):
    as_of = pd.Timestamp("2018-04-01")
    return {
        "model": fit_recipe(features, DEFAULT_PARAMS, as_of),
        "features": list(FEATURES),
        "calibrator": None,
        "version": "test",
    }


def test_recipe_only_learns_outcomes_known_on_its_date(features):
    as_of = pd.Timestamp("2018-04-01")
    rows = fit_rows(features, as_of)
    assert (rows.outcome_known_at < as_of).all()
    assert (pd.to_datetime(rows.purchase_date) < as_of).all()


def test_average_lies_between_its_two_models(recipe_bundle, features):
    model = recipe_bundle["model"]
    rows = features[FEATURES].head(100)
    a = model.logistic.predict_proba(rows)[:, 1]
    b = model.xgboost.predict_proba(rows)[:, 1]
    p = model.predict_proba(rows)[:, 1]
    assert ((p >= np.minimum(a, b) - 1e-9) & (p <= np.maximum(a, b) + 1e-9)).all()


def test_averaged_explanations_add_up_to_the_log_odds(recipe_bundle, features):
    rows = features.head(50)
    effects = contributions(recipe_bundle, rows)
    p = predict_proba(recipe_bundle, rows)
    assert list(effects.columns) == [*FEATURES, "base"]
    assert np.allclose(effects.sum(axis=1), logit(p), atol=1e-4)


def test_averaged_reasons_name_input_features(recipe_bundle, features):
    for reason in top_reasons(recipe_bundle, features.head(3))[0]:
        assert reason["feature"] in FEATURES
