"""Explanations add up, and the evaluation report builds. Validation only: CI never scores test."""

import numpy as np
import pytest

from opspilot.features.load import CATEGORICAL_FEATURES, FEATURES, TARGET
from opspilot.models.calibrate import IsotonicCalibrator, SigmoidCalibrator
from opspilot.models.evaluate import (
    CALIBRATED,
    FINAL,
    LABELS,
    MIN_SLICE,
    period_for,
    results_table,
    score,
    slice_table,
    write_report,
)
from opspilot.models.explain import contributions, top_reasons
from opspilot.models.train import DEFAULT_PARAMS, fit_xgboost


@pytest.fixture(scope="module")
def bundle(features):
    fit_rows, _ = period_for("validation").masks(features)
    model = fit_xgboost(features[fit_rows], DEFAULT_PARAMS)
    return {"model": model, "features": list(FEATURES), "params": DEFAULT_PARAMS, "version": "test"}


@pytest.fixture(scope="module")
def calibrators(bundle, features):
    raw = bundle["model"].predict_proba(features[FEATURES])[:, 1]
    kinds = {"isotonic": IsotonicCalibrator, "sigmoid": SigmoidCalibrator}
    return {
        strategy: kinds[strategy.split("_")[1]]().fit(raw, features[TARGET])
        for strategy in CALIBRATED.values()
    }


def test_contributions_add_up_to_the_raw_log_odds(bundle, features):
    rows = features.head(50)
    effects = contributions(bundle, rows)
    raw = bundle["model"].predict_proba(rows[FEATURES])[:, 1]
    assert list(effects.columns) == [*FEATURES, "base"]
    assert np.allclose(effects.sum(axis=1), np.log(raw / (1 - raw)), atol=1e-4)


def test_top_reasons_name_input_features_not_encoded_columns(bundle, features):
    reasons = top_reasons(bundle, features.head(5), k=3)
    assert len(reasons) == 5
    for order in reasons:
        assert len(order) == 3
        assert all(r["feature"] in FEATURES for r in order)
        for r in order:
            if r["feature"] in CATEGORICAL_FEATURES:
                assert r["value"] is None or isinstance(r["value"], str)


def test_every_model_scores_the_split(bundle, calibrators, features):
    scores = score(features, "validation", bundle, calibrators)
    assert set(LABELS) <= set(scores.columns)
    assert set(CALIBRATED) <= set(scores.columns)
    assert scores[list(LABELS)].apply(lambda p: p.between(0, 1).all()).all()
    assert len(results_table(scores)) == len(LABELS)


def test_small_slices_are_flagged(bundle, calibrators, features):
    table = slice_table(score(features, "validation", bundle, calibrators), "customer_state")
    small = table[table.Orders < MIN_SLICE]
    assert (small.Reliable != "yes").all()


def test_report_builds_with_every_section_and_figure(bundle, calibrators, features, tmp_path):
    scores = score(features, "validation", bundle, calibrators)
    _, rows = period_for("validation").masks(features)
    report = write_report("validation", scores, scores, features[rows], features, bundle, tmp_path)
    text = report.read_text()
    for section in ["## Results", "## Capacity", "## Calibration", "## Slices", "## Explanations"]:
        assert section in text
    for figure in ["pr_curves", "capacity", "calibration", "risk_deciles", "shap_importance"]:
        assert (tmp_path / "figures" / f"{figure}.png").exists()
    assert FINAL in scores.columns
