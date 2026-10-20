"""XGBoost fitting, the candidate list, and the model bundle."""

import numpy as np
import pytest

from opspilot.models.bundle import build_bundle, load_bundle, predict_proba, save_bundle
from opspilot.models.calibrate import fit_calibrator
from opspilot.models.split import assign_split
from opspilot.models.train import DEFAULT_PARAMS, MAX_ROUNDS, candidates, fit_xgboost


@pytest.fixture(scope="module")
def train(features):
    return features[assign_split(features.purchase_date) == "train"]


@pytest.fixture(scope="module")
def model(train):
    return fit_xgboost(train, DEFAULT_PARAMS)


def test_early_stopping_picks_fewer_trees_than_the_maximum(model):
    assert model[-1].n_estimators < MAX_ROUNDS


def test_xgboost_is_deterministic(train, model, features):
    again = fit_xgboost(train, DEFAULT_PARAMS)
    assert np.array_equal(model.predict_proba(features)[:, 1], again.predict_proba(features)[:, 1])


def test_candidates_start_from_defaults_and_are_reproducible():
    first, second = candidates(5), candidates(5)
    assert first[0] == DEFAULT_PARAMS
    assert first == second
    assert len(first) == 5


def test_bundle_round_trip_gives_the_same_calibrated_probabilities(tmp_path, model, features):
    raw = model.predict_proba(features)[:, 1]
    calibrator = fit_calibrator("sigmoid", raw, features.is_late)
    bundle = build_bundle(model, calibrator, "pooled_sigmoid", DEFAULT_PARAMS, "0" * 40)

    loaded = load_bundle(save_bundle(bundle, tmp_path / "bundle.joblib"))
    p = predict_proba(loaded, features)
    assert np.allclose(p, predict_proba(bundle, features))
    assert np.all((p > 0) & (p < 1))
    assert loaded["version"].endswith("+0000000")
