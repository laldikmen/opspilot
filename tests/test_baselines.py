"""Preprocessing is fitted on train only; the baselines run on every selection period."""

import numpy as np
import pytest

from opspilot.features.load import FEATURES, TARGET
from opspilot.features.preprocess import TOP_CATEGORIES, build_preprocessor
from opspilot.models.baselines import BASELINES
from opspilot.models.selection import evaluate_periods, predict_on_periods, selection_summary
from opspilot.models.split import assign_split


@pytest.mark.parametrize("scale", [False, True])
def test_missing_values_are_filled_from_the_fitted_rows_only(features, scale):
    split = assign_split(features.purchase_date)
    fit = features[split == "train"].copy()
    score = features[split == "validation"].copy()
    # Make the two periods disagree, then hide one value in the scored rows.
    fit["n_items"] = 1
    score["n_items"] = 100
    score.loc[score.index[0], "n_items"] = np.nan

    preprocessor = build_preprocessor(scale=scale).fit(fit[FEATURES])
    filled = preprocessor.transform(score[FEATURES])["n_items"]
    expected = preprocessor.transform(fit[FEATURES])["n_items"].iloc[0]
    assert filled.iloc[0] == pytest.approx(expected)


def test_rare_categories_are_grouped(features):
    encoded = build_preprocessor(scale=True).fit_transform(features[FEATURES])
    category_columns = [c for c in encoded.columns if c.startswith("main_category_")]
    assert len(category_columns) <= TOP_CATEGORIES + 1
    assert "main_category_infrequent_sklearn" in category_columns
    assert not encoded.isna().any().any()


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_scores_every_selection_period(features, name):
    results = evaluate_periods(predict_on_periods(features, BASELINES[name]))
    assert list(results) == ["fold_1", "fold_2", "fold_3", "validation"]
    for metrics in results.values():
        assert 0 < metrics["pr_auc"] <= 1
        assert 0 <= metrics["brier"] <= 1
    assert np.isfinite(list(selection_summary(results).values())).all()


def test_prior_baseline_predicts_the_fitted_late_rate(features):
    train = features[assign_split(features.purchase_date) == "train"]
    p = BASELINES["prior"](train, features.head(5))
    assert p == pytest.approx([train[TARGET].mean()] * 5)
