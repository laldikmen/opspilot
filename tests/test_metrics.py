"""Metrics match small hand-computed cases."""

import numpy as np
import pytest

from opspilot.models.metrics import (
    brier,
    evaluate,
    expected_calibration_error,
    lift,
    pr_auc,
    precision_recall_at_top,
    roc_auc,
)

# Ranked by score: late, on time, late, on time.
Y = [1, 0, 1, 0]
P = [0.9, 0.8, 0.7, 0.1]


def test_pr_auc_is_average_precision_at_each_late_order():
    # Precision is 1/1 at the first late order and 2/3 at the second.
    assert pr_auc(Y, P) == pytest.approx((1 + 2 / 3) / 2)


def test_lift_divides_pr_auc_by_the_late_rate():
    assert lift(Y, P) == pytest.approx(((1 + 2 / 3) / 2) / 0.5)


def test_roc_auc_counts_correctly_ordered_pairs():
    # Pairs (late, on time): (0.9, 0.8) yes, (0.9, 0.1) yes, (0.7, 0.8) no, (0.7, 0.1) yes.
    assert roc_auc(Y, P) == pytest.approx(3 / 4)


def test_brier_is_mean_squared_error():
    assert brier(Y, P) == pytest.approx((0.1**2 + 0.8**2 + 0.3**2 + 0.1**2) / 4)


def test_top_10_percent_of_ten_orders_is_the_single_riskiest():
    y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    p = np.linspace(1.0, 0.1, 10)
    assert precision_recall_at_top(y, p) == pytest.approx((1.0, 0.5))
    y_top_on_time = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert precision_recall_at_top(y_top_on_time, p) == pytest.approx((0.0, 0.0))


def test_a_constant_score_is_worth_exactly_the_late_rate():
    y = [1, 0, 0, 0] * 5
    p = [0.3] * 20
    precision, recall = precision_recall_at_top(y, p)
    assert precision == pytest.approx(0.25)
    assert recall == pytest.approx(0.10)
    assert pr_auc(y, p) == pytest.approx(0.25)
    assert lift(y, p) == pytest.approx(1.0)


def test_ece_averages_the_gap_in_each_bin():
    # Two bins, half the orders each, both off by 0.1.
    assert expected_calibration_error([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]) == pytest.approx(0.1)


def test_ece_is_zero_when_predictions_match_the_observed_rate():
    assert expected_calibration_error([1, 0, 0, 0], [0.25] * 4) == pytest.approx(0.0)


def test_ece_puts_a_probability_of_one_in_the_last_bin():
    assert expected_calibration_error([1, 1], [1.0, 0.95]) == pytest.approx(0.025)


def test_evaluate_reports_every_metric():
    assert set(evaluate(Y, P)) == {
        "n",
        "late_rate",
        "pr_auc",
        "lift",
        "roc_auc",
        "precision_top10",
        "recall_top10",
        "brier",
        "ece",
    }
