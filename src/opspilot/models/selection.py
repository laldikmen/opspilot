"""Fit and score a model on every selection period: the three folds plus validation.

Models are compared on these periods only; the test set is never touched here
(docs/decisions/0004-chronological-split.md).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from opspilot.features.load import OUTCOME_KNOWN_AT, TARGET
from opspilot.models.metrics import evaluate
from opspilot.models.split import selection_periods

# A model takes the rows to fit on and the rows to score, and returns probabilities.
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]

# Averaged across the periods to compare models. Lift rather than PR-AUC, so the
# high-late-rate periods do not dominate.
SELECTION_METRICS = ["lift", "roc_auc", "recall_top10", "brier"]


def predict_on_periods(features: pd.DataFrame, fit_predict: FitPredict) -> dict[str, pd.DataFrame]:
    """Out-of-sample predictions for each period, in time order.

    Each frame has order_id, purchase_date, outcome_known_at, the label as y and
    the prediction as p.
    """
    predictions = {}
    for period in selection_periods():
        fit_rows, score_rows = period.masks(features)
        scored = features.loc[score_rows, ["order_id", "purchase_date", OUTCOME_KNOWN_AT, TARGET]]
        predictions[period.name] = scored.rename(columns={TARGET: "y"}).assign(
            p=fit_predict(features[fit_rows], features[score_rows])
        )
    return predictions


def evaluate_periods(predictions: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    return {name: evaluate(frame.y, frame.p) for name, frame in predictions.items()}


def selection_summary(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """The mean of each selection metric across the periods."""
    return {
        f"selection_{metric}": float(np.mean([r[metric] for r in results.values()]))
        for metric in SELECTION_METRICS
    }
