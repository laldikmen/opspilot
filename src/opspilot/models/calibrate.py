"""Calibration under shift: fitted on validation, on pooled out-of-fold predictions, or none.

The periods models are scored on do not share a late rate (2.9% to 12.9%), so a
calibrator learns the base rate of whatever it was fitted on. backtest() measures
how each strategy carries over to the next period, using only earlier periods
(docs/decisions/0006-calibration-under-shift.md).
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from opspilot.models.metrics import brier, expected_calibration_error

METHODS = ["isotonic", "sigmoid"]
# Which earlier predictions a calibrator is fitted on. "last" is the most recent
# scored period, which for the final model is validation; "pooled" is every
# earlier scored period together.
SOURCES = ["last", "pooled"]
STRATEGIES = ["none"] + [f"{source}_{method}" for source in SOURCES for method in METHODS]

_EPS = 1e-6


class SigmoidCalibrator:
    """Platt scaling: a logistic regression on the logit of the raw probability."""

    def fit(self, p, y) -> "SigmoidCalibrator":
        self._model = LogisticRegression(C=1e6).fit(_logit(p), np.asarray(y))
        return self

    def predict(self, p) -> np.ndarray:
        return self._model.predict_proba(_logit(p))[:, 1]


class IsotonicCalibrator:
    """A monotone step function from raw to calibrated probability."""

    def fit(self, p, y) -> "IsotonicCalibrator":
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._model.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, p) -> np.ndarray:
        return self._model.predict(np.asarray(p, dtype=float))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method: str, p, y):
    calibrators = {"isotonic": IsotonicCalibrator, "sigmoid": SigmoidCalibrator}
    return calibrators[method]().fit(p, y)


def calibration_rows(
    predictions: dict[str, pd.DataFrame], strategy: str, upto: int
) -> pd.DataFrame:
    """The predictions a strategy is fitted on, using only periods before index `upto`."""
    names = list(predictions)[:upto]
    source = strategy.split("_")[0]
    if source == "last":
        return predictions[names[-1]]
    return pd.concat([predictions[name] for name in names])


def apply_strategy(strategy: str, fit_rows: pd.DataFrame | None, p) -> np.ndarray:
    if strategy == "none":
        return np.asarray(p, dtype=float)
    method = strategy.split("_")[1]
    return fit_calibrator(method, fit_rows.p, fit_rows.y).predict(p)


def backtest(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score every strategy on each period, fitted only on the periods before it.

    `predictions` are out-of-sample predictions per period in time order, as made
    by selection.predict_on_periods. The first period has nothing before it and
    is only used for fitting.
    """
    rows = []
    names = list(predictions)
    for i in range(1, len(names)):
        target = predictions[names[i]]
        for strategy in STRATEGIES:
            fit_rows = None if strategy == "none" else calibration_rows(predictions, strategy, i)
            p = apply_strategy(strategy, fit_rows, target.p)
            rows.append(
                {
                    "strategy": strategy,
                    "period": names[i],
                    "late_rate": float(target.y.mean()),
                    "mean_prediction": float(np.mean(p)),
                    "brier": brier(target.y, p),
                    "ece": expected_calibration_error(target.y, p),
                }
            )
    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Mean Brier and ECE per strategy across the periods, best first."""
    return (
        results.groupby("strategy")[["brier", "ece"]]
        .mean()
        .sort_values("brier")
        .rename(columns={"brier": "mean_brier", "ece": "mean_ece"})
    )
