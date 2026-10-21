"""Baselines: training prior, two one-feature rules, logistic regression.

Run with `make train`. Each baseline is fitted and scored on the three
expanding-window folds and on validation, and logged to MLflow. The test set
is not touched here.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from opspilot.config import FOLD_STARTS, RANDOM_SEED, SPLITS
from opspilot.features.load import FEATURES, TARGET, load_features
from opspilot.features.preprocess import build_preprocessor
from opspilot.models.selection import (
    FitPredict,
    evaluate_periods,
    predict_on_periods,
    selection_summary,
)
from opspilot.models.tracking import git_commit, log_period_metrics, start_experiment


def prior(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    """Everyone gets the late rate of the rows it was fitted on."""
    return np.full(len(score), fit[TARGET].mean())


def seller_rate_rule(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    """Rank by the primary seller's smoothed late rate as of t0. Nothing is fitted."""
    return score["seller_late_rate_smoothed"].to_numpy()


def route_rate_rule(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    """Rank by the late rate of the order's route as of t0. Nothing is fitted.

    The strongest single-feature rule found on the folds and validation.
    """
    return score["route_late_rate_smoothed"].fillna(fit[TARGET].mean()).to_numpy()


def logistic_regression_on(features: list[str]) -> FitPredict:
    """Logistic regression with train-only preprocessing, on the given features."""

    def fit_predict(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
        model = make_pipeline(
            build_preprocessor(scale=True, features=features),
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        )
        model.fit(fit[features], fit[TARGET])
        return model.predict_proba(score[features])[:, 1]

    return fit_predict


logistic_regression = logistic_regression_on(FEATURES)


BASELINES: dict[str, FitPredict] = {
    "prior": prior,
    "seller_rate_rule": seller_rate_rule,
    "route_rate_rule": route_rate_rule,
    "logistic_regression": logistic_regression,
}


def run_params(model: str) -> dict:
    train_start, train_end = SPLITS["train"]
    validation_start, validation_end = SPLITS["validation"]
    return {
        "model": model,
        "n_features": len(FEATURES),
        "train_period": f"{train_start} to {train_end}",
        "validation_period": f"{validation_start} to {validation_end}",
        "fold_starts": ", ".join(str(d) for d in FOLD_STARTS),
    }


def main() -> None:
    features = load_features()
    start_experiment()
    commit = git_commit()

    rows = []
    for name, fit_predict in BASELINES.items():
        results = evaluate_periods(predict_on_periods(features, fit_predict))
        summary = selection_summary(results)
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"git_commit": commit, "kind": "baseline"})
            mlflow.log_params(run_params(name))
            log_period_metrics(results)
            mlflow.log_metrics(summary)
        rows.append(
            {
                "model": name,
                **{f"{p}_lift": r["lift"] for p, r in results.items()},
                "validation_pr_auc": results["validation"]["pr_auc"],
                **summary,
            }
        )

    table = pd.DataFrame(rows).set_index("model")
    print(table.round(3).T.to_string())
    print(f"\nLogged {len(rows)} runs to MLflow experiment, commit {commit}.")


if __name__ == "__main__":
    main()
