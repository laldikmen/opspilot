"""Baselines: training prior, seller smoothed late-rate rule, logistic regression.

Run with `make train`. Each baseline is fitted and scored on the three
expanding-window folds and on validation, and logged to MLflow. The test set
is not touched here.
"""

from collections.abc import Callable

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from opspilot.config import FOLD_STARTS, RANDOM_SEED, SPLITS
from opspilot.features.load import FEATURES, TARGET, load_features
from opspilot.features.preprocess import build_preprocessor
from opspilot.models.metrics import evaluate
from opspilot.models.split import selection_periods
from opspilot.models.tracking import git_commit, log_period_metrics, start_experiment

# A baseline takes the rows to fit on and the rows to score, and returns probabilities.
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]

# Averaged across the folds and validation to compare models. Lift rather than
# PR-AUC, so the high-late-rate periods do not dominate (docs/decisions/0004).
SELECTION_METRICS = ["lift", "roc_auc", "recall_top10", "brier"]


def prior(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    """Everyone gets the late rate of the rows it was fitted on."""
    return np.full(len(score), fit[TARGET].mean())


def seller_rate_rule(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    """Rank by the primary seller's smoothed late rate as of t0. Nothing is fitted."""
    return score["seller_late_rate_smoothed"].to_numpy()


def logistic_regression(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    model = make_pipeline(
        build_preprocessor(scale=True),
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
    )
    model.fit(fit[FEATURES], fit[TARGET])
    return model.predict_proba(score[FEATURES])[:, 1]


BASELINES: dict[str, FitPredict] = {
    "prior": prior,
    "seller_rate_rule": seller_rate_rule,
    "logistic_regression": logistic_regression,
}


def score_on_periods(features: pd.DataFrame, fit_predict: FitPredict) -> dict[str, dict]:
    """Fit and score once per selection period; metrics keyed by period name."""
    results = {}
    for period in selection_periods():
        fit_rows, score_rows = period.masks(features.purchase_date)
        p = fit_predict(features[fit_rows], features[score_rows])
        results[period.name] = evaluate(features.loc[score_rows, TARGET], p)
    return results


def selection_summary(results: dict[str, dict]) -> dict[str, float]:
    """The mean of each selection metric across the folds and validation."""
    return {
        f"selection_{metric}": float(np.mean([r[metric] for r in results.values()]))
        for metric in SELECTION_METRICS
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
        results = score_on_periods(features, fit_predict)
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
