"""XGBoost training and tuning; every run is logged to MLflow.

Run with `make train`, after the baselines. Each candidate is scored on the
three folds plus validation. The best one is refitted on train, the calibration
strategies are compared with a backtest, and the result is saved as one bundle.
The test set is not touched here.
"""

import json

import joblib
import mlflow
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBClassifier

from opspilot.config import ARTIFACTS_DIR, RANDOM_SEED
from opspilot.features.load import FEATURES, TARGET, load_features
from opspilot.features.preprocess import build_preprocessor
from opspilot.models.bundle import build_bundle, save_bundle
from opspilot.models.calibrate import (
    STRATEGIES,
    backtest,
    calibration_rows,
    fit_calibrator,
    summarize_backtest,
)
from opspilot.models.selection import (
    FitPredict,
    evaluate_periods,
    predict_on_periods,
    selection_summary,
)
from opspilot.models.split import assign_split
from opspilot.models.tracking import git_commit, log_period_metrics, start_experiment

N_TRIALS = 40
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50
# Early stopping watches the most recent 15% of the rows a model is fitted on.
EARLY_STOPPING_HOLDOUT = 0.15

# Trial 0. Also the fallback if the search is ever cut.
DEFAULT_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
SEARCH_SPACE = {
    "max_depth": randint(2, 9),
    "learning_rate": loguniform(0.01, 0.2),
    "min_child_weight": loguniform(1, 50),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.4, 0.6),
    "reg_lambda": loguniform(0.1, 20),
}


def candidates(n_trials: int) -> list[dict]:
    sampled = ParameterSampler(SEARCH_SPACE, n_iter=n_trials - 1, random_state=RANDOM_SEED)
    return [DEFAULT_PARAMS] + [
        {k: int(v) if k == "max_depth" else round(float(v), 4) for k, v in params.items()}
        for params in sampled
    ]


def xgboost(params: dict, n_estimators: int, **kwargs) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **params,
        **kwargs,
    )


def fit_xgboost(fit: pd.DataFrame, params: dict) -> Pipeline:
    """Find the number of trees on the latest rows of `fit`, then refit on all of them."""
    fit = fit.sort_values(["t0", "order_id"])
    cut = int(len(fit) * (1 - EARLY_STOPPING_HOLDOUT))
    early, late = fit.iloc[:cut], fit.iloc[cut:]

    preprocessor = build_preprocessor(scale=False).fit(early[FEATURES])
    probe = xgboost(params, MAX_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    probe.fit(
        preprocessor.transform(early[FEATURES]),
        early[TARGET],
        eval_set=[(preprocessor.transform(late[FEATURES]), late[TARGET])],
        verbose=False,
    )
    model = make_pipeline(
        build_preprocessor(scale=False), xgboost(params, probe.best_iteration + 1)
    )
    return model.fit(fit[FEATURES], fit[TARGET])


def xgboost_fit_predict(params: dict) -> FitPredict:
    def fit_predict(fit: pd.DataFrame, score: pd.DataFrame):
        return fit_xgboost(fit, params).predict_proba(score[FEATURES])[:, 1]

    return fit_predict


def evaluate_candidate(features: pd.DataFrame, params: dict) -> dict:
    predictions = predict_on_periods(features, xgboost_fit_predict(params))
    results = evaluate_periods(predictions)
    return {
        "params": params,
        "predictions": predictions,
        "results": results,
        "summary": selection_summary(results),
    }


def final_calibrators(predictions: dict[str, pd.DataFrame]) -> dict:
    """Every strategy, fitted on all four periods, ready to apply to the test period.

    "last_*" is fitted on validation; "pooled_*" on the folds plus validation.
    """
    calibrators = {"none": None}
    for strategy in STRATEGIES[1:]:
        rows = calibration_rows(predictions, strategy, upto=len(predictions))
        calibrators[strategy] = fit_calibrator(strategy.split("_")[1], rows.p, rows.y)
    return calibrators


def main() -> None:
    features = load_features()
    start_experiment()
    commit = git_commit()
    tags = {"git_commit": commit, "kind": "xgboost"}

    trials = []
    with mlflow.start_run(run_name="xgboost_search"):
        mlflow.set_tags(tags)
        mlflow.log_params({"n_trials": N_TRIALS, "search_space": json.dumps(list(SEARCH_SPACE))})
        for i, params in enumerate(candidates(N_TRIALS)):
            trial = evaluate_candidate(features, params)
            with mlflow.start_run(run_name=f"trial_{i:02d}", nested=True):
                mlflow.set_tags(tags)
                mlflow.log_params(params)
                log_period_metrics(trial["results"])
                mlflow.log_metrics(trial["summary"])
            trials.append(trial)
            print(
                f"trial {i:02d}  selection_lift {trial['summary']['selection_lift']:.3f}  {params}"
            )
        best = max(trials, key=lambda t: t["summary"]["selection_lift"])
        mlflow.log_metrics(best["summary"])

    backtest_results = backtest(best["predictions"])
    backtest_summary = summarize_backtest(backtest_results)
    chosen = backtest_summary.index[0]

    train = features[assign_split(features.purchase_date) == "train"]
    model = fit_xgboost(train, best["params"])
    calibrators = final_calibrators(best["predictions"])
    bundle = build_bundle(model, calibrators[chosen], chosen, best["params"], commit)
    path = save_bundle(bundle)
    joblib.dump(calibrators, ARTIFACTS_DIR / "calibrators.joblib")
    backtest_results.to_csv(ARTIFACTS_DIR / "calibration_backtest.csv", index=False)

    with mlflow.start_run(run_name="xgboost_final"):
        mlflow.set_tags({**tags, "calibration": chosen, "bundle_version": bundle["version"]})
        mlflow.log_params({**best["params"], "calibration": chosen})
        log_period_metrics(best["results"])
        mlflow.log_metrics(best["summary"])
        mlflow.log_metrics(
            {f"backtest_brier_{s}": v for s, v in backtest_summary.mean_brier.items()}
        )
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "calibration_backtest.csv"))

    print("\nBest parameters:", best["params"])
    print(
        pd.DataFrame(best["results"])
        .T[["late_rate", "pr_auc", "lift", "roc_auc", "recall_top10"]]
        .round(3)
    )
    print("\nCalibration backtest (fitted on earlier periods only):")
    print(backtest_results.pivot(index="strategy", columns="period", values="brier").round(4))
    print(backtest_summary.round(4))
    print(f"\nChosen calibration: {chosen}. Bundle {bundle['version']} saved to {path}")


if __name__ == "__main__":
    main()
