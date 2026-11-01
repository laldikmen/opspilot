"""v1.1: refit the served model as of a date, using only outcomes known by then.

Run with `make retrain`, monthly in practice: `make retrain ARGS="--as-of 2018-07-01"`.
The recipe averages logistic regression and XGBoost and uses no calibrator; why is
in reports/v1.1/walk_forward.md. The frozen v1 model is not touched.
"""

import argparse

import pandas as pd

from opspilot.config import OBSERVATION_END, SERVED_BUNDLE_PATH, SPLITS
from opspilot.features.load import FEATURES, load_features
from opspilot.models.baselines import fit_logistic_regression
from opspilot.models.bundle import load_bundle, save_bundle
from opspilot.models.ensemble import AveragedModel
from opspilot.models.split import known_before
from opspilot.models.tracking import git_commit
from opspilot.models.train import fit_xgboost
from opspilot.models.walk_forward import band_cutoffs, recent_scores, run


def fit_rows(features: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(features.purchase_date)
    rows = features[(dates >= pd.Timestamp(SPLITS["train"][0])) & (dates < as_of)]
    return rows[known_before(rows, as_of)]


def fit_recipe(features: pd.DataFrame, params: dict, as_of: pd.Timestamp) -> AveragedModel:
    fit = fit_rows(features, as_of)
    logistic = fit_logistic_regression(fit)
    reference = logistic[0].transform(fit[FEATURES]).mean()
    return AveragedModel(logistic, fit_xgboost(fit, params), reference)


def build_bundle(features: pd.DataFrame, params: dict, as_of: pd.Timestamp) -> dict:
    scored = run(features, params)
    average = scored[scored.model == "average"]
    commit = git_commit()
    return {
        "version": f"1.1-{as_of:%Y-%m-%d}+{commit[:7]}",
        "model": fit_recipe(features, params, as_of),
        "calibrator": None,
        "calibration": "none",
        "risk_bands": band_cutoffs(recent_scores(average, as_of)),
        "features": list(FEATURES),
        "params": params,
        "train_period": f"{SPLITS['train'][0]} to {as_of:%Y-%m-%d}, known outcomes only",
        "git_commit": commit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", default=str(OBSERVATION_END), help="YYYY-MM-DD")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of)
    bundle = build_bundle(load_features(), load_bundle()["params"], as_of)
    path = save_bundle(bundle, SERVED_BUNDLE_PATH)
    print(f"Saved {bundle['version']} to {path}; risk bands {bundle['risk_bands']}")


if __name__ == "__main__":
    main()
