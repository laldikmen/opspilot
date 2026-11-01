"""v1.1: refit the served model as of a date, using only outcomes known by then.

Run with `make retrain`, monthly in practice: `make retrain ARGS="--as-of 2018-07-01"`.
The recipe averages logistic regression and XGBoost and uses no calibrator; why is
in reports/v1.1/walk_forward.md. The frozen v1 model is not touched.
"""

import argparse

import pandas as pd

from opspilot.config import OBSERVATION_END, SERVED_BUNDLE_PATH, SPLITS
from opspilot.features.load import FEATURES, load_features
from opspilot.models.bundle import load_bundle, save_bundle
from opspilot.models.tracking import git_commit
from opspilot.models.walk_forward import cutoffs_for, fit_recipe


def build_bundle(features: pd.DataFrame, params: dict, as_of: pd.Timestamp) -> dict:
    model = fit_recipe(features, params, as_of)
    commit = git_commit()
    return {
        "version": f"1.1-{as_of:%Y-%m-%d}+{commit[:7]}",
        "model": model,
        "calibrator": None,
        "calibration": "none",
        "risk_bands": cutoffs_for(model, features, as_of),
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
