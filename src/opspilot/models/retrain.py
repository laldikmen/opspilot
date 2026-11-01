"""v1.1: refit the served model as of a date, using only outcomes known by then.

Run with `make retrain`, monthly in practice: `make retrain ARGS="--as-of 2018-07-01"`.
The recipe averages logistic regression and XGBoost and uses no calibrator; why is
in reports/v1.1/walk_forward.md. The frozen v1 model is not touched.
"""

import argparse

import pandas as pd

from opspilot.config import (
    OBSERVATION_END,
    SERVED_BUNDLE_PATH,
    SERVED_HANDOVER_BUNDLE_PATH,
    SPLITS,
)
from opspilot.features.load import (
    FEATURES,
    HANDOVER_FEATURES,
    load_features,
    load_handover_features,
)
from opspilot.models.bundle import load_bundle, save_bundle
from opspilot.models.tracking import git_commit
from opspilot.models.walk_forward import cutoffs_for, fit_recipe


def build_bundle(
    rows: pd.DataFrame, params: dict, as_of: pd.Timestamp, features: list[str], stage: str
) -> dict:
    model = fit_recipe(rows, params, as_of, features)
    commit = git_commit()
    return {
        "version": f"1.1-{stage}-{as_of:%Y-%m-%d}+{commit[:7]}",
        "model": model,
        "calibrator": None,
        "calibration": "none",
        "risk_bands": cutoffs_for(model, rows, as_of, features),
        "features": list(features),
        "params": params,
        "train_period": f"{SPLITS['train'][0]} to {as_of:%Y-%m-%d}, known outcomes only",
        "git_commit": commit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", default=str(OBSERVATION_END), help="YYYY-MM-DD")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of)
    features, params = load_features(), load_bundle()["params"]
    open_orders = features.merge(load_handover_features(), on="order_id")
    for rows, names, stage, path in [
        (features, FEATURES, "approval", SERVED_BUNDLE_PATH),
        (open_orders, FEATURES + HANDOVER_FEATURES, "handover", SERVED_HANDOVER_BUNDLE_PATH),
    ]:
        bundle = build_bundle(rows, params, as_of, names, stage)
        print(f"Saved {bundle['version']} to {save_bundle(bundle, path)}")


if __name__ == "__main__":
    main()
