"""SHAP global summary and per-order top reasons.

Contributions come from XGBoost's own TreeSHAP (`pred_contribs`), in log-odds of
the uncalibrated model; for each order they add up to its raw log-odds. One-hot
columns are summed back into the feature they came from, so a reason reads
`customer_state = RJ` rather than `customer_state_RJ`.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from opspilot.features.load import CATEGORICAL_FEATURES


def source_feature(column: str, features: list[str]) -> str:
    """The input feature an encoded column came from."""
    if column in features:
        return column
    return max((f for f in CATEGORICAL_FEATURES if column.startswith(f"{f}_")), key=len)


def contributions(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Log-odds contribution of each input feature for each row, plus the model's base value."""
    preprocessor, model = bundle["model"][0], bundle["model"][-1]
    features = bundle["features"]
    encoded = preprocessor.transform(rows[features])
    values = model.get_booster().predict(xgb.DMatrix(encoded), pred_contribs=True)
    per_column = pd.DataFrame(values[:, :-1], columns=encoded.columns, index=rows.index)
    groups = [source_feature(column, features) for column in encoded.columns]
    by_feature = per_column.T.groupby(groups).sum().T[features]
    return by_feature.assign(base=values[:, -1])


def _plain(value):
    if isinstance(value, float | np.floating) and np.isnan(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def top_reasons(bundle: dict, rows: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """The k features that moved each order's risk the most, either way.

    A positive contribution pushed the order toward late, a negative one toward
    on time.
    """
    effects = contributions(bundle, rows).drop(columns="base")
    reasons = []
    for index, row in effects.iterrows():
        strongest = row.abs().nlargest(k).index
        reasons.append(
            [
                {
                    "feature": feature,
                    "value": _plain(rows.at[index, feature]),
                    "contribution": round(float(row[feature]), 4),
                }
                for feature in strongest
            ]
        )
    return reasons


def global_importance(bundle: dict, rows: pd.DataFrame) -> pd.Series:
    """Mean absolute contribution per feature, largest first."""
    effects = contributions(bundle, rows).drop(columns="base")
    return effects.abs().mean().sort_values(ascending=False)
