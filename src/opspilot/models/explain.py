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
from opspilot.models.ensemble import AveragedModel


def source_feature(column: str, features: list[str]) -> str:
    """The input feature an encoded column came from."""
    if column in features:
        return column
    return max((f for f in CATEGORICAL_FEATURES if column.startswith(f"{f}_")), key=len)


def _by_feature(values: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    groups = [source_feature(column, features) for column in values.columns]
    return values.T.groupby(groups).sum().T[features]


def _tree_contributions(pipeline, features: list[str], rows: pd.DataFrame) -> pd.DataFrame:
    preprocessor, model = pipeline[0], pipeline[-1]
    encoded = preprocessor.transform(rows[features])
    values = model.get_booster().predict(xgb.DMatrix(encoded), pred_contribs=True)
    per_column = pd.DataFrame(values[:, :-1], columns=encoded.columns, index=rows.index)
    return _by_feature(per_column, features).assign(base=values[:, -1])


def _linear_contributions(
    pipeline, reference: np.ndarray, features: list[str], rows: pd.DataFrame
) -> pd.DataFrame:
    """Coefficient times distance from the fitted rows' mean: SHAP for a linear model."""
    preprocessor, model = pipeline[0], pipeline[-1]
    encoded = preprocessor.transform(rows[features])
    per_column = (encoded - reference) * model.coef_[0]
    base = float(model.intercept_[0] + (reference * model.coef_[0]).sum())
    return _by_feature(per_column, features).assign(base=base)


def contributions(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Log-odds contribution of each input feature for each row, plus the model's base value."""
    model, features = bundle["model"], bundle["features"]
    if isinstance(model, AveragedModel):
        tree = _tree_contributions(model.xgboost, features, rows)
        linear = _linear_contributions(model.logistic, model.reference, features, rows)
        return (tree + linear) / 2
    return _tree_contributions(model, features, rows)


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
