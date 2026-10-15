"""Encoders and imputers, fitted on train only (docs/decisions/0005, rule 3).

Every model wraps this in a pipeline, so `fit` only ever sees the rows the
model is fitted on.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from opspilot.features.load import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Long right tails; logged before scaling for linear models.
SKEWED_FEATURES = [
    "approval_delay_hours",
    "total_price",
    "total_freight",
    "total_weight_g",
    "total_volume_cm3",
    "distance_km",
    "seller_past_delivered",
]

# The most frequent values of each category are kept and the rest become
# "infrequent" (the feature list's "top 20 categories, rest grouped as other").
TOP_CATEGORIES = 20


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Median imputation for numbers, one-hot encoding of the top categories.

    scale=True also logs the long-tailed features and standardizes every number,
    for linear models. Trees get the numbers as they are.
    """
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(
            max_categories=TOP_CATEGORIES + 1,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        ),
    )
    if scale:
        plain = [c for c in NUMERIC_FEATURES if c not in SKEWED_FEATURES]
        numeric = [
            (
                "skewed",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    StandardScaler(),
                ),
                SKEWED_FEATURES,
            ),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), plain),
        ]
    else:
        numeric = [("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES)]

    preprocessor = ColumnTransformer(
        [*numeric, ("categorical", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
