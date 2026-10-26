"""Encoders and imputers, fitted on train only (docs/decisions/0005, rule 3).

Every model wraps this in a pipeline, so `fit` only ever sees the rows the
model is fitted on.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from opspilot.features.load import CATEGORICAL_FEATURES, FEATURES

# Long right tails; logged before scaling for linear models.
SKEWED_FEATURES = [
    "approval_delay_hours",
    "total_price",
    "total_freight",
    "total_weight_g",
    "total_volume_cm3",
    "distance_km",
]

# The most frequent values of each category are kept and the rest become
# "infrequent" (the feature list's "top 20 categories, rest grouped as other").
TOP_CATEGORIES = 20


def build_preprocessor(scale: bool, features: list[str] = FEATURES) -> ColumnTransformer:
    """Median imputation for numbers, one-hot encoding of the top categories.

    scale=True also logs the long-tailed features and standardizes every number,
    for linear models. Trees get the numbers as they are. `features` defaults to
    the agreed feature list; experiments can pass a different one.
    """
    numeric_features = [c for c in features if c not in CATEGORICAL_FEATURES]
    categorical_features = [c for c in features if c in CATEGORICAL_FEATURES]
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(
            max_categories=TOP_CATEGORIES + 1,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        ),
    )
    if scale:
        skewed = [c for c in numeric_features if c in SKEWED_FEATURES]
        plain = [c for c in numeric_features if c not in SKEWED_FEATURES]
        numeric = [
            (
                "skewed",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    StandardScaler(),
                ),
                skewed,
            ),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), plain),
        ]
    else:
        numeric = [("numeric", SimpleImputer(strategy="median"), numeric_features)]

    preprocessor = ColumnTransformer(
        [*numeric, ("categorical", categorical, categorical_features)],
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
