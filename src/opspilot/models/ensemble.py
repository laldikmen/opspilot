"""The v1.1 model: logistic regression and XGBoost, averaged in log-odds."""

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.pipeline import Pipeline


class AveragedModel:
    def __init__(self, logistic: Pipeline, xgboost: Pipeline, reference: pd.Series):
        self.logistic = logistic
        self.xgboost = xgboost
        # Mean of the encoded fit rows; linear explanations are relative to it.
        self.reference = reference

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        scores = [m.predict_proba(rows)[:, 1] for m in (self.logistic, self.xgboost)]
        p = expit(np.mean([logit(np.clip(s, 1e-6, 1 - 1e-6)) for s in scores], axis=0))
        return np.column_stack([1 - p, p])
