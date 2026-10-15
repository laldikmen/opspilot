"""The model bundle: everything needed to score an order, saved as one file."""

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from opspilot.config import BUNDLE_PATH, SPLITS
from opspilot.features.load import FEATURES


def build_bundle(model: Pipeline, calibrator, calibration: str, params: dict, commit: str) -> dict:
    train_start, train_end = SPLITS["train"]
    return {
        "version": f"{date.today():%Y.%m.%d}+{commit[:7]}",
        "model": model,
        "calibrator": calibrator,
        "calibration": calibration,
        "features": list(FEATURES),
        "params": params,
        "train_period": f"{train_start} to {train_end}",
        "git_commit": commit,
    }


def save_bundle(bundle: dict, path: Path = BUNDLE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    return joblib.load(path)


def predict_proba(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    """Calibrated late-delivery probabilities for rows of the feature table."""
    raw = bundle["model"].predict_proba(rows[bundle["features"]])[:, 1]
    calibrator = bundle["calibrator"]
    return raw if calibrator is None else calibrator.predict(raw)
