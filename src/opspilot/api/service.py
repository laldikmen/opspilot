"""Load the model bundle, score requests, format the top reasons."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from opspilot.api.prediction_log import PredictionLog
from opspilot.api.schemas import ExplainResponse, PredictRequest, PredictResponse, Reason
from opspilot.features.load import CATEGORICAL_FEATURES, MART, NUMERIC_FEATURES
from opspilot.models.bundle import load_bundle, predict_proba
from opspilot.models.explain import contributions, top_reasons

# The top 10% and the next 20% of final-model scores on validation, where they ran
# 37% and 19% late. Fixed there, not on test. Calibration drifts between periods
# (docs/decisions/0006), so the share of orders in each band moves with it.
HIGH_RISK = 0.30
MEDIUM_RISK = 0.17


class OrderNotFound(LookupError):
    pass


def risk_band(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "high"
    return "medium" if probability >= MEDIUM_RISK else "low"


def _normalized(rows: pd.DataFrame) -> pd.DataFrame:
    """The column types the model was fitted on, whichever way the row arrived."""
    rows = rows.copy()
    rows[NUMERIC_FEATURES] = rows[NUMERIC_FEATURES].astype(float)
    for column in CATEGORICAL_FEATURES:
        rows[column] = rows[column].astype(object).where(rows[column].notna(), np.nan)
    return rows


class Predictor:
    def __init__(self, bundle_path: Path, db_path: Path, log_path: Path):
        self.bundle = load_bundle(bundle_path)
        self.version = self.bundle["version"]
        self.db_path = db_path
        self.log = PredictionLog(log_path)

    def _row_for_order(self, order_id: str) -> pd.DataFrame:
        columns = ", ".join(self.bundle["features"])
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            row = con.execute(f"select {columns} from {MART} where order_id = ?", [order_id]).df()
        finally:
            con.close()
        if row.empty:
            raise OrderNotFound(order_id)
        return row

    def _row_for(self, request: PredictRequest) -> pd.DataFrame:
        if request.order_id is not None:
            row = self._row_for_order(request.order_id)
        else:
            row = pd.DataFrame([request.features.model_dump()])
        return _normalized(row)

    def _score(self, row: pd.DataFrame, request: PredictRequest, endpoint: str) -> float:
        probability = float(predict_proba(self.bundle, row)[0])
        values = row[self.bundle["features"]].iloc[0].astype(object)
        values = values.where(values.notna(), None).to_dict()
        self.log.append(endpoint, request.order_id, values, self.version, probability)
        return probability

    def predict(self, request: PredictRequest) -> PredictResponse:
        row = self._row_for(request)
        probability = self._score(row, request, "predict")
        return PredictResponse(
            order_id=request.order_id,
            probability=probability,
            risk_band=risk_band(probability),
            model_version=self.version,
            reasons=top_reasons(self.bundle, row)[0],
        )

    def explain(self, request: PredictRequest) -> ExplainResponse:
        row = self._row_for(request)
        probability = self._score(row, request, "explain")
        base = float(contributions(self.bundle, row)["base"].iloc[0])
        reasons = top_reasons(self.bundle, row, k=len(self.bundle["features"]))[0]
        return ExplainResponse(
            order_id=request.order_id,
            probability=probability,
            model_version=self.version,
            base_value=base,
            contributions=[Reason(**reason) for reason in reasons],
        )
