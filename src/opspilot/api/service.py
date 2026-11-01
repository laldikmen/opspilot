"""Load the model bundle, score requests, format the top reasons."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from opspilot.api.prediction_log import PredictionLog
from opspilot.api.schemas import ExplainResponse, PredictRequest, PredictResponse, Reason
from opspilot.features.load import CATEGORICAL_FEATURES, HANDOVER_MART, MART
from opspilot.models.bundle import load_bundle, predict_proba
from opspilot.models.explain import contributions, top_reasons

# Fallback cut-offs for a bundle without its own: the v1 model's top 10% and next
# 20% of scores on validation. The v1.1 bundle carries cut-offs from recent scores.
HIGH_RISK = 0.30
MEDIUM_RISK = 0.17


class OrderNotFound(LookupError):
    pass


class StageUnavailable(LookupError):
    pass


def risk_band(probability: float, high: float = HIGH_RISK, medium: float = MEDIUM_RISK) -> str:
    if probability >= high:
        return "high"
    return "medium" if probability >= medium else "low"


def _normalized(rows: pd.DataFrame) -> pd.DataFrame:
    """The column types the model was fitted on, whichever way the row arrived."""
    rows = rows.copy()
    numeric = [c for c in rows.columns if c not in CATEGORICAL_FEATURES]
    rows[numeric] = rows[numeric].astype(float)
    for column in CATEGORICAL_FEATURES:
        rows[column] = rows[column].astype(object).where(rows[column].notna(), np.nan)
    return rows


class Predictor:
    def __init__(
        self,
        bundle_path: Path,
        db_path: Path,
        log_path: Path,
        handover_bundle_path: Path | None = None,
    ):
        self.bundle = load_bundle(bundle_path)
        self.version = self.bundle["version"]
        self.bundles = {"approval": self.bundle}
        if handover_bundle_path is not None and Path(handover_bundle_path).exists():
            self.bundles["handover"] = load_bundle(handover_bundle_path)
        self.db_path = db_path
        self.log = PredictionLog(log_path)

    def _bundle_for(self, request: PredictRequest) -> dict:
        if request.stage not in self.bundles:
            raise StageUnavailable(request.stage)
        return self.bundles[request.stage]

    def _row_for_order(self, bundle: dict, request: PredictRequest) -> pd.DataFrame:
        columns = ", ".join(bundle["features"])
        source = MART
        if request.stage == "handover":
            source = f"{MART} inner join {HANDOVER_MART} using (order_id)"
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            row = con.execute(
                f"select {columns} from {source} where order_id = ?", [request.order_id]
            ).df()
        finally:
            con.close()
        if row.empty:
            raise OrderNotFound(request.order_id)
        return row

    def _row_for(self, bundle: dict, request: PredictRequest) -> pd.DataFrame:
        if request.order_id is not None:
            row = self._row_for_order(bundle, request)
        else:
            row = pd.DataFrame([request.features.model_dump()])
        return _normalized(row)

    def _score(self, bundle, row, request: PredictRequest, endpoint: str) -> float:
        probability = float(predict_proba(bundle, row)[0])
        values = row[bundle["features"]].iloc[0].astype(object)
        values = values.where(values.notna(), None).to_dict()
        name = f"{endpoint}:{request.stage}"
        self.log.append(name, request.order_id, values, bundle["version"], probability)
        return probability

    def predict(self, request: PredictRequest) -> PredictResponse:
        bundle = self._bundle_for(request)
        row = self._row_for(bundle, request)
        probability = self._score(bundle, row, request, "predict")
        cutoffs = bundle.get("risk_bands", {"high": HIGH_RISK, "medium": MEDIUM_RISK})
        return PredictResponse(
            order_id=request.order_id,
            stage=request.stage,
            probability=probability,
            risk_band=risk_band(probability, **cutoffs),
            model_version=bundle["version"],
            reasons=top_reasons(bundle, row)[0],
        )

    def explain(self, request: PredictRequest) -> ExplainResponse:
        bundle = self._bundle_for(request)
        row = self._row_for(bundle, request)
        probability = self._score(bundle, row, request, "explain")
        base = float(contributions(bundle, row)["base"].iloc[0])
        reasons = top_reasons(bundle, row, k=len(bundle["features"]))[0]
        return ExplainResponse(
            order_id=request.order_id,
            stage=request.stage,
            probability=probability,
            model_version=bundle["version"],
            base_value=base,
            contributions=[Reason(**reason) for reason in reasons],
        )
