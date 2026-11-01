"""FastAPI app: /health, /predict, /explain, /model-info."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from opspilot.api.schemas import (
    ExplainResponse,
    Health,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)
from opspilot.api.service import OrderNotFound, Predictor
from opspilot.config import BUNDLE_PATH, DUCKDB_PATH, PREDICTION_LOG_PATH


def create_app(
    bundle_path: Path = BUNDLE_PATH,
    db_path: Path = DUCKDB_PATH,
    log_path: Path = PREDICTION_LOG_PATH,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = Predictor(bundle_path, db_path, log_path)
        yield

    app = FastAPI(
        title="OpsPilot AI", summary="Late-delivery risk for Olist orders", lifespan=lifespan
    )

    def predictor(request: Request) -> Predictor:
        return request.app.state.predictor

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", model_version=predictor(request).version)

    @app.get("/model-info")
    def model_info(request: Request) -> ModelInfo:
        bundle = predictor(request).bundle
        return ModelInfo(
            model_version=bundle["version"],
            calibration=bundle["calibration"],
            train_period=bundle["train_period"],
            git_commit=bundle["git_commit"],
            params=bundle["params"],
            features=bundle["features"],
        )

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        try:
            return predictor(request).predict(body)
        except OrderNotFound as missing:
            raise _not_found(missing) from missing

    @app.post("/explain")
    def explain(body: PredictRequest, request: Request) -> ExplainResponse:
        try:
            return predictor(request).explain(body)
        except OrderNotFound as missing:
            raise _not_found(missing) from missing

    return app


def _not_found(missing: OrderNotFound) -> HTTPException:
    detail = f"Order {missing} is not in the feature table (labeled, uncensored orders only)"
    return HTTPException(status_code=404, detail=detail)


app = create_app()
