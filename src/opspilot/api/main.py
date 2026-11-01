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
from opspilot.api.service import OrderNotFound, Predictor, StageUnavailable
from opspilot.config import (
    DUCKDB_PATH,
    PREDICTION_LOG_PATH,
    SERVED_BUNDLE_PATH,
    SERVED_HANDOVER_BUNDLE_PATH,
)


def create_app(
    bundle_path: Path = SERVED_BUNDLE_PATH,
    db_path: Path = DUCKDB_PATH,
    log_path: Path = PREDICTION_LOG_PATH,
    handover_bundle_path: Path = SERVED_HANDOVER_BUNDLE_PATH,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = Predictor(bundle_path, db_path, log_path, handover_bundle_path)
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
        except (OrderNotFound, StageUnavailable) as missing:
            raise _not_found(missing) from missing

    @app.post("/explain")
    def explain(body: PredictRequest, request: Request) -> ExplainResponse:
        try:
            return predictor(request).explain(body)
        except (OrderNotFound, StageUnavailable) as missing:
            raise _not_found(missing) from missing

    return app


def _not_found(missing: LookupError) -> HTTPException:
    if isinstance(missing, StageUnavailable):
        return HTTPException(status_code=503, detail="The handover model is not loaded")
    detail = (
        f"Order {missing} is not scorable at this stage: only labeled, uncensored orders, "
        "and for handover only orders still open when they were handed over"
    )
    return HTTPException(status_code=404, detail=detail)


app = create_app()
