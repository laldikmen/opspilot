"""Pydantic request and response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrderFeatures(BaseModel):
    """The feature values of one order, as known when it was approved.

    Optional fields are ones the warehouse can also leave empty; the model
    imputes them.
    """

    model_config = ConfigDict(extra="forbid")

    promised_days: float
    promise_slack_days: float | None = None
    approval_delay_hours: float | None = Field(default=None, ge=0)
    purchase_dow: int = Field(ge=1, le=7, description="1 is Monday, 7 is Sunday")
    purchase_hour: int = Field(ge=0, le=23)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    freight_ratio: float = Field(ge=0)
    n_items: int = Field(ge=1)
    n_sellers: int = Field(ge=1)
    n_products: int = Field(ge=1)
    total_weight_g: float | None = Field(default=None, ge=0)
    total_volume_cm3: float | None = Field(default=None, ge=0)
    same_state: int = Field(ge=0, le=1)
    distance_km: float | None = Field(default=None, ge=0)
    route_mean_delivery_days: float | None = None
    route_late_rate_smoothed: float | None = Field(default=None, ge=0, le=1)
    installments: int | None = Field(default=None, ge=0)
    n_payment_methods: int | None = Field(default=None, ge=0)
    seller_late_rate_smoothed: float = Field(ge=0, le=1)
    seller_mean_handover_days: float | None = Field(default=None, ge=0)
    main_category: str | None = None
    customer_state: str = Field(min_length=2, max_length=2)
    seller_state: str = Field(min_length=2, max_length=2)
    payment_type: str | None = None


class PredictRequest(BaseModel):
    """Score an order in the warehouse by id, or any order by its feature values."""

    model_config = ConfigDict(extra="forbid")

    order_id: str | None = None
    features: OrderFeatures | None = None
    stage: Literal["approval", "handover"] = Field(
        default="approval",
        description="handover uses what is known once the seller has shipped; needs order_id",
    )

    @model_validator(mode="after")
    def exactly_one_input(self) -> "PredictRequest":
        if (self.order_id is None) == (self.features is None):
            raise ValueError("send exactly one of order_id or features")
        if self.stage == "handover" and self.order_id is None:
            raise ValueError("the handover stage needs an order_id")
        return self


class Reason(BaseModel):
    feature: str
    value: float | str | None
    contribution: float = Field(description="Log-odds; positive pushes toward late")


class PredictResponse(BaseModel):
    order_id: str | None
    stage: str
    probability: float = Field(description="Calibrated probability of arriving late")
    risk_band: str = Field(description="high, medium or low")
    model_version: str
    reasons: list[Reason]


class ExplainResponse(BaseModel):
    order_id: str | None
    stage: str
    probability: float
    model_version: str
    base_value: float = Field(description="The model's starting point, in log-odds")
    contributions: list[Reason] = Field(description="Every feature, largest effect first")


class ModelInfo(BaseModel):
    model_version: str
    calibration: str
    train_period: str
    git_commit: str
    params: dict[str, float]
    features: list[str]


class Health(BaseModel):
    status: str
    model_version: str
