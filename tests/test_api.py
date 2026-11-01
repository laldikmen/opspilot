"""Schema validation, response shape, deterministic score."""

import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opspilot.api.main import create_app
from opspilot.api.prediction_log import input_hash
from opspilot.api.schemas import OrderFeatures
from opspilot.api.service import HIGH_RISK, MEDIUM_RISK, risk_band
from opspilot.features.load import FEATURES, TARGET
from opspilot.models.bundle import build_bundle, save_bundle
from opspilot.models.calibrate import SigmoidCalibrator
from opspilot.models.split import assign_split
from opspilot.models.train import DEFAULT_PARAMS, fit_xgboost


@pytest.fixture(scope="module")
def log_path(tmp_path_factory):
    return tmp_path_factory.mktemp("logs") / "predictions.sqlite3"


@pytest.fixture(scope="module")
def client(features, warehouse_path, tmp_path_factory, log_path):
    train = features[assign_split(features.purchase_date) == "train"]
    model = fit_xgboost(train, DEFAULT_PARAMS)
    raw = model.predict_proba(train[FEATURES])[:, 1]
    calibrator = SigmoidCalibrator().fit(raw, train[TARGET])
    bundle = build_bundle(model, calibrator, "pooled_sigmoid", DEFAULT_PARAMS, "0" * 40)
    path = save_bundle(bundle, tmp_path_factory.mktemp("bundle") / "bundle.joblib")
    with TestClient(create_app(path, warehouse_path, log_path)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def order(features):
    return features.iloc[len(features) // 2]


def payload(row) -> dict:
    values = row[FEATURES].to_dict()
    clean = {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in values.items()}
    for key in [
        "purchase_dow",
        "purchase_hour",
        "n_items",
        "n_sellers",
        "n_products",
        "same_state",
    ]:
        clean[key] = int(clean[key])
    for key in ["installments", "n_payment_methods"]:
        clean[key] = None if clean[key] is None else int(clean[key])
    return clean


def test_request_schema_covers_exactly_the_model_features():
    assert set(OrderFeatures.model_fields) == set(FEATURES)


def test_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["features"] == FEATURES
    assert info["calibration"] == "pooled_sigmoid"


def test_predict_by_order_id_has_the_agreed_shape(client, order):
    body = client.post("/predict", json={"order_id": order.order_id}).json()
    assert set(body) == {"order_id", "probability", "risk_band", "model_version", "reasons"}
    assert 0 < body["probability"] < 1
    assert body["risk_band"] == risk_band(body["probability"])
    assert len(body["reasons"]) == 3
    assert all(reason["feature"] in FEATURES for reason in body["reasons"])


def test_score_is_deterministic(client, order):
    first = client.post("/predict", json={"order_id": order.order_id}).json()
    second = client.post("/predict", json={"order_id": order.order_id}).json()
    assert first == second


def test_raw_features_score_the_same_as_the_order_id(client, order):
    by_id = client.post("/predict", json={"order_id": order.order_id}).json()
    by_values = client.post("/predict", json={"features": payload(order)}).json()
    assert by_values["probability"] == pytest.approx(by_id["probability"])
    assert by_values["order_id"] is None


def test_orders_with_missing_values_are_scored(client, features):
    rows = features[features.main_category.isna() | features.distance_km.isna()]
    assert len(rows) > 0
    row = rows.iloc[0]
    by_id = client.post("/predict", json={"order_id": row.order_id})
    by_values = client.post("/predict", json={"features": payload(row)})
    assert by_id.status_code == by_values.status_code == 200
    assert by_values.json()["probability"] == pytest.approx(by_id.json()["probability"])


def test_explain_lists_every_feature_largest_effect_first(client, order):
    body = client.post("/explain", json={"order_id": order.order_id}).json()
    assert len(body["contributions"]) == len(FEATURES)
    effects = [abs(reason["contribution"]) for reason in body["contributions"]]
    assert effects == sorted(effects, reverse=True)
    assert np.isfinite(body["base_value"])


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"order_id": "x", "features": {}},
        {"order": "x"},
        {"features": {"promised_days": 10}},
    ],
)
def test_malformed_requests_are_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_out_of_range_values_are_rejected(client, order):
    values = payload(order) | {"purchase_hour": 25}
    assert client.post("/predict", json={"features": values}).status_code == 422


def test_unknown_order_is_a_404(client):
    assert client.post("/predict", json={"order_id": "nope"}).status_code == 404


def test_the_label_never_reaches_the_response(client, order):
    assert "is_late" not in client.post("/predict", json={"order_id": order.order_id}).text


def test_every_prediction_is_logged(client, order, log_path):
    def count():
        with sqlite3.connect(log_path) as con:
            return con.execute("select count(*) from predictions").fetchone()[0]

    before = count()
    body = client.post("/predict", json={"order_id": order.order_id}).json()
    client.post("/explain", json={"order_id": order.order_id})
    assert count() == before + 2
    with sqlite3.connect(log_path) as con:
        logged = con.execute(
            "select endpoint, order_id, model_version, probability, input_hash "
            "from predictions order by rowid desc limit 2"
        ).fetchall()
    assert logged[1][:3] == ("predict", order.order_id, body["model_version"])
    assert logged[1][3] == pytest.approx(body["probability"])
    assert logged[0][4] == logged[1][4]


def test_rejected_requests_are_not_logged(client, log_path):
    with sqlite3.connect(log_path) as con:
        before = con.execute("select count(*) from predictions").fetchone()[0]
    client.post("/predict", json={"order_id": "nope"})
    with sqlite3.connect(log_path) as con:
        assert con.execute("select count(*) from predictions").fetchone()[0] == before


def test_input_hash_ignores_key_order():
    assert input_hash({"a": 1, "b": None}) == input_hash({"b": None, "a": 1})
    assert input_hash({"a": 1}) != input_hash({"a": 2})


def test_risk_bands():
    assert risk_band(HIGH_RISK) == "high"
    assert risk_band(MEDIUM_RISK) == "medium"
    assert risk_band(MEDIUM_RISK - 0.001) == "low"
