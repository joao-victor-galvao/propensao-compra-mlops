from pathlib import Path

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from purchase_propensity.api.main import create_app
from purchase_propensity.api.schemas import SESSION_EXAMPLE
from purchase_propensity.data.preparation import clean_dataset
from purchase_propensity.data.schema import TARGET
from purchase_propensity.models.factory import build_pipeline


@pytest.fixture
def model_path(raw_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Modelo pequeno salvo no mesmo formato que o estágio register exporta."""
    df = clean_dataset(raw_df)
    pipeline = build_pipeline("logistic_regression", {}, seed=42)
    pipeline.fit(df.drop(columns=[TARGET]), df[TARGET])
    path = tmp_path / "champion"
    mlflow.sklearn.save_model(pipeline, str(path))
    (path / "registered_model_meta").write_text("model_name: test-model\nmodel_version: '7'\n")
    return path


@pytest.fixture
def client(model_path: Path) -> TestClient:
    with TestClient(create_app(model_path=model_path, threshold=0.5)) as test_client:
        yield test_client


def test_health_reports_model_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_name": "test-model", "model_version": "7"}


def test_predict_returns_probability(client: TestClient) -> None:
    response = client.post("/predict", json=SESSION_EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["purchase_probability"] <= 1
    assert body["will_purchase"] == (body["purchase_probability"] >= 0.5)


def test_predict_batch_keeps_order(client: TestClient) -> None:
    single = client.post("/predict", json=SESSION_EXAMPLE).json()
    other = {**SESSION_EXAMPLE, "PageValues": 0.0, "Month": "Feb"}
    response = client.post("/predict/batch", json={"sessions": [SESSION_EXAMPLE, other]})
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0] == single


@pytest.mark.parametrize(
    "override",
    [{"BounceRates": 1.5}, {"Month": "Janeiro"}, {"ProductRelated": -1}, {"VisitorType": "Bot"}],
)
def test_predict_rejects_invalid_session(client: TestClient, override: dict[str, object]) -> None:
    response = client.post("/predict", json={**SESSION_EXAMPLE, **override})
    assert response.status_code == 422


def test_predict_batch_rejects_empty(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"sessions": []}).status_code == 422
