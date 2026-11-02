from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from purchase_propensity.config import RegistryParams
from purchase_propensity.data.preparation import clean_dataset
from purchase_propensity.data.schema import TARGET
from purchase_propensity.io import load_json, save_json
from purchase_propensity.models.registry import (
    Candidate,
    ChampionRegistry,
    select_best_candidate,
)
from purchase_propensity.models.tracking import read_data_version
from purchase_propensity.models.trainer import ModelTrainer


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """MLflow isolado por teste: banco SQLite e artifacts em pasta temporária."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test")
    yield uri
    mlflow.set_tracking_uri(None)


@pytest.fixture
def train_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_dataset(raw_df)
    return df.drop(columns=[TARGET]), df[TARGET]


def test_trainer_logs_params_metrics_and_model(tracking_uri: str, train_data) -> None:
    features, target = train_data
    trainer = ModelTrainer(seed=42, cv_folds=3, tags={"data_version": "abc"})
    result = trainer.train("random_forest", {"n_estimators": 10}, features, target)

    run = MlflowClient().get_run(result.run_id)
    assert run.data.params["model"] == "random_forest"
    assert run.data.params["n_estimators"] == "10"
    assert run.data.tags["data_version"] == "abc"
    assert {"cv_roc_auc", "cv_pr_auc", "cv_f1"} <= set(run.data.metrics)
    model = mlflow.sklearn.load_model(f"runs:/{result.run_id}/model")
    assert model.predict_proba(features).shape == (len(features), 2)


def test_select_best_candidate_uses_metric() -> None:
    results = {
        "a": {"run_id": "1", "metrics": {"test_pr_auc": 0.6}},
        "b": {"run_id": "2", "metrics": {"test_pr_auc": 0.8}},
    }
    assert select_best_candidate(results, "test_pr_auc") == Candidate("b", "2", 0.8)


def test_select_best_candidate_requires_results() -> None:
    with pytest.raises(ValueError, match="Nenhum"):
        select_best_candidate({}, "test_pr_auc")


def test_champion_registry_promotes_and_exports(
    tracking_uri: str, train_data, tmp_path: Path
) -> None:
    features, target = train_data
    result = ModelTrainer(seed=42, cv_folds=2).train("logistic_regression", {}, features, target)
    params = RegistryParams(
        selection_metric="test_pr_auc",
        model_name="test-model",
        alias="champion",
        export_dir=tmp_path / "champion",
        report_path=tmp_path / "champion.json",
    )
    registry = ChampionRegistry(params)
    version = registry.promote(Candidate("logistic_regression", result.run_id, 0.7))
    exported = registry.export()

    aliased = MlflowClient().get_model_version_by_alias("test-model", "champion")
    assert aliased.version == version.version
    assert aliased.tags["model_type"] == "logistic_regression"
    assert (exported / "MLmodel").exists()


def test_read_data_version(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    (tmp_path / "data.csv.dvc").write_text("outs:\n- md5: abc123\n  path: data.csv\n")
    assert read_data_version(dataset) == "abc123"
    assert read_data_version(tmp_path / "missing.csv") == "unknown"


def test_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"
    save_json({"modelo": 1.0}, path)
    assert load_json(path) == {"modelo": 1.0}
