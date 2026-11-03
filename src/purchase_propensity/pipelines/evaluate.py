"""Estágio `evaluate` do DVC: mede cada candidato no conjunto de teste (nunca visto no treino)."""

import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from purchase_propensity.config import EvaluateParams, Settings, load_params
from purchase_propensity.data.loader import load_features_and_target
from purchase_propensity.io import load_json, save_json
from purchase_propensity.logging_config import configure_logging
from purchase_propensity.models.evaluator import FloatArray, compute_metrics
from purchase_propensity.models.plots import save_comparison_plots, save_model_plots
from purchase_propensity.models.tracking import configure_tracking
from purchase_propensity.models.trainer import MODEL_ARTIFACT_NAME

logger = logging.getLogger(__name__)


def predict_proba(run_id: str, features: pd.DataFrame) -> FloatArray:
    """Carrega o modelo da run no MLflow e prevê a probabilidade de compra."""
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}")
    probabilities: FloatArray = model.predict_proba(features)[:, 1]
    return probabilities


def evaluate_run(
    model_name: str, run_id: str, y_proba: FloatArray, target: pd.Series, params: EvaluateParams
) -> dict[str, float]:
    """Calcula métricas e gráficos de teste e os loga na mesma run do treino.

    Args:
        model_name: Chave do modelo (pasta e título dos gráficos).
        run_id: Run do MLflow gerada no estágio de treino.
        y_proba: Probabilidades previstas no conjunto de teste.
        target: Alvo de teste.
        params: Limiar de decisão e caminhos de saída.

    Returns:
        Métricas de teste com prefixo ``test_``.
    """
    y_true = target.to_numpy()
    metrics = {
        f"test_{name}": value
        for name, value in compute_metrics(y_true, y_proba, params.threshold).items()
    }
    plots_dir = params.plots_dir / model_name
    plots = save_model_plots(model_name, y_true, y_proba, params.threshold, plots_dir)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        _log_plots(plots)
    return metrics


def _log_plots(plots: list[Path]) -> None:
    for plot in plots:
        mlflow.log_artifact(str(plot), artifact_path="evaluation")


def main() -> None:
    """Avalia todos os candidatos, gera os comparativos e grava ``reports/metrics.json``."""
    configure_logging()
    params = load_params()
    configure_tracking(Settings())
    features, target = load_features_and_target(params.data.test_path)
    runs = {name: str(run_id) for name, run_id in load_json(params.train.runs_path).items()}
    probas = {name: predict_proba(run_id, features) for name, run_id in runs.items()}
    report: dict[str, Any] = {}
    for model_name, run_id in runs.items():
        metrics = evaluate_run(model_name, run_id, probas[model_name], target, params.evaluate)
        report[model_name] = {"run_id": run_id, "metrics": metrics}
        logger.info("%s | %s", model_name, metrics)
    test_metrics = {name: result["metrics"] for name, result in report.items()}
    save_comparison_plots(target.to_numpy(), probas, test_metrics, params.evaluate.plots_dir)
    save_json(report, params.evaluate.metrics_path)


if __name__ == "__main__":
    main()
