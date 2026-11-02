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
from purchase_propensity.models.evaluator import compute_metrics, save_evaluation_plots
from purchase_propensity.models.tracking import configure_tracking
from purchase_propensity.models.trainer import MODEL_ARTIFACT_NAME

logger = logging.getLogger(__name__)


def evaluate_run(
    run_id: str, features: pd.DataFrame, target: pd.Series, params: EvaluateParams, plots_dir: Path
) -> dict[str, float]:
    """Avalia o modelo de uma run e loga métricas e gráficos de teste na mesma run.

    Args:
        run_id: Run do MLflow gerada no estágio de treino.
        features: Features de teste.
        target: Alvo de teste.
        params: Limiar de decisão e caminhos de saída.
        plots_dir: Pasta dos gráficos deste modelo.

    Returns:
        Métricas de teste com prefixo ``test_``.
    """
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}")
    y_proba = model.predict_proba(features)[:, 1]
    y_true = target.to_numpy()
    metrics = {
        f"test_{name}": value
        for name, value in compute_metrics(y_true, y_proba, params.threshold).items()
    }
    plots = save_evaluation_plots(y_true, y_proba, params.threshold, plots_dir)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        for plot in plots:
            mlflow.log_artifact(str(plot), artifact_path="evaluation")
    return metrics


def main() -> None:
    """Avalia todos os candidatos e grava ``reports/metrics.json``."""
    configure_logging()
    params = load_params()
    configure_tracking(Settings())
    features, target = load_features_and_target(params.data.test_path)
    runs: dict[str, str] = {
        name: str(run) for name, run in load_json(params.train.runs_path).items()
    }
    report: dict[str, Any] = {}
    for model_name, run_id in runs.items():
        plots_dir = params.evaluate.plots_dir / model_name
        metrics = evaluate_run(run_id, features, target, params.evaluate, plots_dir)
        report[model_name] = {"run_id": run_id, "metrics": metrics}
        logger.info("%s | %s", model_name, metrics)
    save_json(report, params.evaluate.metrics_path)


if __name__ == "__main__":
    main()
