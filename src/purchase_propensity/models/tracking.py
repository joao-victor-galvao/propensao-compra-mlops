"""Configuração do MLflow e metadados de linhagem (versão dos dados e do código)."""

from pathlib import Path

import mlflow
import yaml

from purchase_propensity.config import Settings


def configure_tracking(settings: Settings) -> None:
    """Aponta o cliente MLflow para o servidor e o experimento configurados."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def read_data_version(dataset_path: Path) -> str:
    """Retorna o hash MD5 registrado pelo DVC para o dataset.

    Gravado como tag em cada run, liga o experimento à versão exata dos dados.

    Args:
        dataset_path: Caminho do arquivo rastreado (o ``.dvc`` fica ao lado).

    Returns:
        Hash MD5 ou ``"unknown"`` se o arquivo ``.dvc`` não existir.
    """
    dvc_file = dataset_path.with_name(dataset_path.name + ".dvc")
    if not dvc_file.exists():
        return "unknown"
    content = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    return str(content["outs"][0]["md5"])
