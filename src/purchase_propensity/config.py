"""Configuração do projeto: parâmetros do pipeline (params.yaml) e variáveis de ambiente (.env)."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PARAMS_PATH = Path("params.yaml")


class Settings(BaseSettings):
    """Configuração dependente do ambiente, lida do .env ou de variáveis de ambiente."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "purchase-propensity"
    model_path: Path = Path("models/champion")
    decision_threshold: float = Field(default=0.5, gt=0, lt=1)


class DataParams(BaseModel):
    """Caminhos dos dados e proporção do split."""

    raw_path: Path
    train_path: Path
    test_path: Path
    test_size: float = Field(gt=0, lt=1)


class TrainParams(BaseModel):
    """Candidatos a modelo e validação cruzada."""

    cv_folds: int = Field(ge=2)
    runs_path: Path
    models: dict[str, dict[str, Any]]


class EvaluateParams(BaseModel):
    """Limiar de decisão e saídas da avaliação."""

    threshold: float = Field(gt=0, lt=1)
    metrics_path: Path
    plots_dir: Path


class RegistryParams(BaseModel):
    """Critério de seleção e destino do modelo campeão."""

    selection_metric: str
    model_name: str
    alias: str
    export_dir: Path
    report_path: Path


class Params(BaseModel):
    """Todos os parâmetros do params.yaml, validados."""

    seed: int
    data: DataParams
    train: TrainParams
    evaluate: EvaluateParams
    registry: RegistryParams


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> Params:
    """Carrega e valida o params.yaml.

    Args:
        path: Caminho do arquivo de parâmetros.

    Returns:
        Parâmetros tipados.
    """
    with path.open(encoding="utf-8") as file:
        return Params.model_validate(yaml.safe_load(file))
