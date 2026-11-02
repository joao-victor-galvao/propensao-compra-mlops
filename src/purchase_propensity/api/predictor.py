"""Carrega o modelo campeão exportado e transforma sessões em probabilidades de compra."""

from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from purchase_propensity.api.schemas import Prediction, Session
from purchase_propensity.data.schema import BOOLEAN_FEATURES, FEATURES

REGISTRY_METADATA_FILE = "registered_model_meta"


@dataclass(frozen=True)
class PurchasePredictor:
    """Modelo pronto para inferência, com o limiar de decisão de negócio.

    Attributes:
        pipeline: Pipeline do Scikit-Learn (pré-processamento + classificador).
        threshold: Probabilidade mínima para considerar que a sessão vai comprar.
        model_name: Nome no Model Registry.
        model_version: Versão no Model Registry.
    """

    pipeline: Pipeline
    threshold: float
    model_name: str
    model_version: str

    @classmethod
    def from_path(cls, model_path: Path, threshold: float) -> "PurchasePredictor":
        """Carrega o modelo no formato MLflow a partir de uma pasta local."""
        pipeline = mlflow.sklearn.load_model(str(model_path))
        name, version = _read_registry_metadata(model_path)
        return cls(pipeline, threshold, name, version)

    def predict(self, sessions: list[Session]) -> list[Prediction]:
        """Calcula a propensão de compra de cada sessão, na ordem recebida."""
        probabilities = self.pipeline.predict_proba(_to_frame(sessions))[:, 1]
        return [
            Prediction(
                purchase_probability=round(float(proba), 6),
                will_purchase=bool(proba >= self.threshold),
                threshold=self.threshold,
            )
            for proba in probabilities
        ]


def _to_frame(sessions: list[Session]) -> pd.DataFrame:
    frame = pd.DataFrame([session.model_dump() for session in sessions], columns=list(FEATURES))
    for column in BOOLEAN_FEATURES:
        frame[column] = frame[column].astype(int)
    return frame


def _read_registry_metadata(model_path: Path) -> tuple[str, str]:
    metadata_file = model_path / REGISTRY_METADATA_FILE
    if not metadata_file.exists():
        return "unregistered", "unknown"
    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return str(metadata["model_name"]), str(metadata["model_version"])
