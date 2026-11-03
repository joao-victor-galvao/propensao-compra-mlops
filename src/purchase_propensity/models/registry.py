"""Seleção do melhor modelo e promoção no MLflow Model Registry."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from purchase_propensity.config import RegistryParams
from purchase_propensity.models.trainer import MODEL_ARTIFACT_NAME

MODEL_DESCRIPTION = (
    "Propensão de compra de uma sessão de e-commerce (dataset Online Shoppers Purchasing "
    "Intention). Pipeline Scikit-Learn com pré-processamento embutido; retorna a probabilidade "
    "de a sessão terminar em compra. O alias 'champion' aponta para a versão em produção."
)


@dataclass(frozen=True)
class Candidate:
    """Modelo avaliado e elegível para promoção."""

    model_name: str
    run_id: str
    score: float


def select_best_candidate(results: dict[str, dict[str, Any]], metric: str) -> Candidate:
    """Escolhe o candidato com maior valor da métrica de seleção.

    Args:
        results: ``modelo -> {"run_id": ..., "metrics": {...}}`` (saída do estágio evaluate).
        metric: Nome da métrica usada no ranking (ex.: ``test_pr_auc``).

    Returns:
        O melhor candidato.

    Raises:
        ValueError: Se não houver candidatos.
    """
    if not results:
        raise ValueError("Nenhum modelo avaliado para selecionar")
    candidates = [
        Candidate(name, result["run_id"], float(result["metrics"][metric]))
        for name, result in results.items()
    ]
    return max(candidates, key=lambda candidate: candidate.score)


class ChampionRegistry:
    """Registra o modelo vencedor, aplica o alias de produção e o exporta para disco."""

    def __init__(self, params: RegistryParams, client: MlflowClient | None = None) -> None:
        """Inicializa com os parâmetros do Registry e um cliente MLflow opcional."""
        self._params = params
        self._client = client or MlflowClient()

    def promote(self, candidate: Candidate) -> ModelVersion:
        """Cria uma nova versão do modelo registrado e move o alias para ela.

        Args:
            candidate: Candidato vencedor.

        Returns:
            A versão registrada.
        """
        model_uri = f"runs:/{candidate.run_id}/{MODEL_ARTIFACT_NAME}"
        version = mlflow.register_model(model_uri, self._params.model_name)
        self._client.set_registered_model_alias(
            self._params.model_name, self._params.alias, version.version
        )
        self._tag_version(version, candidate)
        self._describe(version, candidate)
        return version

    def export(self) -> Path:
        """Baixa o modelo apontado pelo alias para ``export_dir`` (usado pela API).

        Returns:
            Pasta com o modelo no formato MLflow.
        """
        destination = self._params.export_dir
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{self._params.model_name}@{self._params.alias}",
            dst_path=str(destination),
        )
        return destination

    def _tag_version(self, version: ModelVersion, candidate: Candidate) -> None:
        tags = {
            "model_type": candidate.model_name,
            "selection_metric": self._params.selection_metric,
            "selection_score": f"{candidate.score:.4f}",
        }
        for key, value in tags.items():
            self._client.set_model_version_tag(self._params.model_name, version.version, key, value)

    def _describe(self, version: ModelVersion, candidate: Candidate) -> None:
        # Descrições aparecem na UI do Registry: deixam claro o que é o modelo e por que venceu.
        self._client.update_registered_model(self._params.model_name, description=MODEL_DESCRIPTION)
        self._client.update_model_version(
            self._params.model_name,
            version.version,
            description=(
                f"Vencedor entre os candidatos: {candidate.model_name}, com "
                f"{self._params.selection_metric} = {candidate.score:.4f} (run {candidate.run_id})."
            ),
        )
