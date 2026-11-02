"""Treino de modelos candidatos com validação cruzada e rastreamento no MLflow."""

from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from purchase_propensity.models.factory import build_pipeline

# Nome da métrica no relatório -> scorer do Scikit-Learn
CV_SCORERS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}
MODEL_ARTIFACT_NAME = "model"

# O MLflow serializa com skops (mais seguro que pickle) e só aceita tipos auditados.
# Estes são as estruturas internas das árvores dos nossos próprios candidatos.
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


@dataclass(frozen=True)
class TrainingResult:
    """Resumo de um treino: qual modelo, em qual run e com quais métricas de CV."""

    model_name: str
    run_id: str
    cv_metrics: dict[str, float]


@dataclass
class ModelTrainer:
    """Treina candidatos e registra parâmetros, métricas e modelo de cada run no MLflow.

    Attributes:
        seed: Semente usada no modelo e nos folds.
        cv_folds: Número de folds da validação cruzada estratificada.
        tags: Tags comuns a todas as runs (ex.: versão dos dados).
    """

    seed: int
    cv_folds: int
    tags: dict[str, str] = field(default_factory=dict)

    def train(
        self,
        model_name: str,
        hyperparams: dict[str, Any],
        features: pd.DataFrame,
        target: pd.Series,
    ) -> TrainingResult:
        """Executa CV, ajusta no treino completo e loga tudo em uma run do MLflow.

        Args:
            model_name: Chave do modelo na fábrica.
            hyperparams: Hiperparâmetros do params.yaml.
            features: Features de treino.
            target: Alvo de treino.

        Returns:
            Resultado com o ``run_id`` para os próximos estágios.
        """
        pipeline = build_pipeline(model_name, hyperparams, self.seed)
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tags({**self.tags, "model_type": model_name})
            mlflow.log_params(self._run_params(model_name, hyperparams))
            cv_metrics = self._cross_validate(pipeline, features, target)
            mlflow.log_metrics(cv_metrics)
            pipeline.fit(features, target)
            self._log_model(pipeline, features)
        return TrainingResult(model_name, run.info.run_id, cv_metrics)

    def _run_params(self, model_name: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
        return {"model": model_name, "seed": self.seed, "cv_folds": self.cv_folds, **hyperparams}

    def _cross_validate(
        self, pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
    ) -> dict[str, float]:
        folds = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        scores = cross_validate(pipeline, features, target, cv=folds, scoring=CV_SCORERS)
        return {f"cv_{name}": float(scores[f"test_{name}"].mean()) for name in CV_SCORERS}

    @staticmethod
    def _log_model(pipeline: Pipeline, features: pd.DataFrame) -> None:
        example = features.head(5)
        signature = infer_signature(example, pipeline.predict_proba(example)[:, 1])
        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=signature,
            input_example=example,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
