"""Fábrica dos modelos candidatos e do Pipeline completo (pré-processamento + estimador)."""

from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from purchase_propensity.features.preprocessing import build_preprocessor

EstimatorBuilder = Callable[[dict[str, Any], int], ClassifierMixin]


def _logistic_regression(hyperparams: dict[str, Any], seed: int) -> ClassifierMixin:
    return LogisticRegression(class_weight="balanced", random_state=seed, **hyperparams)


def _random_forest(hyperparams: dict[str, Any], seed: int) -> ClassifierMixin:
    return RandomForestClassifier(
        class_weight="balanced", random_state=seed, n_jobs=-1, **hyperparams
    )


def _hist_gradient_boosting(hyperparams: dict[str, Any], seed: int) -> ClassifierMixin:
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **hyperparams)


# class_weight="balanced" compensa o desbalanceamento (~15% de compras).
ESTIMATOR_BUILDERS: dict[str, EstimatorBuilder] = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "hist_gradient_boosting": _hist_gradient_boosting,
}


def build_pipeline(model_name: str, hyperparams: dict[str, Any], seed: int) -> Pipeline:
    """Monta o Pipeline do Scikit-Learn para um modelo candidato.

    Args:
        model_name: Chave em ``ESTIMATOR_BUILDERS``.
        hyperparams: Hiperparâmetros vindos do params.yaml.
        seed: Semente para reprodutibilidade.

    Returns:
        Pipeline ainda não treinado.

    Raises:
        ValueError: Se o modelo não estiver registrado na fábrica.
    """
    if model_name not in ESTIMATOR_BUILDERS:
        raise ValueError(f"Modelo desconhecido: {model_name}. Opções: {list(ESTIMATOR_BUILDERS)}")
    estimator = ESTIMATOR_BUILDERS[model_name](hyperparams, seed)
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])
