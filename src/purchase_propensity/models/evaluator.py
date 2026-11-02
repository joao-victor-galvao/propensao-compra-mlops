"""Métricas e gráficos de avaliação de um classificador binário."""

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (Docker/CI)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


def compute_metrics(y_true: IntArray, y_proba: FloatArray, threshold: float) -> dict[str, float]:
    """Calcula métricas de ranking (AUCs) e de decisão (no limiar informado).

    Args:
        y_true: Rótulos reais (0/1).
        y_proba: Probabilidade prevista da classe positiva.
        threshold: Limiar para converter probabilidade em classe.

    Returns:
        Dicionário ``nome -> valor``.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def save_evaluation_plots(
    y_true: IntArray, y_proba: FloatArray, threshold: float, output_dir: Path
) -> list[Path]:
    """Salva matriz de confusão, curva ROC e curva Precision-Recall em PNG.

    Args:
        y_true: Rótulos reais (0/1).
        y_proba: Probabilidade prevista da classe positiva.
        threshold: Limiar usado na matriz de confusão.
        output_dir: Pasta de destino.

    Returns:
        Caminhos dos arquivos gerados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_proba >= threshold).astype(int)
    plots: dict[str, Callable[[Axes], object]] = {
        "confusion_matrix.png": lambda ax: ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax
        ),
        "roc_curve.png": lambda ax: RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax),
        "pr_curve.png": lambda ax: PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax),
    }
    return [_save_plot(draw, output_dir / name) for name, draw in plots.items()]


def _save_plot(draw: Callable[[Axes], object], path: Path) -> Path:
    figure, ax = plt.subplots(figsize=(6, 5))
    draw(ax)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
