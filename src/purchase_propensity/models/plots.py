"""Gráficos de avaliação em PNG: por modelo e comparativos entre os candidatos."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (Docker/CI)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from purchase_propensity.models.evaluator import FloatArray, IntArray

MODEL_LABELS = {
    "logistic_regression": "Regressão Logística",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}
# Cor fixa por modelo (não por ranking): o mesmo modelo tem a mesma cor em todo gráfico.
MODEL_COLORS = {
    "logistic_regression": "#2a78d6",
    "random_forest": "#eb6834",
    "hist_gradient_boosting": "#1baf7a",
}
COMPARED_METRICS = {
    "test_roc_auc": "ROC-AUC",
    "test_pr_auc": "PR-AUC",
    "test_f1": "F1",
    "test_precision": "Precisão",
    "test_recall": "Recall",
}
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
BASELINE_COLOR = "#8a8984"
CLASS_LABELS = ["Não comprou", "Comprou"]
BASELINE_STYLE: dict[str, Any] = {"linestyle": "--", "color": BASELINE_COLOR, "linewidth": 1.2}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": MUTED_COLOR,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": MUTED_COLOR,
        "ytick.color": MUTED_COLOR,
        "legend.frameon": False,
    }
)


def label_of(model_name: str) -> str:
    """Nome de exibição do modelo."""
    return MODEL_LABELS.get(model_name, model_name)


def color_of(model_name: str) -> str:
    """Cor fixa do modelo nos gráficos."""
    return MODEL_COLORS.get(model_name, MUTED_COLOR)


def save_model_plots(
    model_name: str, y_true: IntArray, y_proba: FloatArray, threshold: float, output_dir: Path
) -> list[Path]:
    """Salva matriz de confusão, curva ROC e curva Precision-Recall de um modelo.

    Args:
        model_name: Chave do modelo (define título e cor).
        y_true: Rótulos reais (0/1).
        y_proba: Probabilidade prevista da classe positiva.
        threshold: Limiar usado na matriz de confusão.
        output_dir: Pasta de destino.

    Returns:
        Caminhos dos arquivos gerados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_proba >= threshold).astype(int)
    curves = {model_name: y_proba}
    figures = {
        "confusion_matrix.png": _draw(lambda ax: _confusion(ax, model_name, y_true, y_pred)),
        "roc_curve.png": _draw(lambda ax: _roc(ax, y_true, curves, label_of(model_name))),
        "pr_curve.png": _draw(lambda ax: _pr(ax, y_true, curves, label_of(model_name))),
    }
    return [_save(figure, output_dir / name) for name, figure in figures.items()]


def save_comparison_plots(
    y_true: IntArray,
    probas: dict[str, FloatArray],
    metrics: dict[str, dict[str, float]],
    output_dir: Path,
) -> list[Path]:
    """Salva os gráficos que comparam todos os candidatos no conjunto de teste.

    Args:
        y_true: Rótulos reais (0/1).
        probas: ``modelo -> probabilidades previstas``.
        metrics: ``modelo -> métricas de teste``.
        output_dir: Pasta de destino.

    Returns:
        Caminhos dos arquivos gerados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "comparison_metrics.png": _draw(lambda ax: _metric_bars(ax, metrics), size=(9, 5)),
        "comparison_roc_curves.png": _draw(lambda ax: _roc(ax, y_true, probas, "Comparação")),
        "comparison_pr_curves.png": _draw(lambda ax: _pr(ax, y_true, probas, "Comparação")),
    }
    return [_save(figure, output_dir / name) for name, figure in figures.items()]


def _draw(plot: Callable[[Axes], None], size: tuple[float, float] = (6.4, 5.2)) -> Figure:
    figure, ax = plt.subplots(figsize=size)
    plot(ax)
    figure.tight_layout()
    return figure


def _save(figure: Figure, path: Path) -> Path:
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    return path


def _confusion(ax: Axes, model_name: str, y_true: IntArray, y_pred: IntArray) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    shares = matrix / matrix.sum(axis=1, keepdims=True)  # cor pelo % da linha, não pelo volume
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for (row, col), count in np.ndenumerate(matrix):
        _annotate_cell(ax, row, col, int(count), float(shares[row, col]))
    ax.set_xticks([0, 1], labels=CLASS_LABELS)
    ax.set_yticks([0, 1], labels=CLASS_LABELS)
    ax.set_xlabel("Previsto pelo modelo")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão — {label_of(model_name)}")


def _annotate_cell(ax: Axes, row: int, col: int, count: int, share: float) -> None:
    dark_cell = share > 0.5  # noqa: PLR2004 - metade da escala de cor
    ax.text(
        col,
        row,
        f"{count}\n({share:.0%} da linha)",
        ha="center",
        va="center",
        color="white" if dark_cell else TEXT_COLOR,
        fontsize=12,
    )


def _roc(ax: Axes, y_true: IntArray, probas: dict[str, FloatArray], title: str) -> None:
    for model_name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        _plot_model_curve(ax, model_name, fpr, tpr, f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], label="Aleatório (AUC = 0.500)", **BASELINE_STYLE)
    _finish_curve_axes(
        ax, f"Curva ROC — {title}", "Taxa de falsos positivos", "Taxa de verdadeiros positivos"
    )


def _pr(ax: Axes, y_true: IntArray, probas: dict[str, FloatArray], title: str) -> None:
    for model_name, y_proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        score = f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}"
        _plot_model_curve(ax, model_name, recall, precision, score)
    prevalence = float(np.mean(y_true))  # precisão de quem chuta "compra" para todos
    ax.axhline(prevalence, label=f"Aleatório (PR-AUC = {prevalence:.3f})", **BASELINE_STYLE)
    _finish_curve_axes(ax, f"Curva Precision-Recall — {title}", "Recall", "Precisão")


def _plot_model_curve(ax: Axes, model_name: str, x: FloatArray, y: FloatArray, score: str) -> None:
    label = f"{label_of(model_name)} ({score})"
    ax.plot(x, y, color=color_of(model_name), linewidth=2, label=label)


def _finish_curve_axes(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right" if "ROC" in title else "lower left", fontsize=9)


def _metric_bars(ax: Axes, metrics: dict[str, dict[str, float]]) -> None:
    positions = np.arange(len(COMPARED_METRICS))
    width = 0.8 / len(metrics)
    for index, (model_name, values) in enumerate(metrics.items()):
        heights = [values[key] for key in COMPARED_METRICS]
        bars = ax.bar(
            positions + index * width,
            heights,
            width * 0.92,
            color=color_of(model_name),
            label=label_of(model_name),
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=MUTED_COLOR, padding=2)
    ax.set_xticks(positions + width * (len(metrics) - 1) / 2, labels=COMPARED_METRICS.values())
    ax.set(ylim=(0, 1.08), ylabel="Valor no conjunto de teste")
    ax.set_title("Comparação dos modelos no conjunto de teste")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(metrics))
