from pathlib import Path

import numpy as np

from purchase_propensity.models.plots import (
    MODEL_COLORS,
    color_of,
    label_of,
    save_comparison_plots,
    save_model_plots,
)

Y_TRUE = np.array([0, 1, 0, 1, 1, 0, 0, 1])
Y_PROBA = np.array([0.2, 0.7, 0.4, 0.9, 0.3, 0.1, 0.6, 0.8])


def _is_png(path: Path) -> bool:
    return path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_model_plots_writes_three_pngs(tmp_path: Path) -> None:
    paths = save_model_plots("random_forest", Y_TRUE, Y_PROBA, 0.5, tmp_path / "rf")
    assert [path.name for path in paths] == [
        "confusion_matrix.png",
        "roc_curve.png",
        "pr_curve.png",
    ]
    assert all(_is_png(path) for path in paths)


def test_save_comparison_plots_writes_three_pngs(tmp_path: Path) -> None:
    probas = {"logistic_regression": Y_PROBA, "random_forest": Y_PROBA[::-1].copy()}
    metric_names = ["test_roc_auc", "test_pr_auc", "test_f1", "test_precision", "test_recall"]
    metrics = {name: dict.fromkeys(metric_names, 0.5) for name in probas}
    paths = save_comparison_plots(Y_TRUE, probas, metrics, tmp_path)
    assert all(_is_png(path) for path in paths)


def test_each_model_has_a_distinct_fixed_color() -> None:
    assert len(set(MODEL_COLORS.values())) == len(MODEL_COLORS)
    assert color_of("random_forest") == MODEL_COLORS["random_forest"]


def test_unknown_model_falls_back_to_its_key() -> None:
    assert label_of("svm") == "svm"
    assert color_of("svm").startswith("#")
