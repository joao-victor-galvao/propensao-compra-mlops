import numpy as np
import pandas as pd
import pytest

from purchase_propensity.data.preparation import clean_dataset
from purchase_propensity.data.schema import TARGET
from purchase_propensity.models.evaluator import compute_metrics, save_evaluation_plots
from purchase_propensity.models.factory import ESTIMATOR_BUILDERS, build_pipeline


@pytest.mark.parametrize("model_name", list(ESTIMATOR_BUILDERS))
def test_build_pipeline_fits_and_predicts_probabilities(
    model_name: str, raw_df: pd.DataFrame
) -> None:
    df = clean_dataset(raw_df)
    features, target = df.drop(columns=[TARGET]), df[TARGET]
    pipeline = build_pipeline(model_name, {}, seed=42).fit(features, target)
    proba = pipeline.predict_proba(features)[:, 1]
    assert proba.shape == (len(df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_build_pipeline_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        build_pipeline("svm", {}, seed=42)


def test_compute_metrics_perfect_classifier() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert set(metrics) == {"roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"}


def test_save_evaluation_plots_writes_pngs(tmp_path) -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_proba = np.array([0.2, 0.7, 0.4, 0.9, 0.3])
    paths = save_evaluation_plots(y_true, y_proba, threshold=0.5, output_dir=tmp_path)
    assert all(path.exists() and path.suffix == ".png" for path in paths)
