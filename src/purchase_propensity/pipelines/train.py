"""Estágio `train` do DVC: treina cada modelo candidato e registra as runs no MLflow."""

import logging

import numpy as np

from purchase_propensity.config import Settings, load_params
from purchase_propensity.data.loader import load_features_and_target
from purchase_propensity.io import save_json
from purchase_propensity.logging_config import configure_logging
from purchase_propensity.models.tracking import configure_tracking, read_data_version
from purchase_propensity.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def main() -> None:
    """Treina os candidatos do params.yaml e salva ``modelo -> run_id``."""
    configure_logging()
    params = load_params()
    np.random.seed(params.seed)
    configure_tracking(Settings())
    features, target = load_features_and_target(params.data.train_path)
    trainer = ModelTrainer(
        seed=params.seed,
        cv_folds=params.train.cv_folds,
        tags={"data_version": read_data_version(params.data.raw_path)},
    )
    runs: dict[str, str] = {}
    for model_name, hyperparams in params.train.models.items():
        result = trainer.train(model_name, hyperparams, features, target)
        runs[model_name] = result.run_id
        logger.info("%s | run %s | %s", model_name, result.run_id, result.cv_metrics)
    save_json(runs, params.train.runs_path)


if __name__ == "__main__":
    main()
