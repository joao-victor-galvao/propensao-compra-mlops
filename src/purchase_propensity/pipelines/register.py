"""Estágio `register` do DVC: promove o melhor modelo no Registry e o exporta para a API."""

import logging
from typing import Any

from mlflow.entities.model_registry import ModelVersion

from purchase_propensity.config import RegistryParams, Settings, load_params
from purchase_propensity.io import load_json, save_json
from purchase_propensity.logging_config import configure_logging
from purchase_propensity.models.registry import (
    Candidate,
    ChampionRegistry,
    select_best_candidate,
)
from purchase_propensity.models.tracking import configure_tracking

logger = logging.getLogger(__name__)


def build_champion_report(
    params: RegistryParams, champion: Candidate, version: ModelVersion
) -> dict[str, Any]:
    """Resume qual modelo foi promovido, em que versão e com qual score."""
    return {
        "model_name": params.model_name,
        "version": version.version,
        "alias": params.alias,
        "model_type": champion.model_name,
        "run_id": champion.run_id,
        params.selection_metric: champion.score,
    }


def main() -> None:
    """Seleciona, registra, aplica o alias e exporta o modelo campeão."""
    configure_logging()
    params = load_params()
    configure_tracking(Settings())
    results = load_json(params.evaluate.metrics_path)
    champion = select_best_candidate(results, params.registry.selection_metric)
    registry = ChampionRegistry(params.registry)
    version = registry.promote(champion)
    export_dir = registry.export()
    report = build_champion_report(params.registry, champion, version)
    save_json(report, params.registry.report_path)
    logger.info("Campeão promovido e exportado para %s: %s", export_dir, report)


if __name__ == "__main__":
    main()
