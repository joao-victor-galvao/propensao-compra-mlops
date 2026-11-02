"""Estágio `preprocess` do DVC: valida, limpa e divide o dataset bruto."""

import logging

from purchase_propensity.config import load_params
from purchase_propensity.data.loader import load_raw, save_dataset
from purchase_propensity.data.preparation import clean_dataset, split_dataset
from purchase_propensity.data.schema import TARGET
from purchase_propensity.data.validation import validate_raw
from purchase_propensity.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Executa o pré-processamento e grava os splits de treino e teste."""
    configure_logging()
    params = load_params()
    raw = load_raw(params.data.raw_path)
    validate_raw(raw)
    cleaned = clean_dataset(raw)
    train, test = split_dataset(cleaned, params.data.test_size, params.seed)
    save_dataset(train, params.data.train_path)
    save_dataset(test, params.data.test_path)
    logger.info(
        "Bruto: %d linhas | limpo: %d | treino: %d | teste: %d | taxa de compra: %.3f",
        len(raw),
        len(cleaned),
        len(train),
        len(test),
        cleaned[TARGET].mean(),
    )


if __name__ == "__main__":
    main()
