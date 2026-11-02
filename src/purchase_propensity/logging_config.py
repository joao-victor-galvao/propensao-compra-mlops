"""Configuração única de logging para os entrypoints."""

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Aplica formato e nível padrão aos logs da aplicação."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
