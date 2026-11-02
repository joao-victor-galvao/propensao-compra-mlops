"""Configuração única de logging para os entrypoints."""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Aplica formato e nível padrão aos logs da aplicação.

    Também força UTF-8 na saída padrão: no Windows o console usa cp1252 e
    quebra com os emojis que o MLflow imprime ao finalizar uma run.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=level, format=LOG_FORMAT)
