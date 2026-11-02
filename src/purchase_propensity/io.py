"""Leitura e escrita de artefatos JSON trocados entre os estágios do pipeline."""

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: Path) -> None:
    """Grava um dicionário como JSON indentado, criando a pasta se preciso."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline fixo: o hash do DVC precisa ser igual no Windows e no container Linux
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(content, encoding="utf-8", newline="\n")


def load_json(path: Path) -> dict[str, Any]:
    """Lê um arquivo JSON como dicionário."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
